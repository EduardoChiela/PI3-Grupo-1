"""
Selecao de nodulos - LIDC-IDRI (etapa "Selecao" do KDD)
=========================================================

Objetivo
--------
Este script varre o banco de anotacoes do LIDC-IDRI (via biblioteca pylidc)
e produz um dataset tabular de nodulos pulmonares elegiveis para as
proximas etapas do pipeline (pre-processamento / extracao de patches),
salvando o resultado em `selecao/nodulos_selecionados.csv`.

Apenas o banco de anotacoes (metadados + contornos + escores dos
radiologistas, que o pylidc mantem em um SQLite proprio) e utilizado.
Nenhuma imagem DICOM e carregada (nao usamos `scan.load_all_dicom_images()`),
pois nesta etapa so precisamos decidir QUAIS nodulos entram no dataset,
nao processar seus pixels.

Criterios de selecao
---------------------
1. Exame (Scan): mantido apenas se `slice_thickness <= 2.5mm`. Cortes mais
   grossos degradam a resolucao espacial ao longo do eixo Z e sao
   tradicionalmente descartados em trabalhos com o LIDC-IDRI.

2. Nodulo (cluster de anotacoes): as anotacoes de um mesmo exame sao
   agrupadas por nodulo fisico com `scan.cluster_annotations()` (metodo
   nativo do pylidc que agrupa anotacoes de radiologistas distintos que
   se referem à mesma estrutura, com base na proximidade espacial dos
   contornos). So mantemos clusters com 3 ou mais anotacoes, ou seja,
   nodulos em que pelo menos 3 radiologistas concordaram que ali existe
   um nodulo (o LIDC-IDRI tem ate 4 radiologistas por exame).

3. Rotulo binario de malignidade: calculamos a mediana do campo
   `malignancy` (escala 1-5, escore subjetivo de cada radiologista) entre
   as anotacoes do cluster.
     - mediana < 3 -> "benigno"
     - mediana > 3 -> "maligno"
     - mediana == 3 -> indeterminado -> nodulo DESCARTADO do dataset final
   Esse e um criterio comum na literatura que usa o LIDC-IDRI (ex.: em
   varios trabalhos de deteccao/classificacao de nodulos), pois o escore
   3 representa "nem benigno nem maligno" na escala do radiologista, uma
   regiao cinzenta que atrapalha o treinamento de classificadores binarios.

4. Diametro do nodulo: cada anotacao individual carrega sua propria
   estimativa de diametro (`annotation.diameter`, calculada a partir dos
   contornos 2D anotados por aquele radiologista). Um mesmo nodulo pode
   ter diametros levemente diferentes entre radiologistas, pois cada um
   desenhou o contorno de forma independente. Optamos por reportar a
   MEDIA dos diametros das anotacoes do cluster (em vez do diametro da
   anotacao "mais confiavel", conceito que o pylidc nao define de forma
   objetiva) porque a media suaviza o ruido de contorno entre observadores
   e e a abordagem mais usada na literatura para consolidar um unico
   valor de tamanho por nodulo a partir de multiplas anotacoes.

Saida
-----
`selecao/nodulos_selecionados.csv`, uma linha por nodulo elegivel, com as
colunas: patient_id, scan_id, n_radiologistas, escores_individuais,
malignidade_mediana, rotulo_binario, diametro_mm.
"""

import csv
import statistics
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Compatibilidade: o pylidc 0.2.3 ainda usa os aliases `np.int`/`np.float`/
# `np.bool`, removidos em versoes recentes do NumPy (>=1.24). Sem este
# "shim", chamadas como `annotation.diameter` ou `scan.cluster_annotations()`
# quebram com `AttributeError: module 'numpy' has no attribute 'int'`.
# Aplicamos o patch antes de importar o pylidc, restaurando os aliases
# antigos como sinonimos dos tipos nativos do Python, sem alterar nenhum
# outro comportamento do NumPy.
# --------------------------------------------------------------------------
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

import pylidc as pl

# Numero minimo de radiologistas que precisam concordar sobre um nodulo
# para que ele seja considerado "real" o suficiente para entrar no dataset.
MIN_RADIOLOGISTAS = 3

# Espessura maxima de corte (em mm) aceita para um exame entrar na selecao.
SLICE_THICKNESS_MAX = 2.5

CSV_SAIDA = Path(__file__).parent / "nodulos_selecionados.csv"


def main():
    # ----------------------------------------------------------------
    # 1) Conectar ao banco de anotacoes do pylidc e checar a contagem
    #    total de exames. O LIDC-IDRI completo tem 1018 exames (Scans);
    #    se esse numero divergir, o banco de anotacoes local do pylidc
    #    esta incompleto ou mal configurado.
    # ----------------------------------------------------------------
    total_scans = pl.query(pl.Scan).count()
    print(f"[checagem] Total de exames (Scans) no banco pylidc: {total_scans}")
    if total_scans != 1018:
        print(
            "[aviso] O total de exames encontrado difere do esperado (1018). "
            "Verifique a configuracao/versao do banco de anotacoes do pylidc."
        )

    # ----------------------------------------------------------------
    # 2) Filtrar exames por espessura de corte (slice_thickness <= 2.5mm).
    #    Fazemos o filtro diretamente na query do pylidc, que ja traduz
    #    isso em SQL, evitando carregar exames que nem interessam.
    # ----------------------------------------------------------------
    scans_filtrados = pl.query(pl.Scan).filter(
        pl.Scan.slice_thickness <= SLICE_THICKNESS_MAX
    ).all()
    n_scans_filtrados = len(scans_filtrados)
    print(
        f"[filtro] Exames com slice_thickness <= {SLICE_THICKNESS_MAX}mm: "
        f"{n_scans_filtrados}"
    )

    # Contadores para o resumo final.
    total_clusters = 0
    excluidos_poucos_radiologistas = 0
    excluidos_indeterminados = 0
    n_benignos = 0
    n_malignos = 0
    linhas_csv = []

    # ----------------------------------------------------------------
    # 3) Para cada exame filtrado, agrupar as anotacoes por nodulo fisico.
    #    Processamos exame por exame com try/except: um exame problematico
    #    (ex.: contornos inconsistentes que travam o algoritmo de cluster)
    #    e logado e pulado, sem interromper o processamento dos demais.
    # ----------------------------------------------------------------
    for scan in scans_filtrados:
        try:
            # scan.cluster_annotations() retorna uma lista de listas de
            # Annotation, onde cada lista interna corresponde a um unico
            # nodulo fisico (anotado por 1 a 4 radiologistas distintos).
            clusters = scan.cluster_annotations(verbose=False)
        except Exception as erro:
            print(
                f"[erro] Falha ao agrupar anotacoes do scan id={scan.id} "
                f"(patient_id={scan.patient_id}): {erro!r}. Pulando este exame."
            )
            continue

        total_clusters += len(clusters)

        # ------------------------------------------------------------
        # 4) Manter apenas clusters com 3+ anotacoes (radiologistas).
        # ------------------------------------------------------------
        for cluster_idx, cluster in enumerate(clusters):
            try:
                n_radiologistas = len(cluster)
                if n_radiologistas < MIN_RADIOLOGISTAS:
                    excluidos_poucos_radiologistas += 1
                    continue

                # --------------------------------------------------------
                # 5) Mediana de malignidade -> rotulo binario (ou descarte
                #    caso a mediana seja exatamente 3, o caso indeterminado).
                # --------------------------------------------------------
                escores = [int(anot.malignancy) for anot in cluster]
                mediana_malignidade = statistics.median(escores)

                if mediana_malignidade == 3:
                    excluidos_indeterminados += 1
                    continue
                elif mediana_malignidade < 3:
                    rotulo = "benigno"
                    n_benignos += 1
                else:
                    rotulo = "maligno"
                    n_malignos += 1

                # Diametro do nodulo: media dos diametros individuais das
                # anotacoes do cluster (ver justificativa no docstring do
                # topo do arquivo).
                diametro_mm = statistics.mean(
                    float(anot.diameter) for anot in cluster
                )

                linhas_csv.append(
                    {
                        "nodule_id": f"{scan.patient_id}_scan{scan.id}_cluster{cluster_idx:03d}",
                        "cluster_idx": cluster_idx,
                        "patient_id": scan.patient_id,
                        "scan_id": scan.id,
                        "n_radiologistas": n_radiologistas,
                        "escores_individuais": ",".join(str(e) for e in escores),
                        "malignidade_mediana": mediana_malignidade,
                        "rotulo_binario": rotulo,
                        "diametro_mm": round(diametro_mm, 2),
                    }
                )
            except Exception as erro:
                print(
                    f"[erro] Falha ao processar um cluster do scan id={scan.id} "
                    f"(patient_id={scan.patient_id}): {erro!r}. Pulando este nodulo."
                )
                continue

    # ----------------------------------------------------------------
    # 6) Salvar o CSV final.
    # ----------------------------------------------------------------
    colunas = [
        "nodule_id",
        "cluster_idx",
        "patient_id",
        "scan_id",
        "n_radiologistas",
        "escores_individuais",
        "malignidade_mediana",
        "rotulo_binario",
        "diametro_mm",
    ]
    with open(CSV_SAIDA, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(linhas_csv)

    # ----------------------------------------------------------------
    # 7) Resumo final do processamento.
    # ----------------------------------------------------------------
    print()
    print("========== RESUMO DA SELECAO ==========")
    print(f"Exames apos filtro de espessura (<= {SLICE_THICKNESS_MAX}mm): {n_scans_filtrados}")
    print(f"Clusters (nodulos candidatos) formados: {total_clusters}")
    print(f"Excluidos por ter menos de {MIN_RADIOLOGISTAS} radiologistas: {excluidos_poucos_radiologistas}")
    print(f"Excluidos por malignidade indeterminada (mediana == 3): {excluidos_indeterminados}")
    print(f"Nodulos no dataset final: {len(linhas_csv)}")
    print(f"  - benignos: {n_benignos}")
    print(f"  - malignos: {n_malignos}")
    print(f"CSV salvo em: {CSV_SAIDA}")
    print("========================================")


if __name__ == "__main__":
    main()
