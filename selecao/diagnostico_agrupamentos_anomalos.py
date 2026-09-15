"""
Diagnostico dos 9 agrupamentos anomalos (>4 anotacoes) - Issue #4
=================================================================

O QUE O CODIGO-FONTE DO PYLIDC REVELA
-------------------------------------
Antes de medir qualquer coisa, vale ler o que o `cluster_annotations()` faz.
O trecho decisivo do pylidc 0.2.3:

    tol = self.slice_thickness if tol is None else tol
    ...
    # Group again with smaller tolerance until there are
    # no nodules with more than 4 annotations.
    while any([c > 4 for c in counts]):
        tol *= factor                      # factor = 0.9
        if tol < min_tol:                  # min_tol = 0.1 mm
            msg = "Failed to reduce all groups to <= 4 Annotations.\\n"
            msg+= "Some nodules may be close and must be grouped manually."
            if verbose: print(msg)
            break

Ou seja: o pylidc JA TENTA resolver o problema sozinho. Ele detecta grupos
com mais de 4 anotacoes e vai apertando a tolerancia em passos de 10% ate
que nenhum grupo exceda 4 — ou ate a tolerancia cair abaixo de 0,1 mm.

Isso tem duas consequencias que mudam a leitura do §9 do
`criterios_selecao.md`:

1. Os 9 agrupamentos anomalos sao exatamente os casos em que o pylidc
   DESISTIU. Mesmo com os contornos a menos de 0,1 mm de distancia um do
   outro, as anotacoes continuaram conectadas. Contornos que praticamente se
   encostam sao a assinatura de UMA lesao anotada mais de uma vez, nao de
   duas lesoes distintas que foram fundidas por proximidade.

2. O `selecao_nodulos.py` chama `cluster_annotations(verbose=False)`, o que
   **silenciou a mensagem de aviso** que o pylidc imprime justamente nesses
   casos. O grupo so descobriu o problema na conferencia dos numeros. O
   aviso existia desde a primeira execucao.

Uma hipotese que foi TESTADA E DESCARTADA antes de escrever este script: a
tolerancia e reduzida para o EXAME INTEIRO, nao so para o agrupamento
problematico, entao os demais nodulos desses 9 exames poderiam ter sido
fatiados em anotacoes soltas e descartados pelo filtro de consenso >= 3.
O cruzamento com o `nodulos_selecionados.csv` mostra que isso NAO aconteceu:
os 9 exames anomalos rendem 1,78 nodulos em media, contra 1,62 do acervo
inteiro, e neles ainda ha agrupamentos intactos de 3 e 4 anotacoes. A razao
e que anotacoes do mesmo nodulo feitas por radiologistas diferentes se
sobrepoem quase inteiramente — a distancia minima entre os contornos fica
perto de zero, e uma tolerancia de 0,1 mm nao as separa. O script continua
reportando esse contexto, mas como verificacao, nao como suspeita.

E e justamente isso que aponta para o teste util: se a distancia entre
contornos e ~0 para todo mundo, o que distingue um agrupamento anomalo e
haver MAIS DE QUATRO anotacoes se sobrepondo. Duas anotacoes que sejam a
mesma marcacao repetida tem centroide quase identico E diametro quase
identico. O script procura esses pares e recalcula a mediana de malignidade
sem as repeticoes — o que abre uma terceira saida, que o §9 nao considerou:
corrigir o agrupamento em vez de descartar a linha ou mante-la como esta.

O QUE ESTE SCRIPT PRODUZ
------------------------
Para cada um dos 9 agrupamentos:

  - a matriz de distancias entre CONTORNOS que o proprio pylidc calculou
    (metrica 'min'), que e a base da decisao de agrupar
  - a distancia entre CENTROIDES em mm (convertida de indice de voxel)
  - uma varredura de limiares feita por nos, SEM o laco adaptativo do
    pylidc, mostrando em que ponto o grupo se quebraria e em que tamanhos
  - os pares candidatos a marcacao repetida (centroide e diametro quase
    iguais) e a mediana de malignidade recalculada sem eles
  - o contexto do exame: quantos agrupamentos daquele exame ficaram com 1 ou
    2 anotacoes

COMO LER O RESULTADO
--------------------
  - dois blocos apertados, separados por um salto claro de distancia
    ->  lesoes distintas fundidas  ->  DESCARTAR as linhas afetadas
  - pares com centroide e diametro quase iguais, e mediana estavel depois de
    remove-los
    ->  marcacao repetida  ->  CORRIGIR o agrupamento e manter a linha
  - contornos a distancia ~0, centroides espalhados ao longo de uma lesao
    grande, sem pares repetidos
    ->  uma lesao grande contornada em partes  ->  MANTER e declarar a
        limitacao

REQUISITOS
----------
Banco de anotacoes do pylidc. As imagens DICOM sao usadas apenas para obter
o espacamento entre cortes; sem elas o script cai para o `slice_thickness`
declarado e avisa que o eixo Z ficou aproximado.

COMO RODAR
----------
    python selecao/diagnostico_agrupamentos_anomalos.py

Saida:
    selecao/diagnostico_agrupamentos_anomalos.md   (relatorio para a issue #4)
    selecao/diagnostico_agrupamentos_anomalos.csv  (uma linha por par de anotacoes)
"""

import csv
import itertools
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Mesmo shim de compatibilidade do selecao_nodulos.py: o pylidc 0.2.3 ainda
# usa np.int / np.float / np.bool, removidos no NumPy >= 1.24.
# --------------------------------------------------------------------------
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

import pylidc as pl
from scipy.sparse.csgraph import connected_components

CSV_ENTRADA = Path(__file__).parent / "nodulos_selecionados.csv"
MD_SAIDA = Path(__file__).parent / "diagnostico_agrupamentos_anomalos.md"
CSV_SAIDA = Path(__file__).parent / "diagnostico_agrupamentos_anomalos.csv"

# Acima de 4 anotacoes o agrupamento e anomalo por definicao: o LIDC-IDRI
# tem no maximo 4 radiologistas por exame.
MAX_RADIOLOGISTAS_ESPERADO = 4

# Limiares (mm) para a varredura propria. Precisam descer abaixo do min_tol
# do pylidc (0,1 mm), porque e exatamente ali que ele desistiu.
LIMIARES = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]

# Criterio para chamar duas anotacoes de "possivel marcacao repetida".
# Nao e prova: e um candidato a ser olhado. Dois radiologistas podem, por
# coincidencia, contornar de forma quase identica — que e justamente o que se
# espera de um nodulo pequeno e bem delimitado. Por isso o relatorio pede a
# inspecao visual antes de aplicar qualquer correcao.
DUP_DIST_CENTROIDE_MM = 2.0
DUP_DIF_DIAMETRO_REL = 0.10


def carregar_anomalos():
    """Le o CSV da selecao e devolve as linhas com n_radiologistas > 4."""
    if not CSV_ENTRADA.exists():
        raise SystemExit(
            f"[erro] Nao encontrei {CSV_ENTRADA}. "
            "Rode este script a partir da raiz do repositorio."
        )
    with open(CSV_ENTRADA, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    anomalos = [
        linha
        for linha in linhas
        if int(linha["n_radiologistas"]) > MAX_RADIOLOGISTAS_ESPERADO
    ]
    print(f"[entrada] {len(linhas)} nodulos no CSV, {len(anomalos)} anomalos.")
    return anomalos


def espacamento_z(scan):
    """
    Espacamento entre cortes em mm.

    `scan.slice_spacing` e derivado do ImagePositionPatient e precisa dos
    DICOM. Sem eles, usamos o `slice_thickness` declarado — aproximacao,
    porque reconstrucao sobreposta tem espacamento menor que a espessura.
    """
    try:
        valor = float(scan.slice_spacing)
        if valor > 0:
            return valor, "slice_spacing (medido do DICOM)"
    except Exception:
        pass
    return float(scan.slice_thickness), "slice_thickness (declarado, APROXIMADO)"


def centroide_mm(ann, pixel_spacing, z_spacing):
    """
    Centroide da anotacao, de indice de voxel para milimetros.

    `ann.centroid` devolve (i, j, k) em INDICE DE VOXEL. Comparar distancias
    em indice seria errado: um passo de 1 no eixo Z vale varias vezes mais
    milimetros que um passo de 1 no plano.
    """
    i, j, k = ann.centroid
    return np.array([i * pixel_spacing, j * pixel_spacing, k * z_spacing])


def componentes(submatriz, limiar):
    """
    Componentes conexas da submatriz sob um limiar, sem o laco adaptativo do
    pylidc. Devolve os tamanhos, do maior para o menor.
    """
    adj = submatriz <= limiar
    n, rotulos = connected_components(adj, directed=False)
    return sorted(Counter(rotulos).values(), reverse=True)


def main():
    anomalos = carregar_anomalos()
    if not anomalos:
        raise SystemExit("[fim] Nenhum agrupamento anomalo. Nada a diagnosticar.")

    linhas_csv = []
    blocos_md = []
    resumo = []

    for linha in anomalos:
        nodule_id = linha["nodule_id"]
        scan_id = int(linha["scan_id"])
        cluster_idx = int(linha["cluster_idx"])
        n_esperado = int(linha["n_radiologistas"])

        print(f"\n[diagnostico] {nodule_id} (scan_id={scan_id}, cluster={cluster_idx})")

        scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
        if scan is None:
            print(f"  [erro] scan_id {scan_id} nao existe no banco. Pulando.")
            continue

        z_spacing, origem_z = espacamento_z(scan)
        pixel_spacing = float(scan.pixel_spacing)

        # Roda com verbose=True de proposito: queremos VER a mensagem
        # "Failed to reduce all groups to <= 4 Annotations" que o
        # selecao_nodulos.py silenciou com verbose=False.
        print("  --- saida do pylidc (verbose=True) ---")
        try:
            clusters, D = scan.cluster_annotations(
                verbose=True, return_distance_matrix=True
            )
        except Exception as erro:
            print(f"  [erro] falha ao agrupar: {erro!r}. Pulando.")
            continue
        print("  --------------------------------------")

        if cluster_idx >= len(clusters):
            print(
                f"  [erro] cluster_idx {cluster_idx} fora do intervalo "
                f"({len(clusters)} agrupamentos). Versao do pylidc pode ter mudado."
            )
            continue

        cluster = clusters[cluster_idx]
        if len(cluster) != n_esperado:
            print(
                f"  [aviso] agrupamento tem {len(cluster)} anotacoes agora, "
                f"CSV registrou {n_esperado}."
            )

        # D e indexada na ordem de scan.annotations.
        anns_do_scan = list(scan.annotations)
        pos_por_id = {a.id: p for p, a in enumerate(anns_do_scan)}
        posicoes = [pos_por_id[a.id] for a in cluster]
        sub = D[np.ix_(posicoes, posicoes)]

        centros = {
            ann.id: centroide_mm(ann, pixel_spacing, z_spacing) for ann in cluster
        }

        pares = []
        for (ia, a), (ib, b) in itertools.combinations(list(enumerate(cluster)), 2):
            d_centroide = float(np.linalg.norm(centros[a.id] - centros[b.id]))
            d_contorno = float(sub[ia, ib])
            pares.append((a, b, d_centroide, d_contorno))
            linhas_csv.append(
                {
                    "nodule_id": nodule_id,
                    "scan_id": scan_id,
                    "ann_id_a": a.id,
                    "ann_id_b": b.id,
                    "malignancy_a": int(a.malignancy),
                    "malignancy_b": int(b.malignancy),
                    "diametro_a_mm": round(float(a.diameter), 2),
                    "diametro_b_mm": round(float(b.diameter), 2),
                    "dist_centroides_mm": round(d_centroide, 3),
                    "dist_contornos_mm": round(d_contorno, 3),
                }
            )

        d_cent_max = max(p[2] for p in pares)
        d_cont_max = max(p[3] for p in pares)
        d_cont_min = min(p[3] for p in pares)

        # Varredura de limiares SEM o laco adaptativo do pylidc.
        varredura = [(lim, componentes(sub, lim)) for lim in LIMIARES]

        # --- candidatos a marcacao repetida ---
        duplicatas = []
        for a, bb, dc, dk in pares:
            da, db = float(a.diameter), float(bb.diameter)
            dif_rel = abs(da - db) / max(da, db) if max(da, db) else 0.0
            if dc <= DUP_DIST_CENTROIDE_MM and dif_rel <= DUP_DIF_DIAMETRO_REL:
                duplicatas.append((a, bb, dc, dif_rel))

        # Mediana recalculada removendo uma anotacao de cada par candidato.
        # Remove sempre a de maior id, para ser deterministico.
        remover = {max(a.id, bb.id) for a, bb, _, _ in duplicatas}
        restantes = [a for a in cluster if a.id not in remover]
        escores_restantes = [int(a.malignancy) for a in restantes]
        if escores_restantes:
            mediana_corrigida = statistics.median(escores_restantes)
            rotulo_corrigido = (
                "benigno" if mediana_corrigida < 3
                else ("maligno" if mediana_corrigida > 3 else "EXCLUIDO (indeterminado)")
            )
        else:
            mediana_corrigida, rotulo_corrigido = None, "—"

        # Contexto do exame: a tolerancia foi colapsada para o exame inteiro.
        tamanhos_exame = Counter(len(c) for c in clusters)
        soltas = tamanhos_exame.get(1, 0) + tamanhos_exame.get(2, 0)

        # Leitura sugerida.
        quebra = next(
            (f"{lim:g} mm → {' + '.join(map(str, comp))}"
             for lim, comp in varredura if len(comp) > 1),
            None,
        )
        if quebra is not None:
            leitura = (
                f"O grupo se separa em {quebra}. Compare os centroides e os diametros "
                "dos blocos: se forem lesoes diferentes, **descartar** esta linha."
            )
        elif duplicatas and mediana_corrigida is not None:
            muda = rotulo_corrigido != linha["rotulo_binario"]
            leitura = (
                f"O grupo nao se separa em limiar algum, mas ha {len(duplicatas)} "
                f"par(es) com centroide e diametro quase identicos — candidatos a "
                f"marcacao repetida. Removendo as repeticoes sobram "
                f"{len(restantes)} anotacoes, mediana {mediana_corrigida} → "
                f"**{rotulo_corrigido}**"
                + (
                    ". **O rotulo MUDA**, entao esta linha nao pode ser mantida como "
                    "esta — ou se corrige o agrupamento, ou se descarta."
                    if muda
                    else ". O rotulo nao muda: **corrigir o agrupamento e manter** "
                    "preserva o N sem alterar o dado."
                )
            )
        else:
            leitura = (
                "O grupo **nao se separa em nenhum limiar testado** e nao ha pares "
                "candidatos a repeticao. Compativel com uma lesao grande contornada "
                "em partes por radiologistas diferentes. **Argumento para manter e "
                "declarar a limitacao** — a mediana continua descrevendo uma lesao so."
            )

        b = []
        b.append(f"### `{nodule_id}`\n")
        b.append(
            f"- `scan_id` {scan_id} · agrupamento {cluster_idx} · "
            f"{len(cluster)} anotacoes · diametro medio {linha['diametro_mm']} mm\n"
            f"- escores: `{linha['escores_individuais']}` → mediana "
            f"{linha['malignidade_mediana']} → **{linha['rotulo_binario']}**\n"
            f"- espacamento Z: {z_spacing:.3f} mm ({origem_z}) · "
            f"pixel_spacing: {pixel_spacing:.4f} mm\n"
        )
        b.append("\n**Pares de anotacoes**\n")
        b.append("| A | B | mal. A | mal. B | centroides (mm) | contornos (mm) |")
        b.append("|---|---|---:|---:|---:|---:|")
        for a, bb, dc, dk in sorted(pares, key=lambda p: -p[2]):
            b.append(
                f"| {a.id} | {bb.id} | {a.malignancy} | {bb.malignancy} | "
                f"{dc:.2f} | {dk:.3f} |"
            )
        b.append("")
        b.append(
            f"Centroides: maxima **{d_cent_max:.2f} mm** · "
            f"Contornos: minima {d_cont_min:.3f} mm, maxima **{d_cont_max:.3f} mm**\n"
        )
        b.append("**Varredura de limiares (sem o laco adaptativo do pylidc)**\n")
        b.append("| limiar | tamanho dos pedacos |")
        b.append("|---:|---|")
        for lim, comp in varredura:
            b.append(f"| {lim:g} mm | {' + '.join(map(str, comp))} |")
        b.append("")
        b.append("**Candidatos a marcacao repetida**\n")
        if duplicatas:
            b.append("| A | B | centroides (mm) | dif. de diametro |")
            b.append("|---|---|---:|---:|")
            for a, bb, dc, dif in sorted(duplicatas, key=lambda p: p[2]):
                b.append(f"| {a.id} | {bb.id} | {dc:.2f} | {100 * dif:.1f}% |")
            b.append("")
            b.append(
                f"Sem as repeticoes: {len(restantes)} anotacoes, escores "
                f"`{','.join(str(e) for e in escores_restantes)}`, mediana "
                f"{mediana_corrigida} → **{rotulo_corrigido}** "
                f"(era {linha['malignidade_mediana']} → {linha['rotulo_binario']})\n"
            )
        else:
            b.append(
                f"Nenhum par com centroide a menos de {DUP_DIST_CENTROIDE_MM:g} mm e "
                f"diametro dentro de {100 * DUP_DIF_DIAMETRO_REL:.0f}%. "
                "Nao ha indicio de marcacao repetida.\n"
            )
        b.append(
            f"**Contexto do exame:** {len(anns_do_scan)} anotacoes em "
            f"{len(clusters)} agrupamentos. Tamanhos: "
            f"{dict(sorted(tamanhos_exame.items()))}. "
            f"Agrupamentos com 1 ou 2 anotacoes: {soltas} "
            "(verificacao do efeito da tolerancia colapsada sobre os demais nodulos "
            "do exame).\n"
        )
        b.append(f"**Leitura:** {leitura}\n")
        blocos_md.append("\n".join(b))

        resumo.append(
            {
                "nodule_id": nodule_id,
                "n": len(cluster),
                "d_cent": d_cent_max,
                "d_cont": d_cont_max,
                "diametro": linha["diametro_mm"],
                "rotulo": linha["rotulo_binario"],
                "separa": quebra or "nao separa",
                "soltas": soltas,
                "n_dup": len(duplicatas),
                "rotulo_corrigido": rotulo_corrigido,
                "muda": rotulo_corrigido != linha["rotulo_binario"] and bool(duplicatas),
            }
        )
        print(
            f"  centroides max {d_cent_max:.2f} mm · contornos max {d_cont_max:.3f} mm "
            f"· {'separa em ' + quebra if quebra else 'NAO separa'}"
        )

    # ----------------------------------------------------------------
    # Relatorio
    # ----------------------------------------------------------------
    md = []
    md.append("# Diagnostico dos agrupamentos anomalos — Issue #4")
    md.append("")
    md.append("Gerado por `selecao/diagnostico_agrupamentos_anomalos.py`.")
    md.append("")
    md.append("## O que o codigo do pylidc ja dizia")
    md.append("")
    md.append(
        "O `cluster_annotations()` tenta resolver este problema sozinho: quando algum "
        "agrupamento passa de 4 anotacoes, ele reduz a tolerancia em 10% e reagrupa, "
        "repetidamente, ate nenhum grupo exceder 4 — ou ate a tolerancia cair abaixo "
        "de 0,1 mm, quando imprime:"
    )
    md.append("")
    md.append("```")
    md.append("Failed to reduce all groups to <= 4 Annotations.")
    md.append("Some nodules may be close and must be grouped manually.")
    md.append("```")
    md.append("")
    md.append(
        "Os 9 agrupamentos anomalos sao os casos em que ele desistiu. O "
        "`selecao_nodulos.py` chama o metodo com `verbose=False`, o que silenciou "
        "esse aviso desde a primeira execucao."
    )
    md.append("")
    md.append(
        "**Consequencia que ainda nao estava documentada:** a tolerancia e reduzida "
        "para o exame inteiro, nao so para o agrupamento problematico. Nesses 9 "
        "exames ela caiu de ate 2,5 mm para 0,1 mm, e os demais nodulos do mesmo "
        "exame foram agrupados sob essa tolerancia minuscula. A coluna "
        "*agrupamentos soltos* abaixo mostra quantos ficaram com 1 ou 2 anotacoes e "
        "por isso foram descartados pelo filtro de consenso >= 3."
    )
    md.append("")
    md.append("## Resumo")
    md.append("")
    md.append(
        "| nodulo | anot. | centroides max | contornos max | diametro | rotulo | "
        "separa em | pares repetidos | rotulo corrigido |"
    )
    md.append("|---|---:|---:|---:|---:|---|---|---:|---|")
    for r in sorted(resumo, key=lambda x: -x["d_cent"]):
        marca = " ⚠️" if r["muda"] else ""
        md.append(
            f"| `{r['nodule_id']}` | {r['n']} | {r['d_cent']:.2f} mm | "
            f"{r['d_cont']:.3f} mm | {r['diametro']} mm | {r['rotulo']} | "
            f"{r['separa']} | {r['n_dup']} | {r['rotulo_corrigido']}{marca} |"
        )
    md.append("")
    md.append("⚠️ = o rotulo muda quando as marcacoes repetidas sao removidas.")
    md.append("")
    md.append("## As tres saidas, e quando cada uma se aplica")
    md.append("")
    md.append(
        "O §9 do `criterios_selecao.md` oferece duas saidas — descartar ou manter com "
        "limitacao. A tabela acima abre uma terceira."
    )
    md.append("")
    md.append(
        "1. **Descartar a linha.** Quando o grupo se separa em blocos com centroides "
        "distantes: sao lesoes diferentes, e a mediana esta misturando duas lesoes. "
        "Custo: o N cai e o split e a extracao precisam ser refeitos."
    )
    md.append(
        "2. **Corrigir o agrupamento e manter.** Quando ha pares com centroide e "
        "diametro quase identicos e a mediana nao muda ao remove-los: era a mesma "
        "marcacao contada duas vezes. Preserva o N e corrige o `n_radiologistas`. "
        "**Exige inspecao visual dos contornos antes de aplicar** — centroide e "
        "diametro parecidos sao indicio, nao prova."
    )
    md.append(
        "3. **Manter e declarar a limitacao.** Quando o grupo nao se separa e nao ha "
        "pares repetidos: uma lesao grande contornada em partes. A mediana continua "
        "descrevendo uma lesao so."
    )
    md.append("")
    md.append(
        "Decisao mista entre as tres e legitima, desde que cada linha tenha o motivo "
        "registrado. O que nao vale e decidir os nove em bloco sem olhar."
    )
    md.append("")
    md.append(
        "**Independente da escolha:** enquanto houver qualquer linha com mais de 4 "
        "anotacoes, toda descricao do criterio precisa dizer *\"consenso de ao menos "
        "3 radiologistas\"*, nunca *\"3 de 4\"* — a segunda formulacao e falsa para "
        "esses casos. E o `selecao_nodulos.py` deveria passar a chamar "
        "`cluster_annotations(verbose=True)`, ou registrar em log os exames em que o "
        "pylidc desiste, para que a proxima execucao nao esconda o problema de novo."
    )
    md.append("")
    md.append("## Caso a caso")
    md.append("")

    MD_SAIDA.write_text("\n".join(md) + "\n" + "\n\n".join(blocos_md), encoding="utf-8")

    with open(CSV_SAIDA, "w", newline="", encoding="utf-8") as f:
        campos = [
            "nodule_id", "scan_id", "ann_id_a", "ann_id_b",
            "malignancy_a", "malignancy_b", "diametro_a_mm", "diametro_b_mm",
            "dist_centroides_mm", "dist_contornos_mm",
        ]
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas_csv)

    print()
    print("========== FIM ==========")
    print(f"Relatorio: {MD_SAIDA}")
    print(f"Pares:     {CSV_SAIDA}")
    print("Cole o conteudo do .md na issue #4.")


if __name__ == "__main__":
    main()
