"""
Verificacao do criterio §3.2 - espacamento entre cortes - Issue #4
==================================================================

O PROBLEMA QUE ESTE SCRIPT RESOLVE
----------------------------------
O `docs/criterios_selecao.md`, secao 3.2, afirmava que "series com cortes
ausentes ou espacamento irregular foram sinalizadas". O
`selecao/selecao_nodulos.py` nao tem nenhuma linha que calcule espacamento -
ele filtra apenas `slice_thickness <= 2.5`. Este script produz o numero que
faltava.

NAO PRECISA DO ACERVO DICOM
---------------------------
A primeira versao deste script lia o `ImagePositionPatient` de cada arquivo
DICOM, o que exigia os 133 GB em disco. Era desnecessario: o pylidc **ja
guarda as posicoes Z de todos os cortes no proprio banco de anotacoes**, na
tabela `Zval`, acessiveis por `scan.slice_zvals`. Os 897 exames elegiveis
sao analisados em cerca de 12 segundos, sem uma unica imagem em disco.

    scan.slice_zvals -> ndarray com o Z de cada corte, ja ordenado

Consequencia pratica: qualquer pessoa do grupo reproduz esta verificacao com
`pip install pylidc`, sem acervo, sem Google Drive, sem espera.

UMA DISTINCAO QUE CUSTA UM TERCO DO DATASET
-------------------------------------------
O texto antigo do §3.2 tratava como o mesmo defeito duas coisas diferentes,
dizendo que "divergencia acentuada entre slice_spacing e slice_thickness
indica reconstrucao sobreposta ou com lacuna". Sao fenomenos distintos:

  1. ESPACAMENTO IRREGULAR (defeito de verdade)
     O passo entre cortes nao e constante: ha lacuna, corte ausente ou passo
     variavel. Quebra a reamostragem isotropica, que assume passo constante.
     E o que o LUNA16 excluiu.

  2. RECONSTRUCAO SOBREPOSTA (nao e defeito)
     O passo e MENOR que a espessura declarada - os cortes se sobrepoem. E
     escolha de protocolo, aumenta a resolucao efetiva em Z e nao atrapalha
     nada.

O script separa as duas e reporta, de proposito, quanto custaria aplicar a
leitura ingenua (excluir tambem os sobrepostos). O numero e grande, e e o
argumento para nunca implementar o §3.2 daquele jeito.

O QUE ELE PRODUZ
----------------
    selecao/espacamento_por_exame.csv   uma linha por exame, com as medidas
    selecao/verificacao_espacamento.md  relatorio para colar na issue #4

NOTA PARA QUEM FOR MEXER EM to_volume() NO PYTHON 3.12+
-------------------------------------------------------
O pylidc 0.2.3 usa `configparser.SafeConfigParser()`, removido no Python
3.12. Isso quebra `scan.get_path_to_dicom_files()` e, por consequencia,
`scan.to_volume()` - de que a etapa de Transformacao depende. O conserto e
uma linha, antes de `import pylidc`:

    import configparser
    if not hasattr(configparser, "SafeConfigParser"):
        configparser.SafeConfigParser = configparser.ConfigParser

Este script nao precisa disso, porque nao toca em arquivo DICOM. O
`extrai_patches.py` precisa.

COMO RODAR
----------
    python selecao/verifica_espacamento.py
"""

import csv
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

SLICE_THICKNESS_MAX = 2.5

# Tolerancia para considerar o passo entre cortes constante. 0,1 mm e bem
# menor que qualquer espessura do acervo (0,45 a 2,5 mm) e maior que o ruido
# de arredondamento do ImagePositionPatient.
TOL_MM = 0.10

CSV_NODULOS = Path(__file__).parent / "nodulos_selecionados.csv"
CSV_SAIDA = Path(__file__).parent / "espacamento_por_exame.csv"
MD_SAIDA = Path(__file__).parent / "verificacao_espacamento.md"


def medir(scan):
    """Mede o espacamento de um exame e classifica o resultado."""
    espessura = float(scan.slice_thickness)
    base = {
        "scan_id": scan.id,
        "patient_id": scan.patient_id,
        "slice_thickness": espessura,
    }

    try:
        zs = scan.slice_zvals
    except Exception as erro:
        return {**base, "n_cortes": 0, "passo_mediano": "", "desvio_max_mm": "",
                "passos_fora": "", "situacao": "sem_zvals", "observacao": repr(erro)[:120]}

    if zs is None or len(zs) < 3:
        n = 0 if zs is None else len(zs)
        return {**base, "n_cortes": n, "passo_mediano": "", "desvio_max_mm": "",
                "passos_fora": "", "situacao": "sem_zvals",
                "observacao": f"apenas {n} corte(s) registrado(s)"}

    diffs = np.diff(np.sort(np.asarray(zs, dtype=float)))
    mediano = float(np.median(diffs))
    desvio_max = float(np.max(np.abs(diffs - mediano)))
    passos_fora = int(np.sum(np.abs(diffs - mediano) > TOL_MM))

    if desvio_max > TOL_MM:
        situacao = "irregular"
        obs = (f"{passos_fora} de {len(diffs)} passos fora do mediano; "
               f"maior desvio {desvio_max:.3f} mm")
    elif (espessura - mediano) > TOL_MM:
        situacao = "sobreposto"
        obs = f"passo {mediano:.3f} < espessura {espessura:.3f}: reconstrucao sobreposta"
    elif (mediano - espessura) > TOL_MM:
        situacao = "espacado"
        obs = f"passo {mediano:.3f} > espessura {espessura:.3f}: cortes com folga entre si"
    else:
        situacao = "regular"
        obs = ""

    return {**base, "n_cortes": int(len(zs)), "passo_mediano": round(mediano, 4),
            "desvio_max_mm": round(desvio_max, 4), "passos_fora": passos_fora,
            "situacao": situacao, "observacao": obs}


def nodulos_por_paciente():
    """Quantos nodulos selecionados cada paciente contribui."""
    if not CSV_NODULOS.exists():
        print(f"[aviso] {CSV_NODULOS} nao encontrado; impacto em nodulos nao sera calculado.")
        return {}
    contagem = Counter()
    with open(CSV_NODULOS, newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            contagem[linha["patient_id"]] += 1
    return contagem


def main():
    total_scans = pl.query(pl.Scan).count()
    print(f"[checagem] exames no banco do pylidc: {total_scans}")
    if total_scans != 1018:
        print("[aviso] esperado 1018. O banco de anotacoes pode estar incompleto.")

    scans = (
        pl.query(pl.Scan)
        .filter(pl.Scan.slice_thickness <= SLICE_THICKNESS_MAX)
        .all()
    )
    print(f"[filtro] exames com slice_thickness <= {SLICE_THICKNESS_MAX} mm: {len(scans)}")
    print("[medindo] lendo as posicoes Z do banco de anotacoes (sem DICOM)...")

    resultados = [medir(s) for s in scans]

    campos = ["scan_id", "patient_id", "slice_thickness", "n_cortes",
              "passo_mediano", "desvio_max_mm", "passos_fora",
              "situacao", "observacao"]
    with open(CSV_SAIDA, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)

    sit = Counter(r["situacao"] for r in resultados)
    n_total = len(resultados)
    n_irregular = sit["irregular"]
    n_sobreposto = sit["sobreposto"]
    restantes = n_total - n_irregular

    por_paciente = nodulos_por_paciente()
    pac_irregular = {r["patient_id"] for r in resultados if r["situacao"] == "irregular"}
    pac_ingenuo = pac_irregular | {
        r["patient_id"] for r in resultados if r["situacao"] in ("sobreposto", "espacado")
    }
    nod_irregular = sum(por_paciente.get(p, 0) for p in pac_irregular)
    nod_ingenuo = sum(por_paciente.get(p, 0) for p in pac_ingenuo)

    md = []
    md.append("# Verificacao do criterio §3.2 — espacamento entre cortes")
    md.append("")
    md.append("Gerado por `selecao/verifica_espacamento.py`, a partir das posicoes Z que o")
    md.append("pylidc guarda no proprio banco de anotacoes (`scan.slice_zvals`). **Nao usa")
    md.append("arquivo DICOM nenhum.**")
    md.append("")
    md.append("## Resposta a pergunta da issue")
    md.append("")
    md.append(
        "O filtro de espacamento **nao estava aplicado nem verificado**: "
        "`selecao_nodulos.py` filtra somente `slice_thickness <= 2.5` e nao le posicao "
        "de corte em ponto algum. Esta e a primeira medicao."
    )
    md.append("")
    md.append("## Numeros")
    md.append("")
    md.append("| Situacao | Exames | % |")
    md.append("|---|---:|---:|")
    for chave, rotulo in [
        ("regular", "Regular — passo constante e igual a espessura"),
        ("sobreposto", "Sobreposto — passo < espessura · **nao e defeito**"),
        ("espacado", "Espacado — passo > espessura, mas constante"),
        ("irregular", "**Irregular** — passo variavel ou corte ausente · **defeito**"),
        ("sem_zvals", "Sem posicoes de corte registradas"),
    ]:
        n = sit.get(chave, 0)
        md.append(f"| {rotulo} | {n} | {100 * n / n_total:.1f}% |")
    md.append(f"| **Total** | **{n_total}** | |")
    md.append("")
    md.append("## Se o §3.2 virar criterio de exclusao")
    md.append("")
    md.append(f"- Exames elegiveis hoje (so espessura): **{n_total}**")
    md.append(f"- Excluidos por espacamento irregular: **{n_irregular}**")
    md.append(f"- Exames restantes: **{restantes}**")
    md.append("- LUNA16 reporta: **888**")
    md.append(f"- Nodulos do manifesto nesses exames: **{nod_irregular}**")
    md.append("")
    if abs(restantes - 888) <= 2:
        md.append(
            f"> {restantes} praticamente coincide com os 888 do LUNA16: a explicacao da "
            "secao 8 do `criterios_selecao.md` se confirma."
        )
    else:
        md.append(
            f"> {restantes} ainda fica {restantes - 888} exames acima dos 888 do LUNA16. "
            "O filtro de espacamento explica **parte** da diferenca, nao toda. A secao 8 "
            "precisa dizer isso: atribuir a diferenca inteira a este filtro seria inventar "
            "uma causa."
        )
    md.append("")
    md.append("## O custo da leitura ingenua")
    md.append("")
    md.append(
        "A redacao antiga do §3.2 mandava sinalizar \"divergencia acentuada entre "
        "`slice_spacing` e `slice_thickness`\". Aplicada ao pe da letra, ela excluiria "
        "tambem os exames de reconstrucao sobreposta:"
    )
    md.append("")
    md.append(f"- Exames excluidos: **{len(pac_ingenuo)}** de {n_total}")
    md.append(f"- Nodulos perdidos: **{nod_ingenuo}** de {sum(por_paciente.values()) or '?'}")
    md.append("")
    md.append(
        f"Sobreposicao nao e defeito — e o protocolo entregando mais informacao em Z, nao "
        f"menos. Sao {n_sobreposto} exames que nao ha razao para descartar. Esta e a razao "
        "de o §3.2 ter sido reescrito separando os dois fenomenos."
    )
    md.append("")
    md.append("## Exames irregulares")
    md.append("")
    md.append("| Paciente | Espessura | Passo mediano | Maior desvio | Passos fora | Cortes |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in sorted(
        (r for r in resultados if r["situacao"] == "irregular"),
        key=lambda r: -float(r["desvio_max_mm"]),
    ):
        md.append(
            f"| `{r['patient_id']}` | {r['slice_thickness']} mm | {r['passo_mediano']} mm | "
            f"**{r['desvio_max_mm']} mm** | {r['passos_fora']} | {r['n_cortes']} |"
        )
    md.append("")
    md.append(f"Tolerancia usada: {TOL_MM} mm de desvio maximo em relacao ao passo mediano.")
    md.append("")
    md.append("Detalhe por exame: `selecao/espacamento_por_exame.csv`.")
    md.append("")

    MD_SAIDA.write_text("\n".join(md), encoding="utf-8")

    print()
    print("========== RESUMO ==========")
    for k, v in sit.most_common():
        print(f"  {k}: {v}")
    print(f"  restariam apos excluir irregulares: {restantes} (LUNA16: 888)")
    print(f"  nodulos nesses exames: {nod_irregular}")
    print(f"  leitura ingenua excluiria: {len(pac_ingenuo)} exames / {nod_ingenuo} nodulos")
    print()
    print(f"Relatorio: {MD_SAIDA}")
    print(f"Por exame: {CSV_SAIDA}")


if __name__ == "__main__":
    main()
