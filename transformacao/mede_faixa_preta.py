"""
Mede a faixa preta dos patches 2.5D e gera transformacao/patches_faixa_preta.csv.

Contexto (issue #13)
--------------------
Quando a janela de 64 mm do patch passa do volume reconstruido, a parte de fora
fica em -1000 HU. Depois da janela de intensidade e da normalizacao, isso vira
0.0 exato, formando uma faixa preta na borda do patch.

Este script nao decide nada: ele mede. A decisao do grupo esta registrada em
docs/criterios_transformacao.md, secao 5.1.

CUIDADO: zero nao e sinonimo de padding
---------------------------------------
A janela de intensidade mapeia -1000 HU para 0.0. Ar de pulmao fica em -1000 HU
ou abaixo, entao **ar normal tambem vira exatamente 0.0**. Em 390 dos 732 patches
sem nenhuma faixa existem pixels zerados, chegando a 21,8% do patch. Contar
"pixels zero" como falta de dado superestima a faixa e, pior, mede a distancia
ate o ar mais proximo em vez da distancia ate a borda do volume.

A separacao usada aqui: padding e o componente conexo de zeros que encosta numa
linha ou coluna de borda INTEIRAMENTE zerada. Tecido real tem textura (vasos,
ruido), entao uma linha de 64 pixels exatamente 0.0 nao ocorre em pulmao
adquirido. Rodado sobre os 732 patches sem faixa, esse criterio marca zero
pixels como padding - nenhum falso positivo.

O que ele mede, por patch
-------------------------
- largura_px      : maior largura da faixa, em pixels (1 px = 1 mm apos a
                    reamostragem para 1 mm isotropico)
- area_padding_px : pixels de padding somados nos tres canais (de 3 x 64 x 64)
- pct_padding     : area_padding_px em porcentagem do patch
- eixo            : em que eixo anatomico a faixa esta (X, Y ou Z)
- margem_mm       : distancia do centro do patch ate o pixel de padding mais
                    proximo
- raio_mm         : metade do diametro_mm do nodulo (selecao/nodulos_com_split.csv)
- folga_mm        : margem_mm - raio_mm. Negativo = a faixa invade a lesao.

Geometria dos canais
--------------------
O extrai_patches.py monta os planos assim (volume com eixos i=linha=Y,
j=coluna=X, k=corte=Z):

    axial   = cubo[:, :, centro]   -> linhas = Y, colunas = X
    coronal = cubo[centro, :, :]   -> linhas = X, colunas = Z
    sagital = cubo[:, centro, :]   -> linhas = Y, colunas = Z

Entao uma faixa nas COLUNAS do coronal e do sagital e uma faixa no eixo Z, e
assim por diante. E dai que sai a coluna `eixo`.

Uso
---
    python transformacao/mede_faixa_preta.py

Sem argumentos. Le patches/patches_2_5d.npz, transformacao/manifest_patches.csv
e selecao/nodulos_com_split.csv a partir da raiz do repositorio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

RAIZ = Path(__file__).resolve().parent.parent

CAMINHO_NPZ = RAIZ / "patches" / "patches_2_5d.npz"
CAMINHO_MANIFESTO = RAIZ / "transformacao" / "manifest_patches.csv"
CAMINHO_SELECAO = RAIZ / "selecao" / "nodulos_com_split.csv"
CAMINHO_SAIDA = RAIZ / "transformacao" / "patches_faixa_preta.csv"

# Nomes dos canais na ordem gravada no .npz, e que eixo anatomico cada
# dimensao da imagem representa. (linhas, colunas)
CANAIS = {
    "axial": ("Y", "X"),
    "coronal": ("X", "Z"),
    "sagital": ("Y", "Z"),
}

LADO = 64  # pixels, e tambem milimetros apos a reamostragem


def bordas_zeradas(img: np.ndarray) -> dict[str, int]:
    """Larguras da faixa preta em cada uma das quatro bordas da imagem.

    Uma borda so conta se a primeira linha (ou coluna) estiver inteiramente
    zerada: isso distingue uma faixa de borda de um pixel preto solto no meio
    do pulmao, que e ar normal e nao falta de dado.
    """
    zero = img == 0
    lin_zerada = zero.all(axis=1)
    col_zerada = zero.all(axis=0)

    def corrida(mascara: np.ndarray, do_fim: bool) -> int:
        seq = mascara[::-1] if do_fim else mascara
        n = 0
        for v in seq:
            if not v:
                break
            n += 1
        return n

    return {
        "linha_inicio": corrida(lin_zerada, do_fim=False),
        "linha_fim": corrida(lin_zerada, do_fim=True),
        "coluna_inicio": corrida(col_zerada, do_fim=False),
        "coluna_fim": corrida(col_zerada, do_fim=True),
    }


def mascara_padding(img: np.ndarray) -> np.ndarray:
    """Pixels que sao padding (fora do volume reconstruido), nao ar de pulmao.

    Semeia apenas nas linhas/colunas de borda inteiramente zeradas e propaga
    pelo componente conexo de zeros. Ver a nota "zero nao e sinonimo de padding"
    no topo do arquivo.
    """
    zero = img == 0
    if not zero.any():
        return np.zeros_like(zero)

    semente = np.zeros_like(zero)
    if zero[0].all():
        semente[0] = True
    if zero[-1].all():
        semente[-1] = True
    if zero[:, 0].all():
        semente[:, 0] = True
    if zero[:, -1].all():
        semente[:, -1] = True
    if not semente.any():
        return np.zeros_like(zero)

    rotulos, _ = ndimage.label(zero)
    ids = set(rotulos[semente].tolist()) - {0}
    return np.isin(rotulos, list(ids))


def mede_patch(patch: np.ndarray) -> dict:
    """Mede a faixa preta de um patch (3, 64, 64) ja normalizado em 0..1."""
    yy, xx = np.mgrid[0:LADO, 0:LADO]
    # Centro geometrico de uma grade par fica entre os pixels 31 e 32.
    dist_centro = np.sqrt((yy - (LADO - 1) / 2) ** 2 + (xx - (LADO - 1) / 2) ** 2)

    largura = 0
    eixos: set[str] = set()
    margem = np.inf
    area = 0

    for indice, (canal, (eixo_linha, eixo_coluna)) in enumerate(CANAIS.items()):
        img = patch[indice]
        b = bordas_zeradas(img)

        if b["linha_inicio"] or b["linha_fim"]:
            eixos.add(eixo_linha)
        if b["coluna_inicio"] or b["coluna_fim"]:
            eixos.add(eixo_coluna)

        largura = max(largura, *b.values())

        pad = mascara_padding(img)
        area += int(pad.sum())
        if pad.any():
            margem = min(margem, float(dist_centro[pad].min()))

    return {
        "tem_faixa": largura > 0,
        "largura_px": int(largura),
        "area_padding_px": area,
        "pct_padding": round(area / (len(CANAIS) * LADO * LADO) * 100, 2),
        "eixo": "+".join(sorted(eixos)) if eixos else "",
        "margem_mm": round(margem, 1) if np.isfinite(margem) else None,
    }


def rotula_causa(eixo: str) -> str:
    """Traduz o eixo da faixa para a causa fisica, em uma palavra."""
    if not eixo:
        return ""
    if eixo == "Z":
        # Coronal e sagital compartilham Z; faixa so em Z = o patch passou da
        # primeira ou da ultima fatia adquirida.
        return "limite do exame (apice/base da cobertura)"
    if "Z" not in eixo:
        # X e/ou Y: o patch passou da borda do campo circular reconstruido.
        return "borda do campo reconstruido (nodulo periferico)"
    return "misto (limite do exame e borda do campo)"


def main() -> int:
    for caminho in (CAMINHO_NPZ, CAMINHO_MANIFESTO, CAMINHO_SELECAO):
        if not caminho.exists():
            print(f"ERRO: nao encontrei {caminho}", file=sys.stderr)
            return 1

    dados = np.load(CAMINHO_NPZ, allow_pickle=True)
    patches = dados["patches"].astype(np.float32)
    nodule_id_npz = list(dados["nodule_id"])

    manifesto = pd.read_csv(CAMINHO_MANIFESTO)
    selecao = pd.read_csv(CAMINHO_SELECAO)[["nodule_id", "diametro_mm"]]

    if list(manifesto["nodule_id"]) != nodule_id_npz:
        print(
            "ERRO: a ordem do manifesto nao bate com a ordem do .npz. "
            "O idx do manifesto indexa o .npz, entao isso invalida a medicao.",
            file=sys.stderr,
        )
        return 1

    tabela = manifesto.merge(selecao, on="nodule_id", how="left", validate="1:1")
    faltando = int(tabela["diametro_mm"].isna().sum())
    if faltando:
        print(
            f"AVISO: {faltando} nodulos sem diametro_mm; folga_mm fica vazia neles.",
            file=sys.stderr,
        )

    linhas = []
    falsos_positivos = 0
    for i in range(len(patches)):
        m = mede_patch(patches[i])
        if not m["tem_faixa"]:
            # Autoconferencia: num patch sem faixa, nenhum pixel pode ser
            # classificado como padding. Se isso falhar, o criterio esta
            # pegando ar de pulmao e a medicao inteira perde o sentido.
            falsos_positivos += m["area_padding_px"]
            continue

        raio = tabela.loc[i, "diametro_mm"]
        raio = None if pd.isna(raio) else round(float(raio) / 2, 1)
        folga = None if (raio is None or m["margem_mm"] is None) else round(m["margem_mm"] - raio, 1)

        linhas.append(
            {
                "idx": int(tabela.loc[i, "idx"]),
                "nodule_id": tabela.loc[i, "nodule_id"],
                "patient_id": tabela.loc[i, "patient_id"],
                "split": tabela.loc[i, "split"],
                "label": tabela.loc[i, "label"],
                "largura_px": m["largura_px"],
                "area_padding_px": m["area_padding_px"],
                "pct_padding": m["pct_padding"],
                "eixo": m["eixo"],
                "causa": rotula_causa(m["eixo"]),
                "diametro_mm": None if raio is None else round(raio * 2, 1),
                "raio_mm": raio,
                "margem_mm": m["margem_mm"],
                "folga_mm": folga,
                "lesao_truncada": None if folga is None else bool(folga < 0),
            }
        )

    saida = pd.DataFrame(linhas).sort_values("folga_mm", na_position="last")
    saida.to_csv(CAMINHO_SAIDA, index=False)

    total = len(patches)
    n = len(saida)
    print(f"patches analisados: {total}")
    print(f"patches com faixa preta: {n} ({n / total * 100:.1f}%)")
    print(
        f"autoconferencia: pixels marcados como padding nos {total - n} patches "
        f"sem faixa = {falsos_positivos} (esperado 0)"
    )
    if falsos_positivos:
        print(
            "ERRO: o criterio de padding esta pegando ar de pulmao.",
            file=sys.stderr,
        )
        return 1
    print()
    print("por rotulo:")
    print(saida["label"].value_counts().to_string())
    print()
    print("por split:")
    print(saida["split"].value_counts().to_string())
    print()
    print("por causa:")
    print(saida["causa"].value_counts().to_string())
    print()
    print("margem do centro ate a faixa, contra o raio do nodulo:")
    print(
        saida[["idx", "split", "label", "largura_px", "pct_padding", "diametro_mm", "margem_mm", "folga_mm"]]
        .to_string(index=False)
    )
    print()
    truncados = saida[saida["lesao_truncada"] == True]  # noqa: E712
    print(f"com a lesao truncada pela faixa (folga_mm < 0): {len(truncados)}")
    print(f"menor folga observada: {saida['folga_mm'].min():.1f} mm")
    print()
    print(f"gravado: {CAMINHO_SAIDA.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
