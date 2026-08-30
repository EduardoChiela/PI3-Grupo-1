"""
Split por paciente com avaliacao sem distorcao.

Regra decidida para lidar com o desequilibrio benigno/maligno:
1. Primeiro fazer o split por paciente, usando groups=patient_id.
2. Manter validacao e teste na distribuicao original do dataset.
3. Corrigir o desequilibrio apenas no treino, por pesos de classe/amostra.

Este script nao cria dados balanceados, nao duplica linhas e nao remove
exemplos. A saida principal e o dataset original com a coluna `split` e a
coluna `peso_amostra_treino`, preenchida apenas para linhas de treino.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

try:
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError as erro:
    raise SystemExit(
        "Dependencias ausentes. Ative o ambiente virtual e instale: "
        "pip install pandas scikit-learn"
    ) from erro


COLUNA_GRUPO = "patient_id"
COLUNA_ROTULO = "rotulo_binario"
COLUNA_TARGET = "_target_split"
COLUNA_SPLIT = "split"
COLUNA_PESO = "peso_amostra_treino"

ROTULO_PARA_TARGET = {
    "benigno": 0,
    "maligno": 1,
}

PROPORCOES = {
    "treino": 0.70,
    "validacao": 0.15,
    "teste": 0.15,
}

ORDEM_SPLITS = ["treino", "validacao", "teste"]


def ler_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera split 70/15/15 estratificado por rotulo e agrupado por paciente."
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=Path("selecao") / "nodulos_selecionados.csv",
        help="CSV de entrada com patient_id e rotulo_binario.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("selecao") / "nodulos_com_split.csv",
        help="CSV de saida com colunas split e peso_amostra_treino.",
    )
    parser.add_argument(
        "--resumo",
        type=Path,
        default=Path("selecao") / "split_por_paciente_resumo.json",
        help="JSON com regra aplicada, pesos e distribuicao final.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente usada pelo StratifiedGroupKFold.",
    )
    return parser.parse_args()


def carregar_dados(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {caminho}")

    df = pd.read_csv(caminho)
    colunas_obrigatorias = {COLUNA_GRUPO, COLUNA_ROTULO}
    faltantes = colunas_obrigatorias.difference(df.columns)
    if faltantes:
        raise ValueError(f"Colunas obrigatorias ausentes: {sorted(faltantes)}")

    if df[COLUNA_GRUPO].isna().any():
        raise ValueError(f"A coluna {COLUNA_GRUPO} contem valores nulos.")

    if df[COLUNA_ROTULO].isna().any():
        raise ValueError(f"A coluna {COLUNA_ROTULO} contem valores nulos.")

    df[COLUNA_TARGET] = df[COLUNA_ROTULO].map(ROTULO_PARA_TARGET)
    rotulos_invalidos = sorted(df.loc[df[COLUNA_TARGET].isna(), COLUNA_ROTULO].unique())
    if rotulos_invalidos:
        raise ValueError(f"Rotulos nao mapeados em {COLUNA_ROTULO}: {rotulos_invalidos}")

    if df[COLUNA_TARGET].nunique() != 2:
        raise ValueError("O split estratificado exige exatamente duas classes.")

    return df


def atribuir_folds(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> pd.Series:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds = pd.Series(index=df.index, dtype="int64")

    for fold, (_, indices_teste) in enumerate(
        splitter.split(df, y=df[COLUNA_TARGET], groups=df[COLUNA_GRUPO])
    ):
        folds.iloc[indices_teste] = fold

    return folds.astype(int)


def pontuar_split(df: pd.DataFrame, split: pd.Series) -> float:
    total_linhas = len(df)
    total_pacientes = df[COLUNA_GRUPO].nunique()
    taxa_maligno_global = df[COLUNA_TARGET].mean()
    pontuacao = 0.0

    for nome_split, proporcao_alvo in PROPORCOES.items():
        parte = df.loc[split == nome_split]
        proporcao_linhas = len(parte) / total_linhas
        proporcao_pacientes = parte[COLUNA_GRUPO].nunique() / total_pacientes
        taxa_maligno = parte[COLUNA_TARGET].mean()

        pontuacao += abs(proporcao_linhas - proporcao_alvo)
        pontuacao += 0.5 * abs(proporcao_pacientes - proporcao_alvo)
        pontuacao += 2.0 * abs(taxa_maligno - taxa_maligno_global)

    return float(pontuacao)


def gerar_split(df: pd.DataFrame, seed: int) -> pd.Series:
    folds_10 = atribuir_folds(df, n_splits=10, seed=seed)
    melhor_split = None
    melhor_pontuacao = None

    for folds_temp in combinations(sorted(folds_10.unique()), 3):
        mascara_temp = folds_10.isin(folds_temp)
        df_temp = df.loc[mascara_temp]

        try:
            folds_2_temp = atribuir_folds(df_temp, n_splits=2, seed=seed)
        except ValueError:
            continue

        for fold_validacao in sorted(folds_2_temp.unique()):
            split = pd.Series("treino", index=df.index, dtype="object")

            indices_validacao = folds_2_temp[folds_2_temp == fold_validacao].index
            indices_teste = folds_2_temp[folds_2_temp != fold_validacao].index

            split.loc[indices_validacao] = "validacao"
            split.loc[indices_teste] = "teste"

            pontuacao = pontuar_split(df, split)
            if melhor_pontuacao is None or pontuacao < melhor_pontuacao:
                melhor_pontuacao = pontuacao
                melhor_split = split

    if melhor_split is None:
        raise RuntimeError("Nao foi possivel gerar split 70/15/15 por paciente.")

    return melhor_split


def validar_sem_vazamento(df: pd.DataFrame) -> None:
    pacientes_por_split = {
        nome: set(df.loc[df[COLUNA_SPLIT] == nome, COLUNA_GRUPO])
        for nome in ORDEM_SPLITS
    }

    for split_a, split_b in combinations(ORDEM_SPLITS, 2):
        intersecao = pacientes_por_split[split_a].intersection(
            pacientes_por_split[split_b]
        )
        if intersecao:
            exemplos = sorted(intersecao)[:5]
            raise ValueError(
                f"Vazamento de pacientes entre {split_a} e {split_b}: {exemplos}"
            )


def calcular_pesos_treino(df: pd.DataFrame) -> dict[str, float]:
    treino = df.loc[df[COLUNA_SPLIT] == "treino"]
    contagens = treino[COLUNA_ROTULO].value_counts().to_dict()
    total = len(treino)
    n_classes = len(contagens)

    return {
        rotulo: round(total / (n_classes * contagem), 6)
        for rotulo, contagem in sorted(contagens.items())
    }


def resumo_distribuicao(df: pd.DataFrame) -> list[dict[str, object]]:
    total_linhas = len(df)
    total_pacientes = df[COLUNA_GRUPO].nunique()
    resumo = []

    for nome_split in ORDEM_SPLITS:
        parte = df.loc[df[COLUNA_SPLIT] == nome_split]
        contagens = parte[COLUNA_ROTULO].value_counts().to_dict()
        malignos = contagens.get("maligno", 0)
        taxa_maligno = malignos / len(parte) if len(parte) else 0

        resumo.append(
            {
                "split": nome_split,
                "linhas": int(len(parte)),
                "proporcao_linhas": round(len(parte) / total_linhas, 4),
                "pacientes": int(parte[COLUNA_GRUPO].nunique()),
                "proporcao_pacientes": round(
                    parte[COLUNA_GRUPO].nunique() / total_pacientes, 4
                ),
                "benigno": int(contagens.get("benigno", 0)),
                "maligno": int(malignos),
                "taxa_maligno": round(taxa_maligno, 4),
            }
        )

    return resumo


def salvar_saidas(
    df: pd.DataFrame,
    pesos_treino: dict[str, float],
    caminho_saida: Path,
    caminho_resumo: Path,
    seed: int,
) -> None:
    df_saida = df.drop(columns=[COLUNA_TARGET]).copy()
    df_saida[COLUNA_PESO] = pd.NA

    mascara_treino = df_saida[COLUNA_SPLIT] == "treino"
    df_saida.loc[mascara_treino, COLUNA_PESO] = df_saida.loc[
        mascara_treino, COLUNA_ROTULO
    ].map(pesos_treino)

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    caminho_resumo.parent.mkdir(parents=True, exist_ok=True)

    df_saida.to_csv(caminho_saida, index=False)

    resumo = {
        "regra": [
            "split por paciente antes de qualquer correcao de desequilibrio",
            "validacao e teste mantidos sem balanceamento",
            "correcao aplicada apenas no treino por pesos de classe/amostra",
        ],
        "metodo_split": "StratifiedGroupKFold",
        "groups": COLUNA_GRUPO,
        "rotulo_estratificacao": COLUNA_ROTULO,
        "proporcoes_alvo": PROPORCOES,
        "seed": seed,
        "pesos_treino": pesos_treino,
        "distribuicao": resumo_distribuicao(df_saida),
    }

    with open(caminho_resumo, "w", encoding="utf-8") as arquivo:
        json.dump(resumo, arquivo, indent=2, ensure_ascii=False)


def imprimir_resumo(df: pd.DataFrame, pesos_treino: dict[str, float]) -> None:
    resumo = pd.DataFrame(resumo_distribuicao(df))
    print("Regra aplicada:")
    print("1. Split por paciente com StratifiedGroupKFold.")
    print("2. Nenhum balanceamento em validacao/teste.")
    print("3. Pesos calculados apenas com o conjunto de treino.")
    print()
    print("Distribuicao final:")
    print(resumo.to_string(index=False))
    print()
    print(f"Pesos de treino: {pesos_treino}")


def main() -> None:
    args = ler_argumentos()
    df = carregar_dados(args.entrada)
    df[COLUNA_SPLIT] = gerar_split(df, args.seed)

    validar_sem_vazamento(df)
    pesos_treino = calcular_pesos_treino(df)
    salvar_saidas(df, pesos_treino, args.saida, args.resumo, args.seed)
    imprimir_resumo(df, pesos_treino)


if __name__ == "__main__":
    main()
