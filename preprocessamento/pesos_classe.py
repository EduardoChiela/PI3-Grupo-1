"""
Conferencia do split por paciente + calculo de pesos de classe.

Responsavel: Leticia Victoria (papel "confere" na Tarefa 2).

O que este script faz, em ordem:
1. Le o CSV com split (saida do split_por_paciente.py do Eduardo).
2. Verifica que nenhum patient_id aparece em mais de um conjunto
   (treino / validacao / teste) -- checagem de vazamento entre grupos.
3. Confere que a estratificacao por classe (benigno/maligno) se mantem
   parecida nos tres conjuntos, comparando com a proporcao global.
4. Recalcula os pesos de classe de forma independente, usando a formula
   w = N / (2 x n_da_classe), com N e n_da_classe medidos SOMENTE no
   conjunto de treino.

Este script nao gera dado novo nem decide o split: ele confere o que
ja foi gerado e produz os pesos que vao para a funcao de perda.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as erro:
    raise SystemExit(
        "Dependencia ausente. Ative o ambiente virtual e instale: pip install pandas"
    ) from erro


COLUNA_GRUPO = "patient_id"
COLUNA_ROTULO = "rotulo_binario"
COLUNA_SPLIT = "split"
ORDEM_SPLITS = ["treino", "validacao", "teste"]

# Tolerancia para a proporcao de cada classe em cada split, comparada com
# a proporcao global. Um desvio maior que isso indica que a estratificacao
# nao ficou boa o suficiente e precisa ser revista antes de seguir.
TOLERANCIA_ESTRATIFICACAO = 0.05


def ler_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confere vazamento e estratificacao, e calcula pesos de classe."
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=Path("selecao") / "nodulos_com_split.csv",
        help="CSV com as colunas patient_id, rotulo_binario e split.",
    )
    return parser.parse_args()


def carregar_dados(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {caminho}")

    df = pd.read_csv(caminho)
    colunas_obrigatorias = {COLUNA_GRUPO, COLUNA_ROTULO, COLUNA_SPLIT}
    faltantes = colunas_obrigatorias.difference(df.columns)
    if faltantes:
        raise ValueError(f"Colunas obrigatorias ausentes: {sorted(faltantes)}")

    splits_invalidos = set(df[COLUNA_SPLIT].unique()) - set(ORDEM_SPLITS)
    if splits_invalidos:
        raise ValueError(f"Valores de split nao reconhecidos: {sorted(splits_invalidos)}")

    return df


def verificar_vazamento(df: pd.DataFrame) -> None:
    """Garante que nenhum patient_id aparece em mais de um split.

    Levanta ValueError com os pacientes duplicados se houver vazamento.
    """
    pacientes_por_split = {
        nome: set(df.loc[df[COLUNA_SPLIT] == nome, COLUNA_GRUPO]) for nome in ORDEM_SPLITS
    }

    vazamentos = []
    for i, split_a in enumerate(ORDEM_SPLITS):
        for split_b in ORDEM_SPLITS[i + 1 :]:
            interseccao = pacientes_por_split[split_a] & pacientes_por_split[split_b]
            if interseccao:
                vazamentos.append((split_a, split_b, sorted(interseccao)))

    if vazamentos:
        detalhes = "; ".join(
            f"{a} x {b}: {exemplos[:5]}" for a, b, exemplos in vazamentos
        )
        raise ValueError(f"VAZAMENTO DE PACIENTE ENTRE SPLITS -> {detalhes}")

    print("[OK] Nenhum patient_id aparece em mais de um conjunto.")


def verificar_estratificacao(df: pd.DataFrame) -> None:
    """Compara a taxa de malignos em cada split com a taxa global."""
    taxa_global = (df[COLUNA_ROTULO] == "maligno").mean()
    print(f"\nTaxa de maligno global: {taxa_global:.4f}")

    problemas = []
    for nome_split in ORDEM_SPLITS:
        parte = df.loc[df[COLUNA_SPLIT] == nome_split]
        taxa = (parte[COLUNA_ROTULO] == "maligno").mean()
        desvio = abs(taxa - taxa_global)
        status = "OK" if desvio <= TOLERANCIA_ESTRATIFICACAO else "FORA DA TOLERANCIA"
        print(
            f"  {nome_split:<10} n={len(parte):>4}  taxa_maligno={taxa:.4f}  "
            f"desvio={desvio:.4f}  [{status}]"
        )
        if desvio > TOLERANCIA_ESTRATIFICACAO:
            problemas.append(nome_split)

    if problemas:
        raise ValueError(
            f"Estratificacao fora da tolerancia ({TOLERANCIA_ESTRATIFICACAO}) em: {problemas}"
        )

    print("[OK] Estratificacao por classe mantida nos tres conjuntos.")


def calcular_pesos_classe(df: pd.DataFrame) -> dict[str, float]:
    """Calcula w = N / (2 x n_da_classe) usando apenas o conjunto de treino."""
    treino = df.loc[df[COLUNA_SPLIT] == "treino"]
    contagens = treino[COLUNA_ROTULO].value_counts().to_dict()

    if len(contagens) != 2:
        raise ValueError(
            f"Esperava exatamente 2 classes no treino, encontrei: {sorted(contagens)}"
        )

    n_total_treino = len(treino)
    pesos = {
        rotulo: round(n_total_treino / (2 * n_classe), 6)
        for rotulo, n_classe in sorted(contagens.items())
    }

    print(f"\nContagem no treino: {contagens}")
    print(f"N total do treino: {n_total_treino}")
    print("Pesos de classe (w = N / (2 x n_da_classe)):")
    for rotulo, peso in pesos.items():
        print(f"  {rotulo:<10} peso={peso}")

    return pesos


def main() -> None:
    args = ler_argumentos()
    df = carregar_dados(args.entrada)

    try:
        verificar_vazamento(df)
        verificar_estratificacao(df)
        pesos = calcular_pesos_classe(df)
    except ValueError as erro:
        print(f"\n[FALHOU] {erro}", file=sys.stderr)
        raise SystemExit(1) from erro

    print("\nConferencia concluida sem problemas.")
    print(f"Pesos finais: {pesos}")


if __name__ == "__main__":
    main()