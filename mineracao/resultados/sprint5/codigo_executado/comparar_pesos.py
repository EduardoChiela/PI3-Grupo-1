"""Compara, NA VALIDAÇÃO, o baseline com e sem pesos de classe.

Lê os dois predicoes_validacao.csv gerados por treino_baseline.py, escolhe o
threshold (Youden) de cada configuração só na validação e imprime a tabela de
avaliacao.comparar_configuracoes. Não usa o teste e não escolhe vencedor.

Uso (raiz do repositório):
  python -m mineracao.comparar_pesos --com-pesos mineracao/resultados/real_com_pesos/predicoes_validacao.csv --sem-pesos mineracao/resultados/real_sem_pesos/predicoes_validacao.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mineracao.avaliacao import (comparar_configuracoes, escolher_threshold_validacao,
                                 validar_com_manifesto)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--com-pesos", type=Path, required=True)
    ap.add_argument("--sem-pesos", type=Path, required=True)
    ap.add_argument("--manifesto", type=Path,
                    default=Path("transformacao/manifest_patches.csv"))
    args = ap.parse_args()

    manifesto = pd.read_csv(args.manifesto)
    predicoes = {"com_pesos": validar_com_manifesto(pd.read_csv(args.com_pesos), manifesto),
                 "sem_pesos": validar_com_manifesto(pd.read_csv(args.sem_pesos), manifesto)}
    thresholds = {nome: escolher_threshold_validacao(df)["threshold"]
                  for nome, df in predicoes.items()}
    pd.set_option("display.width", 200)
    print(comparar_configuracoes(predicoes, thresholds).round(4).to_string())


if __name__ == "__main__":
    main()
