import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent

nodulos = pd.read_csv(BASE / "nodulos_selecionados.csv")
acervo = pd.read_csv(BASE / "acervo_disponivel.csv")

disponiveis = set(acervo.loc[acervo["n_dcm"] > 0, "patient_id"])

nodulos["tem_imagem"] = nodulos["patient_id"].isin(disponiveis)

saida = BASE / "nodulos_com_disponibilidade.csv"
nodulos.to_csv(saida, index=False)

total = len(nodulos)
com_imagem = int(nodulos["tem_imagem"].sum())
sem_imagem = total - com_imagem

print(f"total de nodulos          : {total}")
print(f"com imagem disponivel     : {com_imagem}")
print(f"sem imagem disponivel     : {sem_imagem}")

print("\ndistribuicao benigno/maligno entre os disponiveis:")
print(nodulos.loc[nodulos["tem_imagem"], "rotulo_binario"].value_counts())

print(f"\narquivo gerado: {saida}")
