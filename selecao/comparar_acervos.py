import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent

# versao commitada por outro integrante (Aquila), gerada a partir de uma copia
# do acervo no Google Drive, antes de ser sobrescrita pela auditoria local
COMMIT_AQUILA = "d5cbb6f"

local = pd.read_csv(BASE / "acervo_disponivel.csv")
csv_aquila = subprocess.run(
    ["git", "show", f"{COMMIT_AQUILA}:selecao/acervo_disponivel.csv"],
    cwd=BASE.parent,
    capture_output=True,
    text=True,
    check=True,
).stdout
aquila = pd.read_csv(StringIO(csv_aquila))
nodulos = pd.read_csv(BASE / "nodulos_selecionados.csv")
pacientes_com_nodulo = set(nodulos["patient_id"])

comp = local.merge(aquila, on="patient_id", how="outer", suffixes=("_local", "_aquila"))
comp["n_dcm_local"] = comp["n_dcm_local"].fillna(0).astype(int)
comp["n_dcm_aquila"] = comp["n_dcm_aquila"].fillna(0).astype(int)
comp["diferenca"] = comp["n_dcm_aquila"] - comp["n_dcm_local"]

divergentes = comp[comp["diferenca"] != 0].sort_values("diferenca", ascending=False)
print(f"patient_id com contagem diferente entre local e Aquila: {len(divergentes)} de {len(comp)}")
print(f"soma das diferencas (aquila - local): {comp['diferenca'].sum()}")

so_local = comp[(comp["n_dcm_local"] > 0) & (comp["n_dcm_aquila"] == 0)]
so_aquila = comp[(comp["n_dcm_aquila"] > 0) & (comp["n_dcm_local"] == 0)]
print(f"patient_id so com imagem no meu acervo (local>0, aquila=0): {len(so_local)}")
print(f"patient_id so com imagem no acervo do Aquila (aquila>0, local=0): {len(so_aquila)}")

# pastas vazias no meu acervo local
vazias = local[local["n_dcm"] == 0].copy()
vazias["tem_nodulo_selecionado"] = vazias["patient_id"].isin(pacientes_com_nodulo)

print(f"\npastas vazias no acervo local: {len(vazias)}")
print(f"dessas, quantas tem nodulo selecionado: {int(vazias['tem_nodulo_selecionado'].sum())}")

# dessas vazias, quantas tinham imagem no acervo do Aquila (ou seja, e falha
# de download local, nao ausencia real no acervo completo)
vazias_x_aquila = vazias.merge(aquila, on="patient_id", how="left", suffixes=("_local", "_aquila"))
tinha_no_aquila = vazias_x_aquila[vazias_x_aquila["n_dcm_aquila"].fillna(0) > 0]
print(f"dessas vazias, quantas tinham imagem no acervo do Aquila (falha de download local): {len(tinha_no_aquila)}")

# gera selecao/pastas_incompletas.csv: apenas as pastas vazias (n_dcm_local == 0)
saida = vazias[["patient_id", "n_dcm", "tem_nodulo_selecionado"]].rename(columns={"n_dcm": "n_dcm_local"})
saida = saida.sort_values("patient_id")
caminho_saida = BASE / "pastas_incompletas.csv"
saida.to_csv(caminho_saida, index=False)
print(f"\narquivo gerado: {caminho_saida} ({len(saida)} linhas)")
