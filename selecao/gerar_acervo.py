import csv
import re
from collections import Counter
from pathlib import Path

ACERVO = Path(r"D:\lidc_idri")

SAIDA = Path(__file__).parent / "acervo_disponivel.csv"

if not ACERVO.exists():
    raise SystemExit(f"caminho nao encontrado: {ACERVO}")

PADRAO_PACIENTE = re.compile(r"^LIDC-IDRI-\d{4}$")

pastas = sorted(p for p in ACERVO.iterdir() if p.is_dir() and PADRAO_PACIENTE.match(p.name))
print("pastas de paciente encontradas:", len(pastas))

linhas = []
vazias = 0
duplicatas_total = 0
pacientes_com_duplicata = []

for i, p in enumerate(pastas, 1):
    arquivos_dcm = list(p.rglob("*.dcm"))
    n = len(arquivos_dcm)
    linhas.append((p.name, n))
    if n == 0:
        vazias += 1

    # pastas de serie = pastas-folha (sem subpastas) dentro do paciente
    pastas_serie = [d.name for d in p.rglob("*") if d.is_dir() and not any(x.is_dir() for x in d.iterdir())]
    contagem = Counter(pastas_serie)
    dups = {nome: c for nome, c in contagem.items() if c > 1}
    if dups:
        pacientes_com_duplicata.append((p.name, dups))
        duplicatas_total += sum(c - 1 for c in dups.values())

    if i % 100 == 0:
        print(f"  {i}/{len(pastas)}")

with SAIDA.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["patient_id", "n_dcm"])
    w.writerows(linhas)

com_imagem = sum(1 for _, n in linhas if n > 0)
print(f"\npacientes com imagem   : {com_imagem} de {len(linhas)}")
print(f"pastas vazias (n_dcm=0): {vazias}")
print(f"total de arquivos .dcm : {sum(n for _, n in linhas)}")
print(f"series duplicadas      : {duplicatas_total} ocorrencias em {len(pacientes_com_duplicata)} pacientes")
if pacientes_com_duplicata:
    for nome, dups in pacientes_com_duplicata[:20]:
        print(f"  {nome}: {dups}")
print(f"arquivo gerado         : {SAIDA}")
