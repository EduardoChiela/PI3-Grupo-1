import csv
from pathlib import Path

# >>> COLE AQUI o caminho da pasta lidc_idri no lugar onde ela está agora <
ACERVO = Path(r"G:\Meu Drive\PI3-Grupo 01\Dataset\db\lidc_idri")

SAIDA = Path(__file__).parent / "acervo_disponivel.csv"

if not ACERVO.exists():
    raise SystemExit(f"caminho nao encontrado: {ACERVO}")

pastas = sorted(p for p in ACERVO.glob("LIDC-IDRI-*") if p.is_dir())
print("pacientes encontrados:", len(pastas))

linhas = []
for i, p in enumerate(pastas, 1):
    n = sum(1 for _ in p.rglob("*.dcm"))
    linhas.append((p.name, n))
    if i % 100 == 0:
        print(f"  {i}/{len(pastas)}")

with SAIDA.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["patient_id", "n_dcm"])
    w.writerows(linhas)

com_imagem = sum(1 for _, n in linhas if n > 0)
print(f"\npacientes com imagem: {com_imagem} de {len(linhas)}")
print(f"total de arquivos   : {sum(n for _, n in linhas)}")
print(f"arquivo gerado      : {SAIDA}")