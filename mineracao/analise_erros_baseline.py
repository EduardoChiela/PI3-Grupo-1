import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Limiar oficial de Youden (congelado na validação)
LIMIAR_OFICIAL = 0.4304952919483185

# 2. Busca inteligente dos arquivos de entrada da Sprint 5
caminhos_pred = [
    Path("mineracao/resultados/comparacao_s5/predicoes_teste_pareadas.csv"),
    Path("mineracao/resultados/comparacao/predicoes_teste_pareadas.csv"),
    Path("predicoes_teste_pareadas.csv")
]
path_pred = next((p for p in caminhos_pred if p.exists()), None)
if path_pred is None:
    raise FileNotFoundError("Erro: predicoes_teste_pareadas.csv não encontrado!")

caminhos_nodulos = [
    Path("selecao/nodulos_com_split.csv"),
    Path("nodulos_com_split.csv")
]
path_nodulos = next((p for p in caminhos_nodulos if p.exists()), None)
if path_nodulos is None:
    raise FileNotFoundError("Erro: nodulos_com_split.csv não encontrado!")

caminhos_faixa = [
    Path("transformacao/patches_faixa_preta.csv"),
    Path("patches_faixa_preta.csv")
]
path_faixa = next((p for p in caminhos_faixa if p.exists()), None)

out_dir = Path("mineracao/resultados/interpretacao")
out_dir.mkdir(parents=True, exist_ok=True)

# 3. Leitura e cruzamento dos dados
pred = pd.read_csv(path_pred)
nodulos = pd.read_csv(path_nodulos)

if path_faixa and path_faixa.exists():
    faixa_preta = pd.read_csv(path_faixa)
    ids_faixa = set(faixa_preta["nodule_id"].astype(str)) if "nodule_id" in faixa_preta.columns else set()
else:
    ids_faixa = set()

df = pred.merge(nodulos, on="nodule_id", how="left", suffixes=("", "_nod"))
df["faixa_preta"] = df["nodule_id"].astype(str).isin(ids_faixa)

# 4. Binarização com o limiar oficial (0.4305)
df["pred_binaria"] = (df["p_baseline"] >= LIMIAR_OFICIAL).astype(int)

# Separação dos erros
fn = df[(df["y_true"] == 1) & (df["pred_binaria"] == 0)].copy()
fp = df[(df["y_true"] == 0) & (df["pred_binaria"] == 1)].copy()

fn["tipo_erro"] = "Falso Negativo (Maligno Perdido)"
fp["tipo_erro"] = "Falso Positivo (Benigno Marcado)"

erros = pd.concat([fn, fp], ignore_index=True)

# Autoconferência do protocolo (3 FNs e 8 FPs = 11 erros)
assert len(fn) == 3, f"Erro: Esperado 3 FNs, mas foi encontrado {len(fn)}"
assert len(fp) == 8, f"Erro: Esperado 8 FPs, mas foi encontrado {len(fp)}"
assert len(erros) == 11, f"Erro: Esperado 11 erros no total, mas foi encontrado {len(erros)}"

# 5. Salvar Tabela Resumo dos Erros (CSV)
cols_tabela = [
    "nodule_id", "patient_id", "tipo_erro", "y_true", 
    "p_baseline", "p_resnet", "diametro_mm", 
    "malignidade_mediana", "escores_individuais", "faixa_preta"
]
cols_export = [c for c in cols_tabela if c in erros.columns]
tabela_erros = erros[cols_export].sort_values(by=["tipo_erro", "p_baseline"], ascending=[True, False])
tabela_erros.to_csv(out_dir / "tabela_erros_baseline.csv", index=False)
print(f"Tabela salva: {out_dir / 'tabela_erros_baseline.csv'}")

# 6. Gerar Gráfico de Barras dos Erros (PNG)
fig, ax = plt.subplots(figsize=(10, 6))

colors = ["#d9534f" if "Negativo" in t else "#f0ad4e" for t in tabela_erros["tipo_erro"]]
y_pos = np.arange(len(tabela_erros))

ax.barh(y_pos, tabela_erros["p_baseline"], color=colors, alpha=0.85, edgecolor="black")
ax.axvline(LIMIAR_OFICIAL, color="red", linestyle="--", linewidth=1.5, label=f"Limiar Youden ({LIMIAR_OFICIAL:.4f})")

labels = [f"{row['nodule_id'][:14]}... ({'FN' if 'Negativo' in row['tipo_erro'] else 'FP'}, d={row['diametro_mm']:.1f}mm)" 
          for _, row in tabela_erros.iterrows()]

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Probabilidade Predita pelo Baseline (p_baseline)", fontsize=10, fontweight="bold")
ax.set_title("Detalhamento dos 11 Erros do Modelo Baseline no Teste", fontsize=12, fontweight="bold")
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(out_dir / "grafico_erros_baseline.png", dpi=300)
plt.close()
print(f"Gráfico salvo: {out_dir / 'grafico_erros_baseline.png'}")

# 7. Gerar Mosaico Visual dos Patches dos 11 Erros (PNG)
caminhos_patches = [
    Path("patches/patches_2_5d.npz"),
    Path("transformacao/patches_2_5d.npz"),
    Path("patches_2_5d.npz")
]
path_patches = next((p for p in caminhos_patches if p.exists()), None)

if path_patches:
    z = np.load(path_patches)
    ids_npz = list(z["nodule_id"])
    patches_npz = z["patches"]

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for i, (_, row) in enumerate(tabela_erros.iterrows()):
        nid = row["nodule_id"]
        if nid in ids_npz:
            idx = ids_npz.index(nid)
            patch = patches_npz[idx]
            img = patch[0] if patch.ndim == 3 else patch
            
            axes[i].imshow(img, cmap="gray")
            tipo_sigla = "FN" if "Negativo" in row["tipo_erro"] else "FP"
            axes[i].set_title(
                f"{nid[:14]}...\n[{tipo_sigla}] d={row['diametro_mm']:.1f}mm",
                fontsize=8, fontweight="bold"
            )
        axes[i].axis("off")

    for j in range(len(tabela_erros), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "figura_patches_erros_baseline.png", dpi=300)
    plt.close()
    print(f"Mosaico de patches salvo: {out_dir / 'figura_patches_erros_baseline.png'}")
