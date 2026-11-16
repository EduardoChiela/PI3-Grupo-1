import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Caminhos dos arquivos de entrada da Sprint 5
base_dir = Path("mineracao/resultados/comparacao_s5")
pred_path = base_dir / "predicoes_teste_pareadas.csv"
json_path = base_dir / "resultados.json"

# Se os arquivos estiverem na pasta raiz no Colab, usa o arquivo direto
if not pred_path.exists():
    pred_path = Path("predicoes_teste_pareadas.csv")

out_dir = Path("mineracao/resultados/interpretacao")
out_dir.mkdir(parents=True, exist_ok=True)

# 2. Carregar predições e definir o limiar oficial da ResNet (0.3614)
pred = pd.read_csv(pred_path)

if json_path.exists():
    limiar_resnet = json.loads(json_path.read_text())["thresholds"]["resnet"]
else:
    limiar_resnet = 0.3614325523376465  # Limiar de Youden oficial congelado na validação

# 3. Calcular matriz de confusão diretamente (sem depender do módulo mineracao)
y_true = pred["y_true"].values
p_resnet = pred["p_resnet"].values
pred_binaria = (p_resnet >= limiar_resnet).astype(int)

tp = int(np.sum((y_true == 1) & (pred_binaria == 1)))
fn = int(np.sum((y_true == 1) & (pred_binaria == 0)))
fp = int(np.sum((y_true == 0) & (pred_binaria == 1)))
tn = int(np.sum((y_true == 0) & (pred_binaria == 0)))

# Autoconferência rigorosa exigida pelo protocolo da Sprint 5
assert (tp, fn, fp, tn) == (48, 5, 11, 47), f"Erro nos valores calculados: TP={tp}, FN={fn}, FP={fp}, TN={tn}"

# 4. Montar a Matriz de Confusão (TN, FP / FN, TP)
cm = np.array([[tn, fp], [fn, tp]])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

# Adicionar os valores numéricos nas células
for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > 30 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14, fontweight="bold")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predito Benigno", "Predito Maligno"], fontsize=10)
ax.set_yticklabels(["Real Benigno", "Real Maligno"], fontsize=10)
ax.set_xlabel("Classe Predita", fontsize=11, fontweight="bold")
ax.set_ylabel("Classe Real", fontsize=11, fontweight="bold")
ax.set_title(f"Matriz de Confusão - ResNet18\nLimiar de Youden = {limiar_resnet:.4f}", fontsize=11, fontweight="bold")

plt.colorbar(im)
plt.tight_layout()
plt.savefig(out_dir / "matriz_confusao_resnet.png", dpi=300)
plt.close()

print(f"Sucesso! Matriz da ResNet18 gerada em: {out_dir / 'matriz_confusao_resnet.png'}")
