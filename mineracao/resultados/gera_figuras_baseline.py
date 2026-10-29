import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from mineracao.avaliacao import calcular_metricas

# 1. Carrega as predições e obtém o limiar oficial do JSON
pred = pd.read_csv("mineracao/resultados/comparacao/predicoes_teste_pareadas.csv")
limiar = json.loads(
    Path("mineracao/resultados/comparacao/resultados.json").read_text()
)["thresholds"]["baseline"]

# 2. Calcula as métricas usando a função oficial do repositório
m = calcular_metricas(pred.y_true, pred.p_baseline, limiar)

# 3. Autoconferência: interrompe se houver divergência com o recall e especificidade publicados
assert abs(m["recall"] - 0.943) < 0.001, f"Recall incorreto: {m['recall']}"
assert abs(m["especificidade"] - 0.862) < 0.001, f"Especificidade incorreta: {m['especificidade']}"

matriz = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]

# 4. Desenha a matriz usando matplotlib puro (sem seaborn/IPython)
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(matriz, cmap=plt.cm.Blues)
fig.colorbar(cax)

# Escreve os valores dentro das células
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(matriz[i][j]), va='center', ha='center', fontsize=14, fontweight='bold')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predito Benigno', 'Predito Maligno'])
ax.set_yticklabels(['Real Benigno', 'Real Maligno'])

plt.title(f'Matriz de Confusão - Baseline (CNN Rasa)\nLimiar de Youden = {limiar:.4f}')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')

# 5. Salva a imagem no diretório correto
caminho_saida = Path("mineracao/resultados/interpretacao/matriz_confusao_baseline.png")
caminho_saida.parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(caminho_saida, dpi=300)
plt.close()

print(f"Figura gerada e validada com sucesso em: {caminho_saida}")
