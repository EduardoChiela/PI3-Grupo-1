import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. Matriz de Confusão Oficial do Teste (Total = 111 nódulos)
# Linha 0 (Real Benigno): 50 Verdadeiro Benigno (TN), 8 Falso Positivo (FP)
# Linha 1 (Real Maligno): 3 Falso Negativo (FN), 50 Verdadeiro Maligno (TP)
cm_baseline = np.array([[50, 8], [3, 50]])

# 2. Configuração do gráfico
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_baseline,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=True,
    annot_kws={'size': 14, 'weight': 'bold'},
    xticklabels=['Predito Benigno', 'Predito Maligno'],
    yticklabels=['Real Benigno', 'Real Maligno'],
)

plt.title(
    'Matriz de Confusão - Baseline (CNN Rasa)\nLimiar de Youden = 0,4305',
    fontsize=12,
    pad=12,
)
plt.xlabel('Classe Predita', fontsize=11, labelpad=8)
plt.ylabel('Classe Real', fontsize=11, labelpad=8)

# 3. Salvamento do arquivo no diretório oficial da Issue #33
os.makedirs('mineracao/resultados/interpretacao', exist_ok=True)
caminho_saida = (
    'mineracao/resultados/interpretacao/matriz_confusao_baseline.png'
)

plt.tight_layout()
plt.savefig(caminho_saida, dpi=300)
plt.close()

print(f'Imagem gerada com sucesso em: {caminho_saida}')

from IPython.display import Image, display

# Exibe a imagem salva no caminho
caminho_salvar = 'mineracao/resultados/interpretacao/matriz_confusao_baseline.png'
if os.path.exists(caminho_salvar):
    display(Image(filename=caminho_salvar))
else:
    print('Arquivo não encontrado no caminho especificado.')
