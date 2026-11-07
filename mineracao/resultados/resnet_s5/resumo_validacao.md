# Comparação das configurações de fine-tuning

Resultados extraídos automaticamente dos logs.

| Configuração | Parâmetros treináveis | Melhor época | Recall | Especificidade | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| congelado | 513 | 21 | 0.6182 | 0.7458 | 0.6538 | 0.7217 |
| layer4 | 8394241 | 11 | 0.8000 | 0.6949 | 0.7521 | 0.8059 |
| layer3_layer4 | 10493953 | 13 | 0.8909 | 0.6949 | 0.8033 | 0.8502 |

## Configuração escolhida

Critério de seleção: maior recall da classe maligna na validação.

Configuração escolhida: `layer3_layer4`.

Nenhuma informação do conjunto de teste foi utilizada nesta escolha.
