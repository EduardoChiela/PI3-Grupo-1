# EfficientNet-B0 — seleção na validação

Critério pré-registrado: maior recall maligno na validação; se a diferença entre os melhores for < 0,01, maior especificidade e depois maior AUC.

| Configuração | Parâmetros treináveis | Melhor época | Recall | Especificidade | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| congelado | 1281 | 18 | 0.8364 | 0.5763 | 0.7302 | 0.7587 |
| ultimo_bloco | 718513 | 11 | 0.7818 | 0.7458 | 0.7611 | 0.8040 |
| dois_ultimos_blocos | 2744861 | 9 | 0.7636 | 0.8305 | 0.7850 | 0.8478 |

Configuração escolhida: `congelado`.
O conjunto de teste não foi usado na escolha.
