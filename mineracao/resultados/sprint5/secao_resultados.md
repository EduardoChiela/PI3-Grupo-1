<!-- inicio: issue31 -->
## Sprint 5 — Variabilidade do baseline (Issue #31)

Esta seção é gerada por `python -m mineracao.relatorio_sprint5` a partir dos
logs e predições versionados. O comando confere a integridade dos arquivos e
recalcula a AUC de cada rodada; não carrega imagens nem dados de teste.

### Cinco sementes no split original

Sementes: 42, 43, 44, 45, 46. Mantidos CNN rasa, Adam,
lr=0.001, batch=32, 30 épocas, pesos calculados no treino
e augmentation por flip 3D somente no treino. Muda apenas a semente.
O checkpoint escolhido minimiza a perda de validação, como no baseline.

Ambiente: Python 3.12.14, PyTorch 2.14.1+cpu, dispositivo cpu.
SHA-256 do NPZ: `31a84904c9c39c97d362c46cbfef353afaecf3f531e68a90c1bfbf51d2a73fdd`.
Versões e hashes completos: [protocolo](../mineracao/resultados/sprint5/protocolo.json).
As fontes exatas usadas nos treinos estão em `mineracao/resultados/sprint5/codigo_executado/`;
o patch de integração normaliza apenas o fim de arquivo das fontes originais.

| seed | AUC validação | Época | N treino | N validação | Pacientes validação |
|---|---:|---:|---:|---:|---:|
| 42 | 0.899538 | 27 | 522 | 114 | 70 |
| 43 | 0.876733 | 22 | 522 | 114 | 70 |
| 44 | 0.889368 | 25 | 522 | 114 | 70 |
| 45 | 0.887211 | 20 | 522 | 114 | 70 |
| 46 | 0.897381 | 30 | 522 | 114 | 70 |
| **Média** | **0.890046** | — | — | — | — |
| **Desvio-padrão amostral** | **0.009078** | — | — | — | — |

Mínimo–máximo: 0.876733–0.899538.

### Validação cruzada agrupada por paciente

Treino e validação originais somam 636 nódulos.
Usou-se StratifiedGroupKFold com 5 dobras e seed 42.
A guarda de separação por paciente de `mineracao/dataset.py` é reutilizada.
Cada paciente pertence a uma única dobra de avaliação; os pesos são calculados
somente no treino de cada dobra. As partições estão em
[dobras.csv](../mineracao/resultados/sprint5/dobras.csv).

A seed de treino permanece fixa. Na CV, avalia-se a época final 30,
fixada antes das execuções: a dobra externa não seleciona checkpoint nem limiar.
Essa regra difere da seleção de época das cinco sementes; portanto, as médias
dos dois experimentos não isolam apenas o efeito da partição.

| dobra | AUC validação | Época | N treino | N validação | Pacientes validação |
|---|---:|---:|---:|---:|---:|
| 1 | 0.945107 | 30 | 509 | 127 | 76 |
| 2 | 0.947591 | 30 | 509 | 127 | 83 |
| 3 | 0.927792 | 30 | 509 | 127 | 78 |
| 4 | 0.957278 | 30 | 509 | 127 | 76 |
| 5 | 0.909335 | 30 | 508 | 128 | 79 |
| **Média** | **0.937420** | — | — | — | — |
| **Desvio-padrão amostral** | **0.018963** | — | — | — | — |

Mínimo–máximo: 0.909335–0.957278.

### Comparação com e sem pesos

A rodada ponderada usa a primeira semente acima e é reutilizada, sem novo treino.
A rodada sem pesos usa a mesma semente e a mesma seleção de checkpoint.

| Configuração | AUC validação | Época selecionada |
|---|---:|---:|
| Com pesos | 0.899538 | 27 |
| Sem pesos | 0.905393 | 29 |

A saída completa do script existente `comparar_pesos.py` está em
[comparacao_pesos.txt](../mineracao/resultados/sprint5/comparacao_pesos.txt).
Uma única semente não sustenta declarar uma configuração superior à outra.

### A AUC histórica de teste é típica ou foi sorte?

As cinco sementes mostram baixa dispersão no split de validação fixo; a CV
mostra maior variação entre grupos de pacientes. Os desvios acima são amostrais
(ddof=1), não intervalos de confiança. As dobras compartilham dados de treino.

A AUC histórica de teste apresentada na seção de comparação continua sendo
uma medida de uma execução. Estes experimentos não estimam sua variância:
as novas métricas usam validação e outros grupos de pacientes. Não é correto
somar o desvio da validação à AUC de teste, nem concluir que o resultado de
teste foi sorte por comparar médias de conjuntos diferentes. A conclusão
sustentada é estabilidade entre sementes nesta validação e sensibilidade à
partição por paciente; a tipicidade daquele resultado de teste permanece
indeterminada sem violar o congelamento estabelecido para esta issue.

Os testes de rótulos embaralhados da issue #30, já integrados na seção 9,
não apontaram sinal de vazamento nesse controle. Isso não prova ausência de
toda forma de vazamento nem remove as limitações de seleção do conjunto.
Nenhuma rodada desta issue avaliou o teste; os arquivos da comparação histórica
permanecem inalterados.
<!-- fim: issue31 -->
