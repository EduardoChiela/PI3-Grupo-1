<!-- inicio: issue45 -->
### Extensão da validação cruzada — Sprint 6 (Issue #45)

A validação cruzada usa exatamente as cinco dobras agrupadas por paciente
congeladas na Sprint 5 (`mineracao/resultados/sprint5/dobras.csv`). Para os dois
modelos, a seed de treino é 42 e a avaliação ocorre somente após a época 30,
fixada antes das execuções. A dobra externa não seleciona checkpoint nem
threshold.

A ResNet18 usa a configuração escolhida na Issue #32, com `layer3` e `layer4`
liberadas para fine-tuning. O conjunto de teste não é carregado nem avaliado
nesta rotina.

Sensibilidade e especificidade exigem ponto de operação. Como escolher um
threshold pela própria dobra externa introduziria viés, usa-se **threshold fixo
0,5 para ambos os modelos**. A AUC independe desse threshold.

#### Métricas por dobra e por modelo

| Modelo | Dobra | AUC | Sensibilidade | Especificidade |
|---|---:|---:|---:|---:|
| CNN rasa (baseline) | 1 | 0.945107 | 0.868852 | 0.893939 |
| ResNet18 (layer3+layer4) | 1 | 0.882762 | 0.524590 | 0.984848 |
| CNN rasa (baseline) | 2 | 0.947591 | 0.901639 | 0.893939 |
| ResNet18 (layer3+layer4) | 2 | 0.886985 | 0.737705 | 0.803030 |
| CNN rasa (baseline) | 3 | 0.927792 | 0.870968 | 0.830769 |
| ResNet18 (layer3+layer4) | 3 | 0.871712 | 0.725806 | 0.907692 |
| CNN rasa (baseline) | 4 | 0.957278 | 0.819672 | 0.924242 |
| ResNet18 (layer3+layer4) | 4 | 0.905862 | 0.819672 | 0.833333 |
| CNN rasa (baseline) | 5 | 0.909335 | 0.854839 | 0.727273 |
| ResNet18 (layer3+layer4) | 5 | 0.914223 | 0.822581 | 0.863636 |

#### Resumo das cinco dobras e IC 95%

| Modelo | Métrica | Média das 5 dobras | DP amostral | Estimativa OOF | IC 95% bootstrap por paciente |
|---|---|---:|---:|---:|---:|
| CNN rasa (baseline) | AUC | 0.937420 | 0.018963 | 0.934794 | [0.915601; 0.953063] |
| CNN rasa (baseline) | Sensibilidade | 0.863194 | 0.029726 | 0.863192 | [0.819041; 0.904766] |
| CNN rasa (baseline) | Especificidade | 0.854033 | 0.078619 | 0.854103 | [0.813250; 0.893230] |
| ResNet18 (layer3+layer4) | AUC | 0.892309 | 0.017375 | 0.888578 | [0.862842; 0.913164] |
| ResNet18 (layer3+layer4) | Sensibilidade | 0.726071 | 0.121249 | 0.726384 | [0.674183; 0.778217] |
| ResNet18 (layer3+layer4) | Especificidade | 0.878508 | 0.070920 | 0.878419 | [0.838616; 0.914915] |

A média e o desvio-padrão são calculados sobre as cinco dobras. O IC 95% é
percentil por bootstrap de pacientes (2000 réplicas, seed 42) sobre as
predições out-of-fold concatenadas; portanto, o intervalo é do desempenho OOF
agregado, e não um intervalo t que trate as cinco dobras como independentes.
Cada paciente é reamostrado como unidade para preservar o agrupamento dos
nódulos.

Todos os números acima são regenerados por
`python -m mineracao.relatorio_sprint6`; o script lê somente predições de
validação versionadas e não acessa imagens nem o conjunto de teste.
<!-- fim: issue45 -->
