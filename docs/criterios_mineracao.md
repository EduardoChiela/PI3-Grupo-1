# Critérios de Mineração e Avaliação de Modelos

## 1. Arquitetura do Modelo Baseline
* **Arquitetura adotada**: Rede Neural Convolucional (CNN) rasa composta por 3 camadas convolucionais seguidas de pooling e camadas densas.
* **Justificativa**: Conforme estipulado na Issue #14, estabeleceu-se um modelo de referência simples e de baixo custo computacional para dados 2.5D, avaliando seu desempenho antes e em paralelo à introdução de transfer learning com a ResNet18. Com o backbone congelado, a ResNet alcançou AUC de 0,8269; liberando `layer3` e `layer4`, alcançou AUC de 0,9294. Os resultados completos e a comparação pareada estão em `docs/resultados_comparacao.md`.
* **Entrada**: Tensores de dimensão (3, 64, 64) em float16, representando fatias centradas nos planos ortogonais axial, coronal e sagital.

## 2. Protocolo de Treinamento e Reprodutibilidade
* **Partição dos dados**: O conjunto total de N = 747 nódulos foi particionado estritamente conforme a coluna `split` do arquivo `transformacao/manifest_patches.csv`:
  * **Treino**: 522 nódulos (270 benignos e 252 malignos).
  * **Validação**: 114 nódulos (59 benignos e 55 malignos).
  * **Teste**: 111 nódulos (58 benignos e 53 malignos) provenientes de 69 pacientes únicos.
* **Regra de ouro do teste**: O conjunto de teste é acessado estritamente uma única vez. Toda escolha de arquitetura, ajuste de hiperparâmetros, definição de épocas e determinação de limiares operacionais é realizada exclusivamente no conjunto de validação.
* **Reprodutibilidade**:
  * Fixação de semente pseudoaleatória (seed) global.
  * Registro das versões das bibliotecas no cabeçalho dos scripts e logs.
  * Registro do hash SHA-256 do arquivo `.npz` utilizado no log de treinamento, garantindo rastreabilidade do conjunto de dados oficial corrigido.

## 3. Data Augmentation
* **Decisão adotada**: A rotação 2D e o jitter de 3 mm foram descartados; o espelhamento foi mantido, aplicado exclusivamente ao conjunto de treino, de forma estritamente coerente com o volume 3D (`AugmentacaoFlip3D`).
* **Justificativa técnica**:
  * Aplicar uma rotação 2D uniforme aos três canais gera uma distorção anatômica impossível, pois uma rotação em torno do eixo Z no plano axial corresponde a transformações completamente distintas no coronal e no sagital (mesma objeção que fundamentou a rejeição ao SMOTE). O jitter de 3 mm após a extração demandaria preenchimento artificial de borda (*padding*).
  * O espelhamento preserva a integridade física: espelhar o volume em um eixo equivale a espelhar, em cada canal, o eixo anatômico correspondente (axial = i, j; coronal = j, k; sagital = i, k), mantendo o canal perpendicular inalterado. Cada amostra sintética gerada é uma isometria geométrica de um nódulo real, sem interpolação de voxels. Como limitação declarada, o plano central situa-se no índice 32 de 64, incidindo a meio voxel do centro.
  * Validação e teste não recebem qualquer transformação.

## 4. Balanceamento e Pesos de Classe
* **Pesos calculados**: Utilizando a fórmula balanceada w = N / (2 * n), obtém-se peso **0,966667** para a classe benigna e **1,035714** para a maligna no conjunto de treino.
* **Impacto observado**: Como a distribuição real de treino alcançou 51,7% benignos contra 48,3% malignos (distância inferior a 4% do equilíbrio perfeito), o desbalanceamento severo previsto inicialmente não se materializou.

## 5. Protocolo de Avaliação (Issue #15)
* **Métrica principal**: **Recall da classe maligna** (sensibilidade clínica para não omitir lesões malignas), reportado **obrigatoriamente acompanhado da Especificidade** no mesmo ponto de operação.
* **Métricas complementares**: Área sob a Curva ROC (AUC-ROC) e F1-score.
* **Ponto de Operação**:
  * O limiar ótimo de decisão (tau) é determinado exclusivamente no conjunto de validação por meio do índice de Youden (J = Sensibilidade + Especificidade - 1), sendo subsequentemente aplicado ao teste sem qualquer ajuste a posteriori.
* **Intervalo de Confiança (IC de 95%)**:
  * Calculado via método de **bootstrap reamostrando por paciente (`patient_id`)** com **2.000 iterações**, evitando viés de dependência intra-sujeito entre os 111 nódulos de 69 pacientes no teste.
* **Análise de Patches com Faixa Preta**:
  * Os 15 patches afetados distribuem-se em 11 no treino, 0 na validação e 4 no teste. O cálculo de métricas isoladas para o subgrupo é estatisticamente inviável (sem casos na validação para ajustar o limiar e com apenas 4 casos no teste). Adotou-se a comparação da mediana da probabilidade predita desses 4 casos contra os demais 107 casos de teste, utilizando a reamostragem por paciente como referência de distribuição nula, conforme definido na seção 5.1 de `docs/criterios_transformacao.md`.

## 6. Seção de Métodos (Pronta para o Artigo Científico)
A tarefa de classificação da malignidade de nódulos pulmonares foi formulada como um problema supervisionado binário sobre cortes 2.5D ortogonais (axial, coronal e sagital) de dimensão 64 × 64 pixels. A partição do acervo (N = 747 nódulos) seguiu divisão pré-estratificada por paciente em treino (n = 522), validação (n = 114) e teste independente (n = 111 nódulos de 69 pacientes). O modelo baseline foi estabelecido por meio de uma CNN de 3 camadas convolucionais. A etapa de data augmentation restringiu-se a espelhamentos coerentes com o volume tridimensional aplicados exclusivamente ao treino, sendo rotações 2D excluídas por induzirem inconsistências topológicas entre os planos ortogonais. A avaliação priorizou o Recall da classe maligna acompanhado da Especificidade, com limiar ótimo determinado pelo índice de Youden na validação. Intervalos de confiança de 95% foram computados através de bootstrap em nível de paciente com 2.000 replicações.


### ResNet18 com transfer learning

Foram avaliadas três configurações de transfer learning utilizando os mesmos patches, splits e seed:

1. backbone completamente congelado, com apenas a camada final treinável;
2. `layer4` e camada final treináveis;
3. `layer3`, `layer4` e camada final treináveis.

A escolha da configuração foi realizada exclusivamente no conjunto de validação, utilizando o recall da classe maligna como métrica principal.

A configuração selecionada foi a terceira, com `layer3` e `layer4` liberados. Foram mantidos congelados `layer1` e `layer2`. Na validação, ela obteve recall 0,8909, especificidade 0,6949, F1 0,8033 e AUC 0,8502, superando `layer4` (recall 0,8000; AUC 0,8059) e o backbone congelado (recall 0,6182; AUC 0,7217). Os valores foram extraídos dos logs por `mineracao/resumo_finetuning.py`; a seleção não utilizou o conjunto de teste.

A camada final utilizou taxa de aprendizado de `0.001`, enquanto os blocos pré-treinados liberados utilizaram `0.0001`.

A avaliação no conjunto de teste foi realizada somente após a escolha definitiva da configuração.

### EfficientNet-B0 com transfer learning (Issue #44)

A segunda arquitetura pré-treinada é a EfficientNet-B0 com pesos ImageNet
`IMAGENET1K_V1`. A entrada permanece em 3 × 64 × 64, com os canais axial,
coronal e sagital mantidos na ordem já usada pela ResNet; não há interpolação
para 224 × 224. Normalização ImageNet, partições por paciente, seed, flip 3D
somente no treino, pesos de classe e demais parâmetros seguem o protocolo
anterior. BatchNorm permanece em avaliação durante o ajuste fino.

São comparadas três configurações: classificador treinável com backbone
congelado; último bloco (`features[7]`) e classificador treináveis; dois últimos
blocos (`features[6]` e `features[7]`) e classificador treináveis. A melhor
época de cada configuração é definida pela perda de validação, e a configuração
final é selecionada exclusivamente pelo maior recall maligno na validação.
O bloco de projeção `features[8]` permanece congelado em todas as configurações:
as variantes comparam apenas o descongelamento progressivo dos estágios MBConv
`[6]` e `[7]`, preservando a projeção ImageNet comum de 1280 canais. A decisão
de configuração foi registrada antes do treino; se recalls ficassem a menos de
0,01, o desempate pré-registrado seria maior especificidade e, depois, maior
AUC. `mineracao/resumo_efficientnet.py` verifica hash, tamanhos dos splits e
identidade dos nódulos de validação, e registra `escolha_validacao.json` com
`teste_utilizado_na_escolha: false`.

Na validação, os recalls foram 0,8364 (`congelado`), 0,7818 (`ultimo_bloco`)
e 0,7636 (`dois_ultimos_blocos`). O vencedor ficou 0,0545 acima do segundo,
portanto o desempate não foi aplicado. `congelado` teve especificidade 0,5763
e AUC 0,7587; a configuração com dois blocos teve maior especificidade
(0,8305) e AUC (0,8478), mas não venceu o critério pré-registrado.

Somente depois dessa escolha, `mineracao/avaliar_efficientnet.py` avalia a
configuração vencedora no teste. O script aplica os limiares de Youden e de
máximo F1 escolhidos na validação, calcula os pontos clínicos e seus intervalos
de 95% por bootstrap de paciente (2.000 réplicas), e bloqueia uma segunda
execução do teste. A avaliação final usou o limiar 0,3992 e obteve recall
0,8868, especificidade 0,6207, F1 0,7705 e AUC 0,8549, em 111 nódulos de 69
pacientes. As 2.000 réplicas por paciente foram válidas. O checkpoint foi
treinado com Python 3.13.7, torch 2.10.0+cpu e torchvision 0.25.0+cpu; a
inferência final foi executada no ambiente de referência CPython 3.12.10,
torch 2.14.1+cpu e torchvision 0.29.1+cpu. Essa diferença de versões fica
registrada como limitação da avaliação.

Os logs, predições, escolha e avaliação final estão em
`mineracao/resultados/efficientnet_s6/`. `mineracao/comparacao_efficientnet.py`
confere os artefatos e calcula diferenças pareadas por bootstrap de paciente,
McNemar em cluster e DeLong em cluster para baseline, ResNet e EfficientNet.

No código, todos os módulos `BatchNorm2d` são mantidos em modo de avaliação após cada chamada a `modelo.train()`. Assim, mesmo nas camadas liberadas, os pesos são atualizados, mas as estatísticas de normalização permanecem as do pré-treinamento ImageNet. Essa decisão evita atualizar estatísticas com lotes pequenos, mas pode limitar a adaptação ao domínio de tomografia.

Em duas execuções da mesma configuração final na CPU, com seed 42, os cabeçalhos dos logs e a primeira época foram iguais; a partir da segunda época, as trajetórias divergiram. A AUC de validação foi 0,8502 na execução de seleção e 0,8656 na execução final (diferença de 0,0154). Portanto, a seed e `torch.use_deterministic_algorithms` não garantiram repetibilidade numérica integral nesse ambiente. Não foi feita nova rodada para alterar esse comportamento; a limitação fica registrada e os resultados publicados correspondem aos artefatos versionados de cada execução.
