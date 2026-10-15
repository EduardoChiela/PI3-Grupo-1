# Critérios de Mineração e Avaliação de Modelos

## 1. Arquitetura do Modelo Baseline
* **Arquitetura adotada**: Rede Neural Convolucional (CNN) rasa composta por 3 camadas convolucionais seguidas de pooling e camadas densas.
* **Justificativa**: Conforme estipulado na Issue #14, estabeleceu-se um modelo de referência simples, determinístico e de baixo custo computacional para dados 2.5D, avaliando seu desempenho antes e em paralelo à introdução de transfer learning com a ResNet18 (cujo backbone congelado alcançou AUC de 0,8269).
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
