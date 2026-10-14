# Critérios de Mineração e Avaliação de Modelos

## 1. Arquitetura do Modelo Baseline
* **Arquitetura adotada**: Rede Neural Convolucional (CNN) rasa composta por 3 camadas convolucionais seguidas de pooling e camadas densas.
* **Justificativa**: Conforme estipulado na Issue #14, buscou-se um modelo de referência simples, determinístico e de baixo custo computacional, capaz de estabelecer um piso de desempenho para dados 2.5D antes da introdução de transfer learning com redes mais profundas (como a ResNet).
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
* **Decisão adotada**: Data augmentation geométrica 2D (rotações e espelhamentos aleatórios) foi **descartada** para os patches 2.5D.
* **Justificativa técnica**:
  * Os três canais da imagem correspondem a planos anatômicos ortogonais distintos (axial, coronal e sagital).
  * Aplicar rotações 2D comuns sobre os 3 canais de forma uniforme geraria uma distorção anatômica impossível no espaço 3D (uma rotação em torno do eixo Z no plano axial implica transformações completamente diferentes nos planos coronal e sagital).
  * Isso recai na mesma objeção conceitual adotada pelo grupo ao rejeitar técnicas como o SMOTE ("interpolar pixels de tomografia gera volumes pulmonares inexistentes na anatomia humana").
  * A proposta inicial de *jitter* de 3 mm no centro é inexequível após a extração dos patches de 64 × 64, pois translações posteriores demandariam preenchimento artificial de bordas (*padding*) em vez de tecido pulmonar real.
  * Transformações e normalizações são restritas estritamente ao treino; validação e teste não recebem qualquer transformação.

## 4. Balanceamento e Pesos de Classe
* **Pesos calculados**: Utilizando a fórmula balanceada w = N / (2 * n), obtém-se peso **0,966667** para a classe benigna e **1,035714** para a maligna no conjunto de treino.
* **Impacto observado**: Como a distribuição real de treino alcançou 51,7% benignos contra 48,3% malignos (distância inferior a 4% do equilíbrio perfeito), o desbalanceamento severo previsto inicialmente não se materializou.

## 5. Protocolo de Avaliação (Issue #15)
* **Métrica principal**: **Recall da classe maligna** (sensibilidade clínica para não omitir lesões malignas), reportado **obrigatoriamente acompanhado da Especificidade** no mesmo ponto de operação.
* **Métricas complementares**: Área sob a Curva ROC (AUC-ROC) e F1-score.
* **Ponto de Operação**:
  * O limiar ótimo de decisão (tau) é determinado exclusivamente no conjunto de validação por meio do índice de Youden (J = Sensibilidade + Especificidade - 1), sendo subsequentemente aplicado ao teste sem qualquer ajuste a posteriori.
* **Intervalo de Confiança (IC de 95%)**:
  * Calculado via método de **bootstrap reamostrando por paciente (`patient_id`)** e não por nódulo.
  * O conjunto de teste possui 111 nódulos provenientes de 69 pacientes; nódulos de um mesmo paciente compartilham fatores clínicos, protocolo de escaneamento e ruído do tomógrafo, violando a premissa de independência estatística.
* **Análise de Patches com Faixa Preta**:
  * Avalia-se o desempenho isolado nos 15 patches com bordas cortadas em relação ao restante do teste, atestando se a ausência de preenchimento anatômico degrada a predição.

## 6. Seção de Métodos (Pronta para o Artigo Científico)
A tarefa de classificação da malignidade de nódulos pulmonares foi formulada como um problema supervisionado binário sobre cortes 2.5D ortogonais (axial, coronal e sagital) de dimensão 64 × 64 pixels. A partição do acervo (N = 747 nódulos) seguiu divisão pré-estratificada por paciente em treino (n = 522), validação (n = 114) e teste independente (n = 111 nódulos de 69 pacientes). O modelo baseline foi estabelecido por meio de uma CNN de 3 camadas convolucionais. Procedimentos de data augmentation geométrica 2D foram excluídos por gerarem inconsistências topológicas tridimensionais entre os planos anatômicos ortogonais. A avaliação priorizou o Recall da classe maligna em conjunto com a Especificidade, utilizando ponto de operação ótimo selecionado pelo índice de Youden na validação. Intervalos de confiança de 95% foram computados através de bootstrap em nível de paciente (1.000 iterações), mitigando correlações intra-sujeito.
