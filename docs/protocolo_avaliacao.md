# Protocolo de avaliação — Issue #15

## Fluxo e separação das responsabilidades

**TREINO → VALIDAÇÃO → escolha do threshold → congelamento do threshold → TESTE FINAL.**

A Issue #14 é responsável pelo treinamento/baseline. `mineracao/avaliacao.py`
avalia predições já geradas e independe da arquitetura. Arquitetura, época,
hiperparâmetros, configuração de pesos e threshold devem estar definidos antes
de acessar o teste. O teste é utilizado somente para avaliação final, nunca para
ajustar qualquer dessas escolhas. Registrar essas decisões antes da avaliação.
O código rejeita teste na escolha do threshold e na comparação metodológica;
essa proteção depende de os splits fornecidos estarem corretamente identificados.

O manifesto `transformacao/manifest_patches.csv` contém 747 nódulos: 522 de
treino, 114 de validação e 111 de teste, separados por paciente. Suas colunas
são `idx`, `nodule_id`, `patient_id`, `scan_id`, `split`, `label`, `shape`.

## Entrada e uso

Dependências: `numpy`, `pandas` e `matplotlib` (esta última para figuras).
Instalação: `python -m pip install numpy pandas matplotlib`.
A tabela de predições deve conter uma linha por nódulo e as colunas
`nodule_id`, `patient_id`, `split`, `y_true`, `y_prob`.
`y_true` usa 0 para benigno e 1 para maligno; `y_prob` é a probabilidade
maligna entre 0 e 1. Splits: `treino`, `validacao`, `teste`.

Exemplo executado na raiz do repositório, após gerar as predições:

```python
import pandas as pd
from mineracao.avaliacao import (
    validar_com_manifesto, escolher_threshold_validacao,
    avaliar_com_threshold, bootstrap_por_paciente, gerar_curva_roc,
)

manifesto = pd.read_csv("transformacao/manifest_patches.csv")
validacao = validar_com_manifesto(pd.read_csv("predicoes_validacao.csv"), manifesto)
escolha = escolher_threshold_validacao(validacao)
threshold = escolha["threshold"]
gerar_curva_roc(validacao, "roc_validacao.png", threshold=threshold)
# Registrar e congelar threshold e todas as decisões do modelo ANTES do teste.
teste = validar_com_manifesto(pd.read_csv("predicoes_teste.csv"), manifesto)
resultado = avaliar_com_threshold(teste, threshold)
intervalos = bootstrap_por_paciente(teste, threshold, n_repeticoes=2000, seed=42)
```

`validar_com_manifesto` cruza por `nodule_id` e confere paciente, split e rótulo.
Permite subconjuntos: a cobertura completa do split deve ser conferida pelo
responsável antes de reportar a avaliação final. Nunca excluir casos após olhar
seus erros. As demais funções validam colunas, IDs, valores e vazamento de
pacientes na tabela recebida. `avaliar_com_threshold` não escolhe threshold.

## Métricas e ponto de operação

Recall/sensibilidade da classe maligna é a métrica principal e **sempre deve ser
reportado junto à especificidade no mesmo ponto de operação**. Reportar também
F1, AUC-ROC e TP, TN, FP, FN. AUC usa probabilidades, não classes previstas.
A regra de classificação é `y_prob >= threshold`.

O critério inicial é maximizar o índice de Youden na validação:
`J = sensibilidade + especificidade - 1`. Avaliam-se as probabilidades distintas
e os limites 0 e 1; em empate escolhe-se o maior threshold. A validação deve
conter as duas classes. O critério só poderá ser substituído por decisão
metodológica tomada **antes da avaliação final do teste**. Uma probabilidade
igual a 1 é positiva inclusive no threshold 1.

Denominadores zero produzem `NaN`, indicando métrica indefinida, sem inventar
um valor zero. AUC é indefinida quando há apenas uma classe. A geração de ROC
requer duas classes, marca o threshold fornecido e exige esse valor na validação;
a figura é fechada após salvar, inclusive se ocorrer erro de gravação.

## Intervalos de confiança por paciente

Usar IC 95% percentil (percentis 2,5 e 97,5) de 2000 réplicas por padrão, com
seed configurável. Em cada réplica são realizados N sorteios de pacientes com
reposição, onde N é o número de pacientes distintos do conjunto avaliado.
Cada ocorrência de um paciente inclui **todos** os seus nódulos:
se ele for sorteado duas vezes, seus nódulos aparecem duas vezes. Concatenar
os grupos preserva essa multiplicidade; somente `isin` não a preservaria.

Esse agrupamento é necessário porque múltiplos nódulos do mesmo paciente
compartilham anatomia, protocolo e equipamento e não são observações
independentes. O threshold fica congelado em todas as réplicas. Calcular IC
para recall, especificidade, F1 e AUC. Réplicas sem as duas classes não
contribuem à AUC; denominadores indefinidos são excluídos apenas da métrica
afetada. Reportar `replicas_validas` e `replicas_totais` para cada métrica.
Se não houver réplica válida, os limites são `NaN`; não apresentar isso como
um intervalo estimado. Os intervalos são condicionais ao modelo e threshold
fixos e não incorporam incerteza da seleção ou do treinamento.

## Faixa preta e pesos de classe

A análise de faixa preta é uma análise de subgrupo, usando o mesmo threshold
congelado para comparar nódulos com faixa contra os demais.
`analisar_faixa_preta` aceita uma coluna booleana `tem_faixa_preta` ou uma
coleção de `nodule_id`, sem índices hardcoded. Uma coleção deve representar
todos os casos identificados no conjunto avaliado. O manifesto atual ainda
não possui essa coluna; a identificação definitiva depende da Issue #13.
Subgrupo vazio retorna `n = 0` e métricas ausentes; classe única pode produzir
métricas indefinidas. Não há resultados de subgrupo disponíveis nesta issue.

`comparar_configuracoes` organiza as métricas de duas ou mais tabelas na
**validação**, sobre os mesmos nódulos e com thresholds explicitamente
fornecidos. Usar para comparar baseline com e sem pesos de classe; não treina
nem escolhe vencedor. A comparação/seleção entre configurações ocorre
exclusivamente na validação. Depois que as configurações e seus thresholds
estiverem congelados, os modelos com e sem pesos podem ser avaliados uma única
vez no teste para fins de reporte. Os resultados do teste não podem ser usados
para escolher o vencedor.
Os pesos calculados apenas no treino são próximos de 1 porque há equilíbrio
aproximado: 270 benignos e 252 malignos. Pela regra `N / (2 * n_classe)`,
são aproximadamente 0,9667 e 1,0357, respectivamente. Isso não antecipa
qualquer benefício de desempenho do balanceamento.

## Interpretação e comparação com literatura

Declarar que o conjunto exclui malignidade mediana igual a 3 e exige consenso
de **pelo menos 3 radiologistas**. A tarefa é menos ambígua que avaliações
sobre o LIDC completo; diferenças de seleção, split e ponto de operação
precisam acompanhar comparações com literatura. O alvo é a avaliação
radiológica de malignidade, não confirmação histopatológica de câncer.
Consultar também `criterios_selecao.md` e `criterios_transformacao.md`.

## Benchmarks da literatura

**Pendente até existirem resultados definitivos do baseline.** Nenhum valor de
benchmark é apresentado neste momento.

A futura comparação deve informar, para cada estudo, o conjunto utilizado,
o critério de rotulagem, a métrica, o valor publicado e as diferenças
metodológicas em relação a este projeto. Destacar que aqui são excluídos os
nódulos com malignidade mediana igual a 3 e exigido consenso de **pelo menos
3 radiologistas**.

Esta issue não contém números de desempenho de modelo. Dados sintéticos
utilizados nos testes servem exclusivamente à verificação técnica e não são
resultados científicos.
