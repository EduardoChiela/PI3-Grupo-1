<!-- inicio: pontos_operacao -->
## 12. Sprint 6 — Pontos de operação clínicos

Esta seção é gerada por `python -m mineracao.pontos_operacao` (seed 42,
2000 réplicas de bootstrap por paciente). Saídas em
`mineracao/resultados/pontos_operacao_s6/`: `resultados.json` e `secao_resultados.md`. O script não
treina nem prevê. Ele lê as predições de teste pareadas da §3
(`comparacao_s5/predicoes_teste_pareadas.csv`) e confere que elas reproduzem a
linha `TESTE` de cada log. Também confere que cada CSV de validação reproduz o
limiar de Youden do mesmo log, o que prova que vem da mesma execução.

O documento de instruções define três métricas primárias: AUC-ROC (§3),
**sensibilidade com especificidade fixada em 0,90** e **especificidade com
sensibilidade fixada em 0,85**. As duas últimas estão aqui, junto com o F1 no
limiar que o documento pede.

### 12.1 Regra de escolha do ponto

Com 111 nódulos, a ROC do teste é escalonada e
quase nunca passa exatamente por 0,90 ou 0,85. A regra adotada é:

1. **Limiares candidatos:** as probabilidades previstas no teste (mais 0 e 1),
   com "probabilidade ≥ limiar é maligno", a mesma convenção da §3.
2. **Restrição como piso:** especificidade **≥ 0,90** (ou sensibilidade
   **≥ 0,85**). Entre os limiares que cumprem o piso, fica o que dá a **maior**
   métrica-alvo. Como a curva anda em degraus, o ponto escolhido fica com a
   métrica fixada acima do piso (CNN: especificidade
   0,948), e não exatamente em 0,90.
3. **Empate:** vários limiares dão a mesma métrica-alvo. Fica o de **melhor
   métrica acompanhante** (o ponto não dominado da ROC: mesma especificidade e
   mais malignos detectados, por exemplo).
4. **Sem interpolação.** Interpolar entre os dois vértices vizinhos dá um ponto
   que nenhum limiar real produz. Como referência, a interpolação daria
   sensibilidade 0,811 (CNN) e
   0,774 (ResNet) em especificidade 0,90, e
   especificidade 0,879 (CNN) e
   0,897 (ResNet) em sensibilidade 0,85. Nos
   quatro casos o cruzamento cai num trecho plano da escada, então as duas
   regras coincidem nestes dados. Em outra amostra poderiam divergir.
5. **IC 95%:** bootstrap reamostrando pacientes, como na §3. Em cada réplica o
   ponto é **reescolhido** pela mesma regra, porque o que se estima é "a
   sensibilidade quando a especificidade é fixada", e não o desempenho de um
   limiar fixo.

Os limiares destes pontos foram **lidos na curva do teste**. Eles descrevem a
curva e não são limiares congelados na validação como o de Youden. Usá-los
para operar o modelo daria um resultado otimista.

### 12.2 Tabela (teste, IC 95% por paciente)

| Ponto de operação (teste) | CNN rasa (baseline) | ResNet18 (`layer3` + `layer4`) |
|---|---|---|
| **Sensibilidade com especificidade ≥ 0,90** | 0,811 [0,685; 0,982] | 0,774 [0,627; 0,961] |
| ↳ especificidade no ponto (limiar) | 0,948 (0,6863) | 0,914 (0,7199) |
| ↳ TP / FN / TN / FP | 43 / 10 / 55 / 3 | 41 / 12 / 53 / 5 |
| **Especificidade com sensibilidade ≥ 0,85** | 0,879 [0,792; 0,984] | 0,897 [0,750; 0,967] |
| ↳ sensibilidade no ponto (limiar) | 0,906 (0,5012) | 0,887 (0,5355) |
| ↳ TP / FN / TN / FP | 48 / 5 / 51 / 7 | 47 / 6 / 52 / 6 |
| Referência: Youden da validação (§3), sens / espec (limiar) | 0,943 / 0,862 (0,4305) | 0,906 / 0,810 (0,3614) |

Réplicas válidas: no mínimo 2000 de 2000 em cada linha.

**A ordem entre os modelos muda conforme o critério.** Com especificidade fixada
em 0,90, a CNN tem a sensibilidade maior (0,811 contra
0,774). Com sensibilidade fixada em 0,85, a ResNet tem
a especificidade maior (0,897 contra
0,879). As curvas se cruzam: nenhum modelo domina o
outro em todos os pontos de operação, apesar de a CNN ter a AUC maior. As
diferenças são pequenas perto da largura dos intervalos e não foram testadas
estatisticamente nesta seção.

### 12.3 F1 no limiar de máximo F1 da validação

A §3 reporta o F1 no limiar de Youden. O documento pede o F1 no limiar que
**maximiza o F1 na validação**. Esse limiar é escolhido só na validação
(empate: maior limiar, a mesma convenção do Youden) e aplicado congelado no
teste. O IC usa o limiar fixo, como na §3.

| F1 no teste (IC 95% por paciente) | CNN rasa (baseline) | ResNet18 (`layer3` + `layer4`) |
|---|---|---|
| F1 no limiar de **Youden** (o reportado na §3) | 0,901 [0,835; 0,951] | 0,857 [0,781; 0,923] |
| ↳ limiar (Youden, validação) | 0,4305 | 0,3614 |
| F1 no limiar de **máximo F1 da validação** (pedido no documento) | 0,901 [0,835; 0,951] | **pendente** — `predicoes_validacao.csv` da rodada final não versionado (ver abaixo) |
| ↳ limiar (máximo F1, validação) | 0,4305 | pendente |
| ↳ F1 na própria validação nesse limiar | 0,870 | pendente |
| ↳ TP / FN / TN / FP no teste nesse limiar | 50 / 3 / 50 / 8 | pendente |

Na CNN rasa (baseline), o máximo de F1 na validação (0,8696) é atingido num único limiar, e ele é **o mesmo do Youden**. Por isso os dois F1 do teste são idênticos. Os dois limiares são calculados separadamente, cada um pelo seu critério, sobre as mesmas predições de validação.

> **Pendência — ResNet18 (`layer3` + `layer4`).** O arquivo `mineracao/resultados/resnet_s5/final/predicoes_validacao.csv` não está no repositório: a rodada final (#32, commit `d468daf`) versionou só o log e as predições de teste, e o `modelo_resnet.pt` também não foi guardado. Sem as probabilidades da validação não há como achar o limiar de máximo F1. Retreinar não serve: duas execuções com a mesma seed divergem (§9), então o limiar seria de outro modelo. Escolher o limiar no teste também não serve, porque seria escolher olhando o teste. Quando o CSV original for commitado, basta rodar `python -m mineracao.pontos_operacao` de novo: o script confere se ele reproduz o limiar de Youden do log e preenche esta célula.

### 12.4 Ponto de operação adotado pelo grupo (atividade 4.4)

**O grupo mantém o limiar de Youden escolhido na validação** (CNN rasa:
0,4305), com sensibilidade 0,943 e
especificidade 0,862 no teste. Isso fica **abaixo** dos
0,90 de especificidade que o documento trata como o ponto clinicamente
relevante, e a escolha é consciente. No ponto exigido (especificidade ≥ 0,90),
a sensibilidade cai de 0,943 para 0,811.
Nos 53 malignos do teste, isso significa
10 cânceres não detectados em vez de 3
(7 a mais), para evitar 5 falsos positivos
(3 em vez de 8 nos 58
benignos).

O trade-off que o grupo aceita é este: **errar um maligno custa mais que errar
um benigno.** O falso negativo atrasa o diagnóstico de um câncer. O falso
positivo leva a exames de seguimento (nova tomografia, eventualmente biópsia),
que têm custo e risco, mas são recuperáveis. Por isso o grupo prioriza a
sensibilidade e aceita especificidade abaixo de 0,90. O Youden também cumpre o
piso de sensibilidade de 0,85 com folga.

O limiar de Youden foi escolhido na validação e aplicado congelado. Os pontos
da §12.2 foram lidos na própria curva do teste. Então a comparação favorece os
pontos da §12.2, e a perda de sensibilidade real ao exigir especificidade 0,90
tende a ser pelo menos a mostrada aqui. A decisão vale para este conjunto, mais
fácil que o LIDC completo (§6 e §10). Em uso clínico real, o limiar teria de
ser recalibrado numa população de referência.
<!-- fim: pontos_operacao -->
