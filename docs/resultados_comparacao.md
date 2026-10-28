# Comparação baseline × ResNet — Issue #18

Números gerados por `python -m mineracao.comparacao` (seed 42, 2000 réplicas de
bootstrap). Saídas em `mineracao/resultados/comparacao/`: `tabela_comparativa.md`,
`resultados.json`, `roc_comparacao.png` e `predicoes_teste_pareadas.csv`.
O script não treina nem prevê: lê as predições de teste e os thresholds
congelados nos logs da #14 e da #17, confere tudo e só então calcula.

## 1. Conferência: os dois modelos foram avaliados do mesmo jeito

| Item | Baseline (#14) | ResNet (#17) |
|---|---|---|
| SHA-256 do `.npz` no log | `31a84904c9c39c97d362c46cbfef353afaecf3f531e68a90c1bfbf51d2a73fdd` | `31a84904c9c39c97d362c46cbfef353afaecf3f531e68a90c1bfbf51d2a73fdd` |
| Manifesto lido no log | `transformacao/manifest_patches.csv` | `transformacao/manifest_patches.csv` |
| Teste | 111 nódulos, 69 pacientes (53 malignos, 58 benignos) | idem, mesmos `nodule_id` |
| Threshold (Youden na validação) | 0,4305 | 0,5812 |
| Avaliações de teste no log | 1 | 1 |

- **Hash do `.npz`:** idêntico nos dois logs e igual ao `patches/patches_2_5d.npz`
  atual. A comparação vale.
- **Split:** os dois scripts leem o split da coluna `split` do manifesto via
  `mineracao/dataset.carregar_particoes`, sem sorteio próprio. O script de
  comparação passa as duas tabelas por `validar_com_manifesto` e exige que
  cubram exatamente os 111 nódulos de teste do manifesto, com os mesmos
  pacientes e rótulos. Os logs registram o caminho do manifesto, não o hash;
  o SHA-256 do manifesto atual (`207f44c7…16e4e0`) está em `resultados.json`.
- **CSV × log:** as métricas recalculadas de cada `predicoes_teste.csv` com o
  threshold do log batem exatamente com a linha `TESTE` do mesmo log, o que
  garante que predição e threshold vêm da mesma execução.
- **Teste usado uma vez:** cada modelo foi avaliado no teste uma única vez, com
  arquitetura, época e threshold congelados na validação. Este script só lê as
  predições gravadas; não reabre o teste.

> **Origem das predições do baseline.** A #14 não deixou `predicoes_teste.csv`
> no repositório (`mineracao/resultados/` está no `.gitignore`). As predições
> foram geradas nesta issue com a configuração padrão congelada da #14 —
> `python -m mineracao.treino_baseline --liberado-11 --npz patches/patches_2_5d.npz --avaliar-teste`
> (pesos de classe, flip 3D só no treino, 30 épocas, seed 42), que é a mesma
> configuração de perda e augmentation da ResNet. Nenhuma escolha foi feita
> olhando o teste. Ambiente: Python 3.13.2, torch 2.11.0 (CPU); o docstring da
> #14 cita torch 2.14.1, então pequenas diferenças numéricas são possíveis se
> outra pessoa reexecutar.

## 2. Decisão sobre independência (ameaça T7)

Os 111 nódulos vêm de 69 pacientes; 25 pacientes têm mais de um nódulo e
concentram 67 nódulos (até 6 em um paciente). **A análise agrega por paciente**:

- IC 95% de cada métrica: bootstrap percentil reamostrando pacientes
  (`avaliacao.bootstrap_por_paciente`, #15).
- IC 95% das diferenças: bootstrap **pareado** por paciente — o mesmo sorteio
  de pacientes é aplicado aos dois modelos em cada réplica.
- McNemar: versão para pares em cluster de Durkalski et al. (2003), com o
  paciente como cluster.
- DeLong: versão em cluster de Obuchowski (1997), que soma os componentes
  estruturais por paciente; reduz ao DeLong usual quando cada paciente tem um
  nódulo.

As versões por nódulo (McNemar exato e DeLong de 1988) aparecem só como
referência, marcadas como tais, porque assumem independência entre nódulos.

## 3. Tabela comparativa (teste, IC 95% por paciente)

Recall e especificidade estão no ponto de operação de cada modelo
(threshold de Youden escolhido na validação e congelado).

| Métrica | CNN rasa (baseline) | ResNet18 (transfer learning) | Diferença (ResNet − baseline) |
|---|---|---|---|
| Recall (maligno) | 0,943 [0,878; 1,000] | 0,679 [0,556; 0,803] | -0,264 [-0,375; -0,148] |
| Especificidade | 0,862 [0,765; 0,946] | 0,793 [0,684; 0,885] | -0,069 [-0,177; 0,035] |
| F1 | 0,901 [0,835; 0,951] | 0,713 [0,604; 0,810] | -0,188 [-0,285; -0,103] |
| AUC-ROC | 0,961 [0,927; 0,986] | 0,827 [0,743; 0,900] | -0,134 [-0,215; -0,060] |
| Limiar (Youden, validação) | 0,4305 | 0,5812 | — |
| TP / FN / TN / FP | 50 / 3 / 50 / 8 | 36 / 17 / 46 / 12 | — |

Todas as réplicas (2000/2000) foram válidas para todas as métricas.

## 4. Testes pareados

| Teste | Estatística | p-valor |
|---|---|---|
| **McNemar em cluster por paciente** (principal) | X² = 11,57; 22 nódulos só o baseline acerta, 4 só a ResNet; 22 pacientes discordantes | **0,0007** |
| McNemar exato por nódulo (referência) | b = 22, c = 4 | 0,0005 |
| **DeLong em cluster por paciente** (principal) | ΔAUC = −0,134, EP 0,039, IC 95% [−0,211; −0,057], z = −3,40 | **0,0007** |
| DeLong por nódulo (referência) | ΔAUC = −0,134, EP 0,037, z = −3,59 | 0,0003 |

O erro-padrão em cluster é maior que o por nódulo, como esperado com nódulos
correlacionados no mesmo paciente. O IC do DeLong em cluster coincide com o do
bootstrap pareado por paciente ([−0,215; −0,060]).

## 5. Curva ROC

![ROC dos dois modelos no teste](../mineracao/resultados/comparacao/roc_comparacao.png)

Os marcadores são o ponto de operação de cada modelo (círculo: baseline;
quadrado: ResNet).

## 6. Comparação com a literatura — com ressalva

> **Ressalva obrigatória.** Nosso conjunto exige que o nódulo tenha sido marcado
> por **pelo menos 3 radiologistas** e **exclui a malignidade mediana 3**. Os
> casos ambíguos saem do problema, que fica mais fácil do que o reportado em
> trabalhos sobre o LIDC completo. O alvo também é a opinião radiológica, não
> histopatologia. Números da literatura não são diretamente comparáveis.

| Estudo | Conjunto e rótulo | Divisão | Resultado publicado |
|---|---|---|---|
| Causey et al. (2018), *Sci. Rep.* 8:9286 — NoduleX | LIDC; média dos escores arredondada; nota 3 excluída; um nódulo sorteado por paciente | 80/20 por paciente, partição única | CNN: AUC 0,974 (1 × 4–5) e 0,938 (1–2 × 4–5); com atributos radiômicos: 0,993 e 0,971 |
| Al-Shabi, Lee e Tan (2019), *IEEE Access* — Gated-Dilated | LIDC-IDRI, > 1000 exames | ver artigo | AUC > 0,95 (resumo) |

O baseline (AUC 0,961) está na mesma faixa desses trabalhos, mas as diferenças
de seleção impedem dizer que ele "empata" com eles. O desenho 1 × 4–5 de Causey
é ainda mais separado que o nosso (exclui também a nota 2); o desenho
1–2 × 4–5 é o mais próximo e não exige consenso de 3 radiologistas. Muitos
trabalhos sobre LIDC dividem por nódulo e não por paciente, o que infla o
desempenho. A tabela deve ser ampliada por quem tiver acesso ao texto completo
dos artigos, informando para cada estudo o critério de rotulagem, a divisão e
a métrica — apenas os valores conferidos na fonte foram incluídos aqui.

## 7. Texto para a seção de Resultados

> No conjunto de teste (111 nódulos de 69 pacientes; 53 malignos), a CNN rasa
> obteve AUC-ROC de 0,961 (IC 95%: 0,927–0,986) e, no limiar escolhido na
> validação, recall de 0,943 (0,878–1,000) para a classe maligna com
> especificidade de 0,862 (0,765–0,946) e F1 de 0,901 (0,835–0,951). A ResNet18
> pré-treinada no ImageNet, com o extrator congelado, obteve AUC-ROC de 0,827
> (0,743–0,900), recall de 0,679 (0,556–0,803), especificidade de 0,793
> (0,684–0,885) e F1 de 0,713 (0,604–0,810). Todos os intervalos foram obtidos
> por bootstrap reamostrando pacientes. Os dois modelos foram treinados sobre o
> mesmo arquivo de patches (SHA-256 conferido nos dois registros de execução)
> e avaliados no mesmo split, definido por paciente.
>
> Como os modelos foram avaliados nos mesmos nódulos, a comparação foi pareada
> e agregada por paciente para respeitar a dependência entre nódulos do mesmo
> indivíduo. A diferença de AUC (ResNet − CNN rasa) foi de −0,134 (IC 95%:
> −0,211 a −0,057; DeLong em cluster, p < 0,001). No ponto de operação, o
> baseline acertou 22 nódulos que a ResNet errou e a ResNet acertou 4 que o
> baseline errou (McNemar em cluster, p < 0,001). A diferença concentrou-se na
> sensibilidade (−0,264; IC 95%: −0,375 a −0,148); a diferença de
> especificidade (−0,069; −0,177 a 0,035) não foi distinguível de zero.
>
> Não houve evidência de que o transfer learning, na configuração avaliada
> (extrator congelado e apenas a camada linear final treinada), ajude neste
> tamanho de amostra; ao contrário, a CNN rasa treinada do zero foi superior.
> Os resultados devem ser lidos à luz da seleção adotada: ao exigir consenso de
> pelo menos três radiologistas e excluir a malignidade mediana 3, o conjunto
> remove os casos ambíguos e é mais fácil que o LIDC-IDRI completo, de modo que
> os valores não são diretamente comparáveis aos de trabalhos que usam outros
> critérios.

## 8. Para a Discussão

- **O que foi comparado.** A ResNet da #17 treina só a camada final
  (513 parâmetros) sobre atributos do ImageNet. O resultado mostra que esses
  atributos, sem ajuste fino, não representam bem patches 2.5D de 64 × 64 cujos
  canais são planos anatômicos, não RGB. Ele **não** diz que uma ResNet com
  ajuste fino perderia para a CNN rasa — essa configuração não foi testada.
- **O resultado já era visível na validação** (AUC de validação 0,72 da ResNet
  contra 0,90 do baseline), portanto não depende do teste.
- **Teste contaminado para a ResNet.** Qualquer variante nova (descongelar
  camadas, redimensionar para 224) agora seria escolhida depois de ver o teste.
  Se for feita, precisa ser declarada como análise exploratória pós-hoc.

## 9. Testes de estresse do Checkpoint 2

O feedback do Checkpoint 2 considerou a AUC de 0,96 alta demais e pediu três
testes de estresse. O primeiro, o modelo só de diâmetro (AUC 0,9463), está na
PR #29. Esta seção traz os outros dois. Eles foram gerados por
`python -m mineracao.teste_estresse --npz patches/patches_2_5d.npz`, com seed 42
e 2000 réplicas de bootstrap. As saídas ficam em `mineracao/resultados/estresse/`:
`log_estresse.txt` e `resultados.json`. O SHA-256 do `.npz` no log é
`31a84904c9c39c97d362c46cbfef353afaecf3f531e68a90c1bfbf51d2a73fdd`, o mesmo da
#14 e da #18. Duas execuções seguidas deram números idênticos.

### 9.1 Rótulos embaralhados (validação)

**Desenho.** O split vem do manifesto, como na #14
(`dataset.carregar_particoes`). Os rótulos dos 522 nódulos de **treino** foram
permutados **depois** dessa divisão. A permutação mantém 252 malignos, então
os pesos de classe não mudam. Todo o resto é o baseline da #14: CNN rasa,
seed de treino 42, 30 épocas, lr 1e-3, batch 32, flip 3D só no treino, pesos
de classe, época escolhida pela menor perda de validação, CPU. Foram feitos 3
embaralhamentos (seeds 1, 2 e 3) e uma rodada de controle com os rótulos
verdadeiros. O controle mostra que o pipeline reproduz o baseline neste
ambiente; sem ele, uma AUC de 0,5 poderia indicar só um ambiente quebrado.
A avaliação usa só a validação (114 nódulos, 55 malignos). O teste não foi
usado.

**Previsto antes de rodar** (registrado no log antes do primeiro treino):
sem vazamento, a AUC de validação de cada embaralhamento cai em
0,5 ± 1,96 × EP nulo, ou seja, em **[0,393; 0,607]** (EP nulo = 0,054 para
55 × 59). O controle deve ficar perto da AUC de validação da #14 (0,898).

**O que saiu:**

| Rodada | Rótulos do treino inalterados | Época escolhida | AUC val. (época escolhida) | AUC val. (época 30) |
|---|---|---|---|---|
| Controle (rótulos verdadeiros) | 100% | 27 | 0,901 | 0,895 |
| Embaralhado, seed 1 | 52% | 7 | 0,570 | 0,487 |
| Embaralhado, seed 2 | 52% | 3 | 0,449 | 0,358 |
| Embaralhado, seed 3 | 45% | 9 | 0,581 | 0,461 |

O controle escolheu a mesma época da #14 (27) e chegou a AUC 0,901 contra
0,898 no log original. A pequena diferença vem do ambiente (torch 2.7.1 aqui,
2.11.0 na #14). Os três embaralhamentos ficaram dentro da faixa prevista, com
média de 0,533. A AUC da época escolhida é ligeiramente otimista, porque a
época é escolhida pela perda na própria validação. Na época 30 os três ficam
em 0,49, 0,36 e 0,46. O 0,36 fica abaixo da faixa, ou seja, abaixo do acaso,
na direção oposta à de um vazamento.

**O que significa.** Com os rótulos embaralhados a rede não acerta acima do
acaso. Não há evidência de canal escondido levando o rótulo até a entrada: nódulo
repetido entre treino e validação, ordem dos arquivos ou artefato do patch
correlacionado com o rótulo. Este teste **não** descarta que o problema seja
fácil por construção, que é a hipótese da seção 9.2.

### 9.2 Composição e desempenho por faixa de diâmetro (teste)

**Desenho.** A leitura é descritiva sobre o teste congelado. O script só lê
`predicoes_teste_pareadas.csv` (#18) e os diâmetros de
`selecao/nodulos_com_split.csv`, sem treinar, prever ou escolher nada. Os
cortes foram fixados na issue antes de olhar os resultados: pequeno
≤ 8,4 mm, médio 8,4–14,4 mm e grande > 14,4 mm. O limiar é o oficial do
baseline (0,4305, Youden na validação, #14).

**Previsto antes de rodar:** o diâmetro sozinho já dá AUC 0,9463 (PR #29),
então a fração de malignos deveria subir muito de uma faixa para outra. Só a
faixa do meio teria as duas classes em número suficiente para medir
desempenho.

**O que saiu — composição (resultado principal):**

| Faixa | Diâmetro (mm) | Nódulos | Pacientes | Malignos | Benignos | % malignos |
|---|---|---|---|---|---|---|
| Pequeno | 4,83 – 8,40 | 38 | 28 | **2** | 36 | 5% |
| Médio | 8,43 – 14,39 | 36 | 29 | 15 | 21 | 42% |
| Grande | 14,47 – 37,90 | 37 | 28 | 36 | **1** | 97% |
| Total | | 111 | 69 | 53 | 58 | 48% |

A soma de pacientes por faixa passa de 69 porque alguns pacientes têm nódulos
em mais de uma faixa.

> **Conferência da tabela da issue.** Os tercis exatos do teste são
> 8,387 mm e 14,417 mm. A tabela da issue usou esses tercis exatos (37/37/37
> nódulos). Os cortes registrados, arredondados para 8,4 mm com "≤", põem na
> faixa pequena um nódulo **benigno com exatamente 8,40 mm**
> (`LIDC-IDRI-0443_scan448_cluster000`). Por isso aqui saem 38/36/37. Valem
> os cortes registrados. A diferença é um único benigno com probabilidade
> prevista de 0,0007 e não muda a conclusão: com os tercis exatos, a faixa do
> meio dá AUC 0,852, o número da issue.

**O que saiu — desempenho na faixa do meio (CNN rasa, limiar 0,4305,
IC 95% por bootstrap por paciente, 2000/2000 réplicas válidas):**

| Métrica | Faixa do meio (36 nódulos, 29 pacientes) | Teste inteiro (seção 3) |
|---|---|---|
| AUC-ROC | 0,844 [0,692; 0,956] | 0,961 [0,927; 0,986] |
| Recall (maligno) | 0,933 [0,786; 1,000] | 0,943 [0,878; 1,000] |
| Especificidade | 0,762 [0,563; 0,929] | 0,862 [0,765; 0,946] |
| TP / FN / TN / FP | 14 / 1 / 16 / 5 | 50 / 3 / 50 / 8 |

**As faixas extremas não permitem medir desempenho.** A faixa pequena tem 2
malignos e a grande, 1 benigno. Com isso, a AUC é a fração de acertos de 2 × 36
ou 36 × 1 pares, e o recall ou a especificidade de cada faixa são calculados
sobre 1 ou 2 nódulos. Um único nódulo muda o resultado em 50 ou 100 pontos
percentuais. Esses números seriam ruído com cara de resultado, então não são
reportados.

**O que significa.** O rótulo é quase uma função do tamanho: 5% de malignos
entre os pequenos, 97% entre os grandes. É isso que sustenta o piso de 0,9463
do modelo só de diâmetro. Boa parte da AUC de 0,961 vem de separar pequenos
de grandes, o que o diâmetro já faz. Onde o tamanho não decide, na faixa do
meio, a AUC cai para 0,844, com IC largo (0,69–0,96) porque são 36 nódulos.
O recall se mantém (14 de 15 malignos), mas a especificidade cai para 0,76.
Somado à seção 9.1, o resultado alto não indica vazamento. Ele vem de um
problema fácil por construção: o critério de seleção (consenso de ≥ 3
radiologistas e exclusão da malignidade mediana 3) deixa malignos e benignos
muito separados em tamanho. A medida honesta da dificuldade "de verdade" é
a faixa do meio, e ela tem amostra pequena demais para uma estimativa
precisa. Nenhuma escolha de modelo, limiar ou época foi feita com base nesta
leitura.

## Referências

- Causey JL et al. Highly accurate model for prediction of lung nodule malignancy with CT scans. *Sci Rep*. 2018;8:9286. doi:10.1038/s41598-018-27569-w
- Al-Shabi M, Lee HK, Tan M. Gated-Dilated Networks for Lung Nodule Classification in CT Scans. *IEEE Access*. 2019. doi:10.1109/ACCESS.2019.2958663
- DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated ROC curves. *Biometrics*. 1988;44(3):837–845.
- Obuchowski NA. Nonparametric analysis of clustered ROC curve data. *Biometrics*. 1997;53(2):567–578.
- Durkalski VL, Palesch YY, Lipsitz SR, Rust PF. Analysis of clustered matched-pair data. *Stat Med*. 2003;22(15):2417–2428.
