# Critérios de Seleção — G3 · Classificação

**Projeto Integrador III — Câncer de Pulmão**
Etapa KDD: **Seleção (Patches LIDC-IDRI — nódulos)**
Documento de critérios e rastreabilidade · versão 2 · 15/09/2026

> **Mudanças da versão 2 (Sprint 3, issue #4):** a seção 3.2 foi corrigida e medida — o filtro de espaçamento não havia sido aplicado nem verificado, ao contrário do que o texto anterior afirmava; agora tem número, e um nódulo foi excluído por ele. A seção 8 deixou de atribuir ao espaçamento a diferença inteira para o LUNA16, que a medição mostrou ser explicada apenas em parte. A seção 9 foi reescrita com o diagnóstico dos nove agrupamentos anômalos e a decisão tomada: três excluídos, seis mantidos. **O N final passou de 751 para 747.**

---

## 1. O que esta etapa entrega

Esta etapa define **quais nódulos entram no estudo e qual é o rótulo de cada um**. A saída é um número (o N) e um arquivo tabular — não é imagem. O recorte dos patches pertence à etapa seguinte, Transformação, e não altera o N definido aqui.

**Arquivo produzido:** `selecao/nodulos_selecionados.csv` (751 linhas — a saída bruta do script)
**Exclusões posteriores:** `selecao/excluidos_agrupamento_instavel.csv` (3 linhas — seção 9.1) e `selecao/excluidos_espacamento_irregular.csv` (1 linha — seção 3.2)
**Conjunto final:** as 747 linhas do manifesto que não constam de nenhum dos dois arquivos de exclusão
**Scripts:** `selecao/selecao_nodulos.py` · `selecao/diagnostico_agrupamentos_anomalos.py` · `selecao/verifica_espacamento.py`

> **Atenção de quem consome o manifesto.** O `nodulos_selecionados.csv` **não foi reescrito**: ele continua com as 751 linhas originais, porque é a saída direta do script e regravá-lo apagaria a rastreabilidade da execução. Quem usar o manifesto precisa filtrar pelos `nodule_id` dos dois arquivos de exclusão. Isso vale também para o `nodulos_com_split.csv` e para o manifesto de patches da etapa de Transformação.

---

## 2. Fonte dos dados

Coleção pública **LIDC-IDRI** (Lung Image Database Consortium and Image Database Resource Initiative), hospedada no *The Cancer Imaging Archive* (TCIA) sob licença **CC BY 3.0**.

| Item | Valor |
|---|---|
| Pacientes | 1.010 |
| Exames de TC (casos) | 1.018 |
| Estudos | 1.308 |
| Imagens | 244.527 |
| Volume DICOM | 133 GB |
| Anotações (XML) | 8,6 MB, arquivo separado |
| Radiologistas por exame | 4, torácicos, de forma independente |

As anotações foram acessadas pela biblioteca **pylidc**, que embute o banco de anotações. **Esta etapa não requer acesso às imagens DICOM** — a consulta às anotações dispensa a configuração do caminho das imagens (`~/.pylidcrc` sem a linha `path`, com `warn = False`). O acervo de imagem só passa a ser necessário na etapa de Transformação.

### Como as anotações funcionam

Os quatro radiologistas leram cada exame de forma independente, em duas fases: primeiro isoladamente, depois com acesso às marcações anonimizadas dos colegas, com direito a revisar a própria. **O consenso nunca foi forçado** — quem seguiu discordando teve a discordância registrada. Não existe campo de "resposta correta" no acervo; a verdade de referência é uma regra definida por quem usa a base.

Cada radiologista classificou cada achado em uma de três categorias, e o que ele anota depende da categoria:

| Categoria | O que o radiologista registra |
|---|---|
| Nódulo ≥ 3 mm | Contorno completo em cada corte + 9 atributos semânticos, entre eles malignidade de 1 a 5 |
| Nódulo < 3 mm | Apenas uma coordenada (x, y, z) de centro |
| Não-nódulo ≥ 3 mm | Apenas uma coordenada (x, y, z) de centro |

---

## 3. Critérios de elegibilidade do exame

### 3.1 Espessura de corte ≤ 2,5 mm

**Regra:** exames com `slice_thickness` superior a 2,5 mm foram excluídos.

**Justificativa:** nódulos de 3 a 6 mm ocupam menos de três cortes em protocolos espessos, o que impede caracterizar sua morfologia. Além disso, a reamostragem para voxel isotrópico prevista na etapa de Transformação passaria a interpolar estrutura que não foi adquirida — o algoritmo inventaria informação inexistente.

**Precedente:** é o mesmo critério do desafio **LUNA16**, o que permite comparar nossos resultados com a literatura.

**Efeito:** 1.018 → **897 exames** (121 excluídos).

### 3.2 Espaçamento entre cortes consistente

**Situação até a versão 1 deste documento:** o critério **não havia sido aplicado nem verificado**. A redação anterior — "séries com cortes ausentes ou espaçamento irregular foram sinalizadas" — descrevia uma checagem que não existia no código. A medição registrada mais abaixo foi feita na Sprint 3 e corrige isso.

A constatação é direta: o `selecao/selecao_nodulos.py` tem um único filtro de exame, `pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 2.5)`, e não carrega nenhum arquivo DICOM. O `ImagePositionPatient`, de onde o espaçamento real seria calculado, não é lido em ponto algum do script. **A contagem de 897 exames corresponde exclusivamente ao filtro de espessura.**

**Uma distinção que a redação anterior não fazia.** O texto tratava "reconstrução sobreposta ou com lacuna" como o mesmo defeito. São coisas diferentes:

| Fenômeno | O que é | Consequência |
|---|---|---|
| **Espaçamento irregular** | passo entre cortes não constante: lacuna, corte ausente, passo variável | quebra a premissa de passo constante da reamostragem isotrópica — **é defeito**, e é o que o LUNA16 excluiu |
| **Reconstrução sobreposta** | espaçamento menor que a espessura declarada, cortes se sobrepõem | escolha de protocolo que **aumenta** a resolução efetiva em Z — **não é defeito** e não justifica exclusão |

Só a primeira é candidata a critério de exclusão.

**Justificativa (mantida):** espaçamento irregular desalinha o volume tridimensional e compromete qualquer reamostragem posterior.

**Medição realizada.** O script `selecao/verifica_espacamento.py` mede o passo entre cortes de cada exame elegível. A medição **não requer o acervo DICOM**: o pylidc já guarda a posição Z de todos os cortes no próprio banco de anotações, acessível por `scan.slice_zvals`. Os 897 exames são analisados em cerca de 12 segundos.

| Situação | Exames | % |
|---|---:|---:|
| Regular — passo constante e igual à espessura | 601 | 67,0% |
| Sobreposto — passo menor que a espessura | 292 | 32,6% |
| **Irregular — passo variável ou corte ausente** | **4** | 0,4% |

**Os quatro exames irregulares:**

| Paciente | Espessura | Passo mediano | Maior desvio | Passos fora | Nódulos no manifesto |
|---|---:|---:|---:|---:|---:|
| `LIDC-IDRI-0514` | 2,5 mm | 2,0 mm | **36,0 mm** | 1 de 141 | 0 |
| `LIDC-IDRI-0267` | 2,5 mm | 2,5 mm | 2,0 mm | 2 de 149 | 1 |
| `LIDC-IDRI-0123` | 1,25 mm | 1,25 mm | 0,75 mm | 3 de 235 | 0 |
| `LIDC-IDRI-0672` | 1,25 mm | 0,625 mm | 0,625 mm | 3 de 461 | 1 |

**Critério adotado: a exclusão é por nódulo, não por exame.** O patch da etapa de Transformação usa apenas 64 mm em torno do centroide. Uma falha de espaçamento fora dessa janela não entra na reamostragem e não corrompe dado nenhum. Por isso, foi excluído **o nódulo cuja janela de 64 mm contém a falha**, e não todo nódulo de exame irregular:

| Nódulo | Distância do centroide até a falha | Dentro da janela de ±32 mm? | Decisão |
|---|---:|---|---|
| `LIDC-IDRI-0267_scan267_cluster000` | 31 cortes ≈ **78 mm** | não | mantido |
| `LIDC-IDRI-0672_scan983_cluster000` | 8 cortes ≈ **5 mm** | **sim** | **excluído** |

**Efeito:** 1 nódulo excluído. A lista está em `selecao/excluidos_espacamento_irregular.csv`; o relatório completo, em `selecao/verificacao_espacamento.md`.

> **Por que o critério não é "exame irregular sai".** Essa é a regra do LUNA16, e ela é razoável para quem monta um desafio de detecção sobre o volume inteiro. Aqui a unidade amostral é o nódulo e o dado que chega à rede é um cubo de 64 mm. Excluir o `LIDC-IDRI-0267` significaria descartar um nódulo maligno cuja vizinhança de 64 mm é perfeitamente regular. A regra por nódulo é mais estreita e precisa ser declarada como escolha nossa, porque diverge do precedente.

> **O que teria acontecido com a redação anterior.** O texto antigo mandava sinalizar "divergência acentuada entre `slice_spacing` e `slice_thickness`". Aplicado ao pé da letra, esse critério excluiria também os 292 exames de reconstrução sobreposta: **295 exames e 234 nódulos**, quase um terço do conjunto, descartados porque o protocolo entregou mais resolução em Z, não menos.

### 3.3 Exame sem nódulo elegível

Exames que não contêm nenhum nódulo ≥ 3 mm com consenso não contribuem amostra para a classificação. Eles continuam sendo relevantes para um pipeline de **detecção** — competência do G2 — mas não para esta etapa.

---

## 4. Critérios de elegibilidade da lesão

### 4.1 Apenas lesões da categoria "nódulo ≥ 3 mm"

**Regra:** somente lesões classificadas como nódulo de 3 mm ou mais entraram no conjunto.

**Justificativa:** é a **única categoria que recebeu escore de malignidade** no protocolo do LIDC. Nódulos menores e não-nódulos possuem apenas uma coordenada de centro, sem rótulo de qualquer espécie.

**Observação importante:** esta não é uma escolha metodológica do grupo — é uma restrição imposta pelo dado. Não existe configuração deste estudo em que essas lesões entrem, porque não há rótulo a ser aprendido. Consequentemente, **não há número de exclusão associado**: o pylidc só instancia objetos de anotação para nódulos ≥ 3 mm, de modo que as demais lesões sequer chegam a ser contabilizadas na consulta.

### 4.2 Agrupamento das anotações

As anotações são independentes: não há identificador ligando o nódulo marcado pelo radiologista 1 ao mesmo nódulo marcado pelo radiologista 3. O agrupamento foi feito pelo método `cluster_annotations()` do pylidc, que agrupa anotações por **proximidade espacial**. O número de anotações em cada agrupamento corresponde ao número de radiologistas que identificaram aquele nódulo.

**Efeito:** **2.291 agrupamentos** de nódulo ≥ 3 mm nos 897 exames elegíveis.

### 4.3 Consenso de ao menos 3 radiologistas

**Regra:** mantidos apenas os agrupamentos com 3 ou mais anotações.

**Justificativa:** o rótulo passa a ser apoiado pela maioria dos leitores, o que reduz o ruído de rotulagem, e a mediana fica bem definida com 3 ou 4 escores. Alinha o conjunto com o critério do LUNA16.

**Custo assumido:** viés de conspicuidade. Um nódulo identificado por três ou quatro radiologistas é, por construção, mais evidente. O desempenho medido será otimista em relação a um cenário real de rastreamento.

**Efeito:** 2.291 → **1.192 nódulos** (1.099 excluídos).

---

## 5. Definição do rótulo

### 5.1 Agregação

Cada radiologista atribui um escore de malignidade de **1** (altamente improvável) a **5** (altamente suspeito). Com 3 ou 4 escores por nódulo, adotou-se a **mediana**.

**Justificativa da mediana sobre a média:** a escala é ordinal — não se pode assumir que a distância entre 1 e 2 seja igual à distância entre 4 e 5 — e a mediana é robusta a um radiologista discordante.

### 5.2 Binarização

| Mediana | Rótulo |
|---|---|
| < 3 | Benigno |
| = 3 | **Excluído** (indeterminado) |
| > 3 | Maligno |

**Efeito da exclusão:** 1.192 → **751 nódulos** (441 excluídos, 37,0%).

**Custo assumido:** os nódulos de mediana 3 são precisamente os clinicamente ambíguos, aqueles em que os próprios especialistas não convergem. Removê-los torna o problema mais fácil do que o problema clínico real, e a métrica final é otimista por construção.

---

## 6. Fluxograma do conjunto

| Etapa | Unidade | Restantes | Excluídos | Motivo da exclusão |
|---|---|---|---|---|
| Acervo completo | exames | **1.018** | — | — |
| Exames elegíveis | exames | **897** | 121 | espessura de corte > 2,5 mm |
| Agrupamentos de nódulo ≥ 3 mm | nódulos | **2.291** | n/d | nódulos < 3 mm e não-nódulos não possuem escore e não são contabilizados |
| Consenso ≥ 3 radiologistas | nódulos | **1.192** | 1.099 | marcados por menos de 3 radiologistas |
| Nódulos rotulados | nódulos | **751** | 441 | malignidade mediana = 3 |
| Agrupamento estável | nódulos | **748** | 3 | rótulo dependente do limiar de agrupamento — seção 9.1 |
| **Conjunto final** | nódulos | **747** | 1 | falha de espaçamento dentro da janela do patch — seção 3.2 |

**Composição final:** 387 benignos (51,81%) e 360 malignos (48,19%).
**Divisão:** 522 treino · 114 validação · 111 teste.

> **Nota sobre a leitura do fluxograma:** a unidade de contagem muda entre a segunda e a terceira linha. As duas primeiras contam exames de tomografia; as demais contam nódulos. Um exame pode conter zero, um ou vários nódulos, de modo que os valores das duas metades não são comparáveis entre si.

**Verificação aritmética:** 2.291 − 1.099 = 1.192 · 1.192 − 441 = 751 · 751 − 3 = 748 · 748 − 1 = 747 · 387 + 360 = 747.

> **N metodológico e N operacional não são o mesmo número.** Os 747 acima são o que os critérios desta etapa produzem. Quantos deles têm imagem de fato em disco é outra pergunta, respondida na auditoria do acervo (`docs/auditoria_acervo.md`) e sujeita a mudar conforme a cópia do acervo utilizada. A seção de Métodos reporta o N metodológico; relatórios de execução da Sprint 3 reportam o operacional. Trocar um pelo outro no texto é erro de leitura do funil.

---

## 7. Estrutura do manifesto

`selecao/nodulos_selecionados.csv` — uma linha por nódulo elegível.

| Coluna | Origem | Finalidade |
|---|---|---|
| `patient_id` | `scan.patient_id` | Chave da divisão treino/validação/teste. Sem ela há vazamento entre conjuntos |
| `scan_id` | `scan.id` | Liga o nódulo à série de origem |
| `n_radiologistas` | `len(anns)` | Nível de consenso; permite análise de sensibilidade sem reprocessar |
| escores individuais | `ann.malignancy` | Escores por radiologista; permitem recalcular o rótulo sem reprocessar o acervo |
| `malignidade_mediana` | calculado | Rótulo agregado antes da binarização |
| `rotulo_binario` | calculado | 0 = benigno, 1 = maligno |
| `diametro_mm` | `ann.diameter` | Diâmetro médio do agrupamento; permite estratificar a análise de erro por tamanho |

---

## 8. Conferência contra a literatura

O LUNA16 publica os resultados de uma filtragem equivalente sobre a mesma coleção. A comparação é a principal evidência de que o pipeline está correto.

| Métrica | Nosso resultado | LUNA16 | Diferença |
|---|---|---|---|
| Exames elegíveis (espessura ≤ 2,5 mm) | 897 | 888 | +1,0% |
| Nódulos com consenso ≥ 3 | 1.192 | 1.186 | +0,5% |

**A explicação anterior para os 9 exames de diferença não se sustenta inteira.** O documento atribuía o gap ao filtro de espaçamento do LUNA16. A medição da seção 3.2 mostra que **apenas 4 exames** do nosso conjunto têm espaçamento irregular. Excluindo os quatro, ficaríamos com 893 — ainda **5 exames acima** dos 888.

O filtro de espaçamento explica menos da metade da diferença. As causas restantes não foram identificadas; candidatas plausíveis são divergências na versão do acervo, no tratamento de séries duplicadas ou no ponto de corte da espessura (`< 2,5` contra `≤ 2,5`). **Enquanto não forem verificadas, a diferença de 5 exames fica registrada como não explicada** — o que é preferível a atribuí-la a uma causa que a medição já descartou como suficiente.

---

## 9. Achados de qualidade de dados

### 9.1 Agrupamentos com mais de quatro anotações

Durante a conferência foram identificados agrupamentos com **mais de quatro anotações**, o que não deveria ocorrer, dado que cada exame foi lido por no máximo quatro radiologistas:

| Anotações no agrupamento | Ocorrências |
|---|---|
| 5 | 6 |
| 6 | 2 |
| 7 | 1 |
| **Total** | **9** (1,2% do conjunto) |

**Origem: o que o `cluster_annotations()` faz com esses casos.** O método já tenta resolvê-los sozinho. Quando algum agrupamento passa de quatro anotações, ele reduz a tolerância de distância em 10% e reagrupa, repetidamente, até que nenhum grupo exceda quatro — ou até a tolerância cair abaixo de 0,1 mm, quando desiste e imprime:

```
Failed to reduce all groups to <= 4 Annotations.
Some nodules may be close and must be grouped manually.
```

Estes nove são exatamente os exames em que ele desistiu. O `selecao_nodulos.py` chama o método com `verbose=False`, o que **silenciou esse aviso desde a primeira execução** — o problema só apareceu na conferência dos números, quando já estava no CSV.

**Por que a tolerância não resolve.** O agrupamento é feito por componentes conexas sobre a matriz de distâncias entre contornos. Nos nove casos, algumas anotações se tocam (distância exatamente 0,000 mm) e formam uma cadeia que mantém tudo conectado, por menor que seja a tolerância. No `LIDC-IDRI-0055_scan66_cluster000` o padrão é explícito: há dois blocos apertados — `{572, 577, 585}` e `{579, 584}` — separados por 13,24 mm, e uma anotação (`588`) cujo contorno encosta em **todas** as outras. Ela costura os dois blocos. Seu diâmetro é 26,9 mm, contra 8,7 a 12,7 mm de todas as demais.

**Teste de robustez aplicado.** Como qualquer limiar de "mesma lesão" é arbitrário, o rótulo de cada um dos nove foi recalculado sobre o bloco principal, variando o limiar de 2 a 10 mm. Um rótulo que muda conforme esse parâmetro não é sustentável:

| Nódulo | Rótulo atual | 2 mm | 3 mm | 4 mm | 5 mm | 6 mm | 8 mm | 10 mm | Veredito |
|---|---|---|---|---|---|---|---|---|---|
| `LIDC-IDRI-0204_scan205_cluster000` | maligno | maligno | maligno | maligno | maligno | maligno | maligno | maligno | estável |
| `LIDC-IDRI-0252_scan252_cluster000` | maligno | maligno | maligno | maligno | maligno | maligno | maligno | maligno | estável |
| `LIDC-IDRI-0608_scan613_cluster005` | maligno | maligno | maligno | maligno | maligno | maligno | maligno | maligno | estável |
| `LIDC-IDRI-0942_scan713_cluster000` | maligno | maligno | maligno | maligno | maligno | maligno | maligno | maligno | estável |
| `LIDC-IDRI-0865_scan790_cluster000` | maligno | maligno | maligno | maligno | maligno | maligno | maligno | maligno | estável |
| `LIDC-IDRI-0863_scan792_cluster001` | benigno | benigno | benigno | benigno | benigno | benigno | benigno | benigno | estável |
| `LIDC-IDRI-0055_scan66_cluster000` | maligno | excluído | excluído | excluído | excluído | benigno | maligno | maligno | **instável** |
| `LIDC-IDRI-0137_scan140_cluster002` | maligno | excluído | excluído | excluído | maligno | maligno | maligno | maligno | **instável** |
| `LIDC-IDRI-0815_scan840_cluster000` | maligno | excluído | excluído | excluído | excluído | maligno | maligno | maligno | **instável** |

Os seis estáveis compartilham a mesma estrutura: um bloco principal de três ou quatro anotações de diâmetro grande e semelhante, mais uma anotação bem menor e deslocada. No `LIDC-IDRI-0608`, o bloco principal tem quatro anotações de ~20 mm e a intrusa tem 6,4 mm. Sozinha, essa intrusa jamais passaria no consenso ≥ 3; ela entrou de carona e não desloca a mediana.

**Decisão (Sprint 3):** os **três agrupamentos instáveis foram excluídos** do conjunto final; os **seis estáveis foram mantidos**, com a limitação declarada nesta seção e em T5. A lista das linhas excluídas está em `selecao/excluidos_agrupamento_instavel.csv`, e a evidência completa — distâncias par a par e varredura de limiares — em `selecao/diagnostico_agrupamentos_anomalos.md`, gerada por `selecao/diagnostico_agrupamentos_anomalos.py`.

**Efeito:** 751 → **748 nódulos** (a exclusão desta seção; o conjunto final de 747 sai após a exclusão da seção 3.2). A divisão treino/validação/teste não precisou ser refeita — as linhas remanescentes mantêm a atribuição original.

**Limite desta análise.** Agrupar anotações por proximidade de centroide é heurística, não verdade de referência. O que o teste demonstra é que o rótulo dos três excluídos **não é robusto**, não que sejam comprovadamente duas lesões. Confirmar isso exigiria inspeção visual dos contornos — trabalho que não foi feito e cuja ausência não invalida a exclusão, já que a exclusão se justifica pela instabilidade em si.

**Formulação obrigatória.** Como seis agrupamentos com mais de quatro anotações permanecem no conjunto, toda descrição do critério deve dizer **"consenso de ao menos 3 radiologistas"**, nunca "3 de 4" — a segunda formulação é falsa para esses casos.

**Correção pendente no script.** O `selecao_nodulos.py` deve passar a chamar `cluster_annotations(verbose=True)`, ou registrar em log os exames em que o pylidc desiste, para que uma próxima execução não esconda o mesmo problema.

---

## 10. Ameaças à validade

Estas limitações não invalidam o trabalho; omiti-las, sim.

**T1 · O rótulo é percepção, não patologia.** A variável `malignancy` do LIDC é a probabilidade de malignidade **estimada pelo radiologista a partir da imagem**, não confirmação histopatológica. Um modelo treinado sobre ela aprende a reproduzir a avaliação radiológica, não a detectar câncer confirmado. Toda conclusão deve ser enunciada nesses termos.

**T2 · Viés de conspicuidade.** Exigir consenso de três ou mais radiologistas seleciona os nódulos mais evidentes. O desempenho não transfere diretamente para um cenário de rastreamento populacional.

**T3 · Remoção da zona cinzenta.** Excluir os 441 nódulos de mediana 3 elimina os casos clinicamente mais difíceis. A acurácia relatada é sistematicamente superior à que se obteria no problema completo.

**T4 · Corte de espessura.** Descartar séries acima de 2,5 mm remove protocolos mais antigos e limita a generalização a equipamentos e protocolos modernos.

**T5 · Agrupamento heurístico.** A correspondência entre anotações de radiologistas diferentes é inferida por proximidade e não possui verdade de referência. Seis agrupamentos do conjunto final reúnem mais de quatro anotações, acima do número de radiologistas por exame — foram mantidos após teste de robustez do rótulo, e três outros foram excluídos por instabilidade. Ver seção 9.1.

**T8 · Critério de espaçamento mais estreito que o do LUNA16.** A consistência do espaçamento foi medida (seção 3.2) e a exclusão foi feita **por nódulo**, não por exame: permanece no conjunto um nódulo de exame com espaçamento irregular, cuja falha está a 78 mm do centroide e portanto fora da janela de 64 mm do patch. O LUNA16 excluiria o exame inteiro. A escolha é defensável pela unidade amostral adotada, mas é uma divergência de precedente e reduz a comparabilidade direta.

**T9 · Diferença de 5 exames com o LUNA16 sem explicação.** Ver seção 8. Não compromete os resultados, mas impede afirmar que a filtragem é equivalente à do desafio.

**T6 · Coorte não representativa.** O LIDC-IDRI é retrospectivo e enriquecido em nódulos; a prevalência não corresponde à de uma população rastreada. Valor preditivo positivo calculado sobre este conjunto não possui leitura clínica direta.

**T7 · Múltiplos nódulos por paciente.** Nódulos do mesmo paciente compartilham anatomia, protocolo e equipamento. A divisão treino/validação/teste é feita por paciente, e os intervalos de confiança devem ser calculados sobre pacientes, não sobre nódulos.

---

## 11. Texto para a seção de Métodos

> Utilizou-se a coleção pública LIDC-IDRI (*The Cancer Imaging Archive*), composta por 1.010 pacientes e 1.018 exames de tomografia computadorizada de tórax, na qual quatro radiologistas torácicos anotaram lesões de forma independente em um processo de leitura em duas fases, sem imposição de consenso. As anotações foram acessadas por meio da biblioteca *pylidc*.
>
> Foram considerados elegíveis os exames com espessura de corte igual ou inferior a 2,5 mm, resultando em 897 exames. Entre as três categorias de achado definidas pelo protocolo do LIDC, apenas as lesões classificadas como "nódulo ≥ 3 mm" foram incluídas, por serem as únicas que receberam a pontuação de probabilidade de malignidade. As anotações independentes foram agrupadas por proximidade espacial, totalizando 2.291 agrupamentos, dos quais se retiveram os 1.192 identificados por pelo menos três radiologistas.
>
> O rótulo foi definido pela mediana das pontuações de malignidade do agrupamento: mediana inferior a 3 foi codificada como benigno e superior a 3 como maligno; os 441 nódulos com mediana igual a 3 foram excluídos por indeterminação, restando 751 nódulos.
>
> O agrupamento por proximidade produziu nove conjuntos com mais de quatro anotações, número superior ao de radiologistas por exame. Submetidos a um teste de robustez em que o rótulo foi recalculado sob limiares de agrupamento de 2 a 10 mm, seis mantiveram o mesmo rótulo em todas as condições e foram conservados, ao passo que três apresentaram rótulo dependente do limiar e foram excluídos.
>
> A consistência do espaçamento entre cortes foi verificada a partir das posições das fatias registradas no acervo: quatro exames apresentaram passo irregular, e foi excluído o único nódulo cuja janela de 64 mm contém a descontinuidade. O conjunto final reuniu **747 nódulos**, distribuídos em 387 benignos (51,81%) e 360 malignos (48,19%).
>
> A divisão em treino, validação e teste foi realizada no nível do paciente, estratificada por classe, de modo que nenhum paciente contribuísse com amostras para mais de um conjunto. Como conferência externa, os critérios adotados foram comparados aos do desafio LUNA16, que reporta 888 exames e 1.186 nódulos sob filtragem equivalente — diferenças de 1,0% e 0,5% em relação aos nossos resultados. Registre-se que o LUNA16 aplicou, adicionalmente, um filtro de consistência do espaçamento entre cortes que não foi replicado neste trabalho.
>
> Cabe destacar que a variável-alvo do LIDC-IDRI corresponde à probabilidade de malignidade estimada por radiologistas a partir da imagem, com variabilidade interobservador registrada, e não a confirmação histopatológica. Os resultados devem, portanto, ser interpretados como predição da avaliação radiológica.

---

## 12. Reprodutibilidade

```bash
pip install pylidc
```

Arquivo `~/.pylidcrc` (Linux/macOS) ou `pylidc.conf` em `%USERPROFILE%` (Windows):

```ini
[dicom]
warn = False
```

Verificação obrigatória antes de qualquer filtro:

```python
import pylidc as pl
print(pl.query(pl.Scan).count())   # deve retornar 1018
```

Se o valor for diferente de 1018, o banco de anotações não está completo e todos os números deste documento deixam de ser reproduzíveis.

Execução:

```bash
python selecao/selecao_nodulos.py
```

---

## 13. Referências

- Armato SG et al. (2011). *The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans.* Medical Physics, 38(2), 915–931.
- LIDC-IDRI — The Cancer Imaging Archive. https://www.cancerimagingarchive.net/collection/lidc-idri/
- LUNA16 — Data and reference standard. https://luna16.grand-challenge.org/Data/
- pylidc — documentação. https://pylidc.github.io/
