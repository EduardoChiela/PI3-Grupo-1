# Critérios de Seleção — G3 · Classificação

**Projeto Integrador III — Câncer de Pulmão**
Etapa KDD: **Seleção (Patches LIDC-IDRI — nódulos)**
Documento de critérios e rastreabilidade · versão 1 · 31/08/2026

---

## 1. O que esta etapa entrega

Esta etapa define **quais nódulos entram no estudo e qual é o rótulo de cada um**. A saída é um número (o N) e um arquivo tabular — não é imagem. O recorte dos patches pertence à etapa seguinte, Transformação, e não altera o N definido aqui.

**Arquivo produzido:** `selecao/nodulos_selecionados.csv`
**Script:** `selecao/selecao_nodulos.py`

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

**Regra:** séries com cortes ausentes ou espaçamento irregular foram sinalizadas. Divergência acentuada entre `slice_spacing` (calculado a partir do `ImagePositionPatient`) e `slice_thickness` (declarado no cabeçalho) indica reconstrução sobreposta ou com lacuna.

**Justificativa:** espaçamento irregular desalinha o volume tridimensional e compromete qualquer reamostragem posterior.

**Efeito:** `[CONFIRMAR COM O RESPONSÁVEL PELA EXECUÇÃO]` — a contagem de 897 exames corresponde à aplicação do filtro de espessura. Confirmar se o espaçamento foi aplicado como exclusão adicional ou apenas verificado. Se foi apenas verificado, este item deve ser descrito como *verificação*, não como *critério de exclusão*.

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

**Composição final:** 388 benignos (51,66%) e 363 malignos (48,34%).

> **Nota sobre a leitura do fluxograma:** a unidade de contagem muda entre a segunda e a terceira linha. As duas primeiras contam exames de tomografia; as três seguintes contam nódulos. Um exame pode conter zero, um ou vários nódulos, de modo que os valores das duas metades não são comparáveis entre si.

**Verificação aritmética:** 2.291 − 1.099 = 1.192 · 1.192 − 441 = 751 · 388 + 363 = 751.

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

A pequena diferença no número de exames é esperada: o LUNA16 também excluiu séries com espaçamento inconsistente ou cortes ausentes, o que reduz ligeiramente o total.

---

## 9. Achados de qualidade de dados

Durante a conferência foram identificados agrupamentos com **mais de quatro anotações**, o que não deveria ocorrer, dado que cada exame foi lido por no máximo quatro radiologistas:

| Anotações no agrupamento | Ocorrências |
|---|---|
| 5 | 6 |
| 6 | 2 |
| 7 | 1 |
| **Total** | **9** (1,2% do conjunto) |

**Interpretação:** o `cluster_annotations()` agrupa por proximidade espacial usando um limiar de distância. Agrupamentos com mais de quatro anotações indicam que lesões distintas e próximas foram fundidas, ou que um mesmo radiologista marcou a mesma lesão duas vezes. É uma limitação conhecida da heurística de agrupamento, não um defeito do acervo.

**Encaminhamento:** `[DECIDIR]` — descartar as 9 linhas, ou mantê-las e declarar a limitação. Enquanto a decisão não for tomada, toda descrição do critério deve dizer **"consenso de ao menos 3 radiologistas"**, e nunca "3 de 4", uma vez que a segunda formulação é falsa para esses nove casos.

---

## 10. Ameaças à validade

Estas limitações não invalidam o trabalho; omiti-las, sim.

**T1 · O rótulo é percepção, não patologia.** A variável `malignancy` do LIDC é a probabilidade de malignidade **estimada pelo radiologista a partir da imagem**, não confirmação histopatológica. Um modelo treinado sobre ela aprende a reproduzir a avaliação radiológica, não a detectar câncer confirmado. Toda conclusão deve ser enunciada nesses termos.

**T2 · Viés de conspicuidade.** Exigir consenso de três ou mais radiologistas seleciona os nódulos mais evidentes. O desempenho não transfere diretamente para um cenário de rastreamento populacional.

**T3 · Remoção da zona cinzenta.** Excluir os 441 nódulos de mediana 3 elimina os casos clinicamente mais difíceis. A acurácia relatada é sistematicamente superior à que se obteria no problema completo.

**T4 · Corte de espessura.** Descartar séries acima de 2,5 mm remove protocolos mais antigos e limita a generalização a equipamentos e protocolos modernos.

**T5 · Agrupamento heurístico.** A correspondência entre anotações de radiologistas diferentes é inferida por proximidade e não possui verdade de referência. Ver seção 9.

**T6 · Coorte não representativa.** O LIDC-IDRI é retrospectivo e enriquecido em nódulos; a prevalência não corresponde à de uma população rastreada. Valor preditivo positivo calculado sobre este conjunto não possui leitura clínica direta.

**T7 · Múltiplos nódulos por paciente.** Nódulos do mesmo paciente compartilham anatomia, protocolo e equipamento. A divisão treino/validação/teste é feita por paciente, e os intervalos de confiança devem ser calculados sobre pacientes, não sobre nódulos.

---

## 11. Texto para a seção de Métodos

> Utilizou-se a coleção pública LIDC-IDRI (*The Cancer Imaging Archive*), composta por 1.010 pacientes e 1.018 exames de tomografia computadorizada de tórax, na qual quatro radiologistas torácicos anotaram lesões de forma independente em um processo de leitura em duas fases, sem imposição de consenso. As anotações foram acessadas por meio da biblioteca *pylidc*.
>
> Foram considerados elegíveis os exames com espessura de corte igual ou inferior a 2,5 mm, resultando em 897 exames. Entre as três categorias de achado definidas pelo protocolo do LIDC, apenas as lesões classificadas como "nódulo ≥ 3 mm" foram incluídas, por serem as únicas que receberam a pontuação de probabilidade de malignidade. As anotações independentes foram agrupadas por proximidade espacial, totalizando 2.291 agrupamentos, dos quais se retiveram os 1.192 identificados por pelo menos três radiologistas.
>
> O rótulo foi definido pela mediana das pontuações de malignidade do agrupamento: mediana inferior a 3 foi codificada como benigno e superior a 3 como maligno; os 441 nódulos com mediana igual a 3 foram excluídos por indeterminação. O conjunto final reuniu 751 nódulos, distribuídos em 388 benignos (51,66%) e 363 malignos (48,34%).
>
> A divisão em treino, validação e teste foi realizada no nível do paciente, estratificada por classe, de modo que nenhum paciente contribuísse com amostras para mais de um conjunto. Como conferência externa, os critérios adotados foram comparados aos do desafio LUNA16, que reporta 888 exames e 1.186 nódulos sob filtragem equivalente — diferenças de 1,0% e 0,5% em relação aos nossos resultados.
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
