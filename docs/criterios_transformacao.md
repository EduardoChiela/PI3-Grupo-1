Critérios de Transformação

Etapa KDD: Transformação

Documento de critérios e justificativas

1. Objetivo

A etapa de Transformação converte os volumes de tomografia dos nódulos selecionados em uma representação padronizada para utilização no classificador.
A transformação parte dos nódulos definidos na etapa de Seleção e produz patches 2.5D com dimensão de 3 × 64 × 64, utilizando os planos axial, 
coronal e sagital. A sequência das operações foi definida para padronizar a escala física dos exames e reduzir diferenças causadas pelos diferentes 
espaçamentos de aquisição.
A implementação da transformação está localizada em transformacao/extrai_patches.py.

2. Critérios adotados

2.1 ROI inicial de 96 mm

Inicialmente é recortada uma região cúbica de 96 mm ao redor do centroide do nódulo, mantendo o espaçamento original do exame.
O recorte é definido em milímetros, e não em pixels, pois os exames possuem diferentes espaçamentos entre voxels. Dessa forma, a região representa
aproximadamente a mesma dimensão física independentemente da resolução original do exame.
A região de 96 mm também fornece uma margem para a etapa de reamostragem. Essa margem permite realizar a interpolação antes do recorte definitivo 
de 64 mm, evitando que a região utilizada no patch final fique dependente das bordas do recorte inicial.

2.2 Reamostragem para 1 mm isotrópico

Após o recorte da ROI, ela é reamostrada para 1 mm isotrópico utilizando interpolação trilinear.
A padronização da resolução faz com que a mesma quantidade de pixels represente aproximadamente a mesma dimensão física em todos os exames. Isso evita 
que diferenças no espaçamento original sejam interpretadas pelo modelo como diferenças anatômicas.
A reamostragem é realizada somente sobre a ROI, e não sobre o volume completo. Dessa forma, evita-se a criação de um grande volume intermediário que 
seria posteriormente descartado.
No código, a interpolação é realizada com interpolação linear em três dimensões, correspondente à interpolação trilinear.

2.3 Campo de visão de 64 mm

Após a reamostragem, é extraído o cubo central de 64 × 64 × 64 voxels, correspondente a aproximadamente 64 × 64 × 64 mm.
A escolha de realizar o recorte somente depois da reamostragem permite que os 64 voxels correspondam a aproximadamente 64 mm em cada dimensão. 
Isso mantém um campo de visão físico padronizado entre os exames, independentemente do espaçamento original.
O cubo final é obtido a partir da região de 96 mm, retirando 16 voxels de cada lado.

2.4 Janela de intensidade

Os valores de intensidade são limitados à faixa de −1000 a 400 HU e posteriormente normalizados para o intervalo de 0 a 1.
Valores inferiores a −1000 HU são tratados como −1000 HU e valores superiores a 400 HU são tratados como 400 HU. Em seguida, os valores são normalizados
de acordo com a faixa definida.
A utilização dessa janela limita a influência de valores extremos e estabelece uma faixa de intensidade padronizada para os patches.
Os volumes são obtidos por meio de scan.to_volume() e convertidos para float32. Não é aplicado novamente RescaleSlope ou RescaleIntercept durante 
essa etapa.

2.5 Representação 2.5D

A representação adotada é 2.5D, formada por três planos ortogonais do volume central: axial, coronal e sagital.
Cada plano possui dimensão de 64 × 64 pixels e os três são empilhados como canais, produzindo uma representação final de 3 × 64 × 64.
A escolha da representação 2.5D permite utilizar informações espaciais de três orientações diferentes sem utilizar todo o volume tridimensional 
como entrada. Dessa forma, a representação contém mais informações espaciais do que uma imagem 2D isolada, mantendo uma entrada menor do que um 
volume 3D completo.
A escolha do 2.5D não é justificada pelo uso de pesos pré treinados do ImageNet, pois o baseline desta Sprint não utiliza esse tipo de transferência
de aprendizado.

2.6 Armazenamento em float16

Após a normalização e a criação dos três canais, os patches são convertidos para float16.
A utilização de float16 reduz o espaço necessário para armazenar os patches, mantendo a representação numérica adequada para os dados que já 
foram normalizados para o intervalo de 0 a 1.
O uso de float16 também corresponde ao formato definido para a saída da Sprint e implementado no código de extração.

3. Ordem das transformações
A ordem das operações deve ser mantida para garantir que o recorte seja realizado considerando a escala física original do exame e que a representação
final seja padronizada, ao final, a ordem obedeceu os seguintes passos:

Volume original em HU;

Centroide obtido a partir do consenso;

Recorte da ROI de 96 mm no espaçamento original;

Reamostragem da ROI para 1 mm isotrópico;

Recorte do cubo central de 64 mm;

Aplicação da janela de −1000 a 400 HU;

Normalização para o intervalo de 0 a 1;

Extração dos planos axial, coronal e sagital;

Empilhamento dos três planos como canais;

Conversão para float16;

4. Centroide do nódulo

O centro utilizado para os recortes é obtido a partir do consenso das anotações dos radiologistas. O código utiliza consensus com clevel 
igual a 0.5, considerando como pertencente ao consenso a região presente em pelo menos 50% das segmentações.
A partir da máscara de consenso é calculada a média das coordenadas dos voxels pertencentes ao nódulo, obtendo o centroide utilizado no recorte da ROI.
Essa escolha mantém a localização do patch associada ao consenso das anotações utilizado na etapa de Seleção.

5. Limitações

A principal limitação da transformação está relacionada à resolução original dos exames.
Na etapa de Seleção foram considerados exames com espessura de corte de até 2,5 mm. O espaçamento no plano transversal é menor que a espessura dos cortes,
fazendo com que, após a reamostragem para 1 mm isotrópico, os planos coronal e sagital sejam majoritariamente obtidos por interpolação.
A reamostragem padroniza a grade espacial, mas não recupera informações que não foram originalmente adquiridas. Dessa forma, a resolução efetiva dos planos
coronal e sagital permanece inferior à do plano axial. De acordo com os critérios definidos para a Sprint, essa diferença pode representar uma resolução
real aproximadamente 3 a 5 vezes pior nesses planos.
Essa limitação deve ser considerada na interpretação dos resultados obtidos pelo modelo.

## 5.1 Patches com faixa preta na borda

### O que é

Quando a janela de 64 mm do patch ultrapassa o volume reconstruído, a região de fora não tem dado
adquirido e fica em −1000 HU. Depois da janela de intensidade e da normalização (§2.4), isso vira
**0,0 exato** e aparece como uma faixa preta na borda do patch.

São **15 patches de 747 (2,0%)**. O levantamento inicial é da seção 5 do
`transformacao/inspecao_visual.ipynb`; a medição aqui foi refeita de forma independente pelo
`transformacao/mede_faixa_preta.py`, que chega aos mesmos 15 e produz
`transformacao/patches_faixa_preta.csv`.

> **Zero não é sinônimo de falta de dado.** A janela mapeia −1000 HU para 0,0, e ar de pulmão está
> em −1000 HU ou abaixo — ou seja, **ar normal também vira 0,0**. Em 390 dos 732 patches sem
> nenhuma faixa existem pixels zerados, chegando a 21,8% do patch. Contar "pixels zero" como falta
> de dado superestima a faixa e mede a distância até o ar mais próximo, não até a borda do volume.
>
> O critério adotado: padding é o componente conexo de zeros que encosta numa linha ou coluna de
> borda **inteiramente** zerada. Tecido adquirido tem textura — vasos, ruído —, então uma linha de
> 64 pixels exatamente 0,0 não ocorre em pulmão real. Aplicado aos 732 patches sem faixa, esse
> critério marca **zero** pixels como padding. O script refaz essa conferência a cada execução e
> falha se ela quebrar.

### Duas causas diferentes, não uma

A faixa não tem origem única. Olhando em qual eixo anatômico ela aparece — o `extrai_patches.py`
monta axial = (Y, X), coronal = (X, Z) e sagital = (Y, Z), então uma faixa que aparece no coronal
**e** no sagital está no eixo Z —, os 15 se separam:

| Causa | Patches | Benigno | Maligno |
|---|---:|---:|---:|
| Borda do campo reconstruído — nódulo periférico, perto da parede torácica | 11 | 9 | 2 |
| Limite do exame — o patch passa da primeira ou última fatia adquirida | 4 | 1 | 3 |

### A lesão está inteira em todos os 15

A reamostragem para 1 mm isotrópico faz 1 pixel valer 1 mm, então dá para comparar diretamente a
distância do centro do patch até a faixa com o raio do nódulo (`diametro_mm` do
`selecao/nodulos_com_split.csv`):

| | menor valor | mediana |
|---|---:|---:|
| Largura da faixa | 1 px | 4 px |
| Padding no patch | 0,52% | 4,17% |
| Margem do centro até a faixa | 15,5 mm | 26,9 mm |
| **Folga (margem − raio do nódulo)** | **+9,0 mm** | +22,2 mm |

**Nenhum dos 15 tem a lesão tocada pela faixa.** O caso mais apertado é o idx 696, um nódulo de
35,0 mm cuja borda fica a 9,0 mm da faixa. O que falta nesses patches é contexto periférico, não
a lesão nem a margem imediata dela.

### Decisão: manter os 15

Os 15 permanecem no conjunto. **N continua 747.**

**O argumento não é o p-valor.** O teste de proporção do levantamento inicial dá p ≈ 0,30 para 10
benignos em 15, e seria errado ler isso como ausência de viés. Com n = 15 o poder é baixo: se a
proporção real de benignos no subgrupo fosse 70%, o teste só a detectaria em **30%** das vezes; a
60%, em 9%. O intervalo de confiança de 95% da proporção benigna vai de **0,384 a 0,882**, ou seja,
é compatível tanto com "não há viés" quanto com "quase 9 em cada 10 são benignos". O dado
disponível **não decide** se existe associação com a classe.

O que sustenta a decisão são três pontos verificáveis:

1. **Excluir introduz um viés pior do que o que evita.** Onze dos 15 são nódulos periféricos. Tirá-los
   do estudo remove sistematicamente uma apresentação anatômica real — é viés de seleção, e ao
   contrário do confundidor da faixa ele não é mensurável depois.
2. **A lesão está intacta nos 15.** O patch é uma observação válida do nódulo que o rótulo descreve;
   o que falta é contexto de fundo.
3. **Manter é reversível; excluir não é.** Mantendo, o efeito pode ser medido depois do treino e a
   exclusão ainda é possível, aí com evidência. Excluindo agora, a medição deixa de existir.

### Padrão que precisa ser testado, não afirmado

A tabela das duas causas sugere que a **direção** da faixa acompanha a classe: periférica tende a
benigno (9 de 11), limite do exame tende a maligno (3 de 4). Agrupando por paciente, Fisher exato
dá p = 0,041.

> **Esse p não vale como achado.** A divisão entre as duas causas foi encontrada olhando os dados e
> testada nos mesmos dados que a geraram, o que infla a significância e não controla quantas outras
> divisões poderiam ter sido tentadas. O valor dele é outro: mostra que a pergunta do levantamento
> inicial ("a faixa prediz a classe?") era menos informativa do que "a **direção** da faixa prediz a
> classe?". Fica registrado como hipótese a verificar, não como resultado.

No conjunto de teste os 4 casos seguem o mesmo padrão (os 2 periféricos são benignos, os 2 de
limite são malignos). São 4 de 111 — 3,6% do teste. Se um modelo aprendesse esse atalho e acertasse
os quatro em vez de dois, o ganho seria de cerca de **1,8 ponto percentual** de acurácia, abaixo da
largura típica do intervalo de confiança por bootstrap com 69 pacientes. Pequeno, mas é
contaminação de teste e precisa ser verificado.

### Teste obrigatório depois do treino, com o critério fixado antes

Esta decisão **não está encerrada** até que o teste abaixo seja executado e o resultado registrado
no PR do baseline (issue #14). O critério está escrito aqui antes de qualquer resultado existir,
justamente para que ninguém defina a régua depois de ver o número:

1. Identificar o subgrupo com `analisar_faixa_preta` (`mineracao/avaliacao.py`), passando a coluna
   `nodule_id` do `transformacao/patches_faixa_preta.csv` no parâmetro `nodule_ids`. O docstring
   dessa função diz que "a identificação definitiva depende da Issue #13" — é este CSV que fecha
   essa dependência.
   As **métricas** que ela devolve para o subgrupo não são interpretáveis com n = 4: servem só para
   conferir que a identificação casou com os dados de predição.
2. O teste de fato é outro, e precisa ser escrito no PR da #14: comparar a mediana de `y_prob` dos
   4 patches com faixa contra a dos outros 107. A referência de acaso vem de reamostrar 4 pacientes
   do teste 10.000 vezes e tomar o intervalo de 95% das diferenças de mediana.
3. **Se a diferença observada cair fora desse intervalo**, a faixa é declarada ameaça à validade na
   seção de Limitações do artigo, e a exclusão dos 15 volta à mesa com evidência.
4. **Se cair dentro**, fica registrado que a preocupação foi medida e descartada — com a ressalva
   explícita de que 4 casos de teste dão pouco poder, e que o resultado não prova ausência de efeito.
5. Na etapa de Interpretação, inspecionar o Grad-CAM desses 4 patches: se a ativação se concentrar
   na faixa em vez do nódulo, vale como evidência independente do item 3.

Reportar métrica separada para o subgrupo **não** é opção: são 11 casos no treino, **0 na validação**
e 4 no teste. Com zero na validação não há como escolher ponto de operação para o subgrupo, e com 4
no teste qualquer métrica teria intervalo de confiança maior que o próprio efeito.

### Onde estão os dados

- `transformacao/mede_faixa_preta.py` — script da medição, reexecutável, sem necessidade do acervo DICOM
- `transformacao/patches_faixa_preta.csv` — os 15 casos com `idx`, `nodule_id`, `patient_id`, `split`, `label`, `largura_px`, `area_padding_px`, `pct_padding`, `eixo`, `causa`, `diametro_mm`, `raio_mm`, `margem_mm`, `folga_mm` e `lesao_truncada`

6. Métodos

Os volumes de tomografia foram transformados em patches 2.5D centrados no consenso das anotações. Inicialmente, foi recortada uma região de 96 mm 
ao redor do centroide do nódulo, mantendo o espaçamento original, seguida de reamostragem para 1 mm isotrópico por interpolação trilinear. Em seguida,
foi extraído o volume central de 64 mm, aplicando-se uma janela de intensidade de −1000 a 400 HU e normalização para o intervalo de 0 a 1. Os planos
axial, coronal e sagital foram utilizados como três canais, resultando em patches de 64 × 64 pixels, armazenados em float16. A representação 2.5D 
foi adotada para preservar informações espaciais complementares em diferentes orientações, mantendo uma representação menos complexa que uma entrada
tridimensional completa.
