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

6. Métodos

Os volumes de tomografia foram transformados em patches 2.5D centrados no consenso das anotações. Inicialmente, foi recortada uma região de 96 mm 
ao redor do centroide do nódulo, mantendo o espaçamento original, seguida de reamostragem para 1 mm isotrópico por interpolação trilinear. Em seguida,
foi extraído o volume central de 64 mm, aplicando-se uma janela de intensidade de −1000 a 400 HU e normalização para o intervalo de 0 a 1. Os planos
axial, coronal e sagital foram utilizados como três canais, resultando em patches de 64 × 64 pixels, armazenados em float16. A representação 2.5D 
foi adotada para preservar informações espaciais complementares em diferentes orientações, mantendo uma representação menos complexa que uma entrada
tridimensional completa.
