from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


# ============================================================
# CONFIGURAÇÕES DO PROJETO
# ============================================================

# Menor valor da janela de HU.
# Tudo abaixo de -1000 será transformado em -1000.
HU_MIN = -1000.0

# Maior valor da janela de HU.
# Tudo acima de 400 será transformado em 400.
HU_MAX = 400.0


# Primeiro recortamos uma região maior de 96 mm ao redor
# do nódulo.
#
# Essa folga é importante porque depois vamos interpolar
# o volume para 1 mm isotrópico.
ROI_MM = 96


# Patch final desejado pela Sprint:
#
# 64 mm x 64 mm x 64 mm
PATCH_MM = 64


# Depois da reamostragem queremos:
#
# 1 voxel = 1 mm
TARGET_SPACING_MM = 1.0

def recortar_roi_mm(
    volume,
    centro,
    espacamento,
    tamanho_mm=ROI_MM
):
    """
    Recorta uma região cúbica ao redor do nódulo.

    IMPORTANTE:
    O tamanho é definido em milímetros e não em pixels.

    Parâmetros
    ----------
    volume:
        Volume 3D do exame.

    centro:
        Coordenadas do centro do nódulo no volume.
        Formato esperado:
        [i, j, k]

    espacamento:
        Tamanho físico de cada voxel em milímetros.
        Exemplo:
        [0.7, 0.7, 2.0]

    tamanho_mm:
        Tamanho físico da ROI.
        Por padrão usamos 96 mm.
    """

    # Garante que os valores sejam arrays do NumPy.
    centro = np.asarray(
        centro,
        dtype=float
    )

    espacamento = np.asarray(
        espacamento,
        dtype=float
    )


    # --------------------------------------------------------
    # CONVERTE MILÍMETROS PARA VOXELS
    # --------------------------------------------------------
    #
    # Imagine:
    #
    # spacing = 0.7 mm
    #
    # Para ter aproximadamente 96 mm:
    #
    # 96 / 0.7 = 137 voxels
    #
    # Já no eixo Z:
    #
    # spacing = 2 mm
    #
    # 96 / 2 = 48 slices
    #
    # Por isso não podemos simplesmente pegar
    # "96 pixels" em todos os eixos.
    tamanho_vox = np.ceil(
        tamanho_mm / espacamento
    ).astype(int)


    # --------------------------------------------------------
    # CALCULA ONDE O RECORTE COMEÇA
    # --------------------------------------------------------
    #
    # Queremos deixar o centro do nódulo aproximadamente
    # no centro da ROI.
    inicio = np.floor(
        centro - (tamanho_vox - 1) / 2
    ).astype(int)


    # Onde o recorte termina.
    fim = inicio + tamanho_vox


    # --------------------------------------------------------
    # CRIA UMA ROI VAZIA
    # --------------------------------------------------------
    #
    # Inicializamos tudo com -1000 HU.
    #
    # -1000 HU representa aproximadamente ar.
    #
    # Isso também serve como padding se o nódulo estiver
    # perto da borda da imagem.
    roi = np.full(
        tuple(tamanho_vox),
        HU_MIN,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # EVITA ACESSAR FORA DO VOLUME
    # --------------------------------------------------------

    # Limite inicial dentro do volume original.
    origem_inicio = np.maximum(
        inicio,
        0
    )

    # Limite final dentro do volume original.
    origem_fim = np.minimum(
        fim,
        np.asarray(volume.shape)
    )


    # Descobre em que posição da ROI os dados
    # do volume original devem ser colocados.
    destino_inicio = (
        origem_inicio - inicio
    )

    destino_fim = (
        destino_inicio
        + (origem_fim - origem_inicio)
    )


    # --------------------------------------------------------
    # COPIA O TRECHO REAL DO EXAME PARA A ROI
    # --------------------------------------------------------

    roi[
        destino_inicio[0]:destino_fim[0],
        destino_inicio[1]:destino_fim[1],
        destino_inicio[2]:destino_fim[2],
    ] = volume[
        origem_inicio[0]:origem_fim[0],
        origem_inicio[1]:origem_fim[1],
        origem_inicio[2]:origem_fim[2],
    ]


    # --------------------------------------------------------
    # NOVA POSIÇÃO DO CENTRO
    # --------------------------------------------------------
    #
    # O centro original estava nas coordenadas do exame.
    #
    # Agora precisamos saber onde ele ficou dentro da ROI.
    centro_roi = centro - inicio


    return roi, centro_roi

def reamostrar_roi_1mm(
    roi,
    centro_roi,
    espacamento
):
    """
    Reamostra a ROI para resolução isotrópica de 1 mm.

    Depois desta função:

        1 voxel = 1 mm

    O resultado terá aproximadamente:

        96 x 96 x 96 voxels
    """

    espacamento = np.asarray(
        espacamento,
        dtype=float
    )

    centro_roi = np.asarray(
        centro_roi,
        dtype=float
    )


    # --------------------------------------------------------
    # CRIA AS COORDENADAS DA NOVA ROI
    # --------------------------------------------------------
    #
    # Queremos exatamente 96 posições:
    #
    # 0, 1, 2, ..., 95
    #
    # Como cada posição representa 1 mm,
    # temos aproximadamente 96 mm.
    tamanho_saida = ROI_MM


    # Cria offsets em milímetros ao redor do centro.
    #
    # Exemplo simplificado:
    #
    # -47.5
    # -46.5
    # ...
    #  46.5
    #  47.5
    deslocamentos_mm = (
        np.arange(
            tamanho_saida,
            dtype=np.float32
        )
        - (tamanho_saida - 1) / 2
    )


    # --------------------------------------------------------
    # CONVERTE OS DESLOCAMENTOS DE MM PARA VOXELS
    # --------------------------------------------------------
    #
    # Exemplo:
    #
    # se spacing = 0.7 mm:
    #
    # 1 mm corresponde a:
    #
    # 1 / 0.7 = 1.42 voxels
    eixo_0 = (
        centro_roi[0]
        + deslocamentos_mm / espacamento[0]
    )

    eixo_1 = (
        centro_roi[1]
        + deslocamentos_mm / espacamento[1]
    )

    eixo_2 = (
        centro_roi[2]
        + deslocamentos_mm / espacamento[2]
    )


    # Cria a grade 3D com todas as coordenadas.
    coords = np.meshgrid(
        eixo_0,
        eixo_1,
        eixo_2,
        indexing="ij"
    )


    # --------------------------------------------------------
    # INTERPOLAÇÃO TRILINEAR
    # --------------------------------------------------------
    #
    # map_coordinates consulta o volume original em
    # coordenadas que podem ser decimais.
    #
    # order=1 significa interpolação linear.
    #
    # Como estamos trabalhando em 3 dimensões,
    # isso corresponde à interpolação trilinear.
    roi_1mm = map_coordinates(
        roi,
        coords,

        # Interpolação linear/trilinear.
        order=1,

        # Se tentar acessar algo fora do volume,
        # usa um valor constante.
        mode="constant",

        # Esse valor constante será -1000 HU.
        cval=HU_MIN
    )


    return roi_1mm.astype(
        np.float32
    )
    
def recortar_cubo_64mm(
    roi_1mm
):
    """
    Retira o cubo central de 64 mm.

    Como a ROI já está em 1 mm isotrópico:

        64 voxels = 64 mm
    """

    # Temos 96 voxels e queremos 64.
    #
    # 96 - 64 = 32
    #
    # Tiramos 16 de cada lado.
    inicio = (
        ROI_MM - PATCH_MM
    ) // 2

    fim = inicio + PATCH_MM


    # Recorta:
    #
    # 16:80
    #
    # nos três eixos.
    cubo = roi_1mm[
        inicio:fim,
        inicio:fim,
        inicio:fim
    ]


    return cubo

def janelar_hu(
    volume
):
    """
    Aplica a janela de HU:

        -1000 HU -> 0
         400 HU  -> 1

    Tudo abaixo de -1000 vira 0.
    Tudo acima de 400 vira 1.
    """

    # Limita os valores entre -1000 e 400.
    volume = np.clip(
        volume,
        HU_MIN,
        HU_MAX
    )


    # --------------------------------------------------------
    # NORMALIZA PARA O INTERVALO 0 A 1
    # --------------------------------------------------------
    #
    # Fórmula:
    #
    # valor_normalizado =
    #
    #     valor - minimo
    #     --------------
    #     maximo - minimo
    #
    volume = (
        (volume - HU_MIN)
        /
        (HU_MAX - HU_MIN)
    )


    return volume.astype(
        np.float32
    )

def criar_patch_2_5d(
    cubo
):
    """
    Cria um patch 2.5D usando três planos:

        canal 0 = axial
        canal 1 = coronal
        canal 2 = sagital

    O resultado terá formato:

        (3, 64, 64)
    """

    # Como o cubo possui 64 posições:
    #
    # 0 até 63
    #
    # escolhemos aproximadamente o centro.
    centro = PATCH_MM // 2


    # --------------------------------------------------------
    # PLANO AXIAL
    # --------------------------------------------------------
    #
    # Mantemos X e Y.
    # Fixamos Z.
    #
    # Resultado:
    #
    # 64 x 64
    axial = cubo[
        :,
        :,
        centro
    ]


    # --------------------------------------------------------
    # PLANO CORONAL
    # --------------------------------------------------------
    #
    # Mantemos X e Z.
    # Fixamos Y.
    coronal = cubo[
        :,
        centro,
        :
    ]


    # --------------------------------------------------------
    # PLANO SAGITAL
    # --------------------------------------------------------
    #
    # Mantemos Y e Z.
    # Fixamos X.
    sagital = cubo[
        centro,
        :,
        :
    ]


    # Empilha as três imagens.
    #
    # Antes:
    #
    # axial    -> (64, 64)
    # coronal  -> (64, 64)
    # sagital  -> (64, 64)
    #
    # Depois:
    #
    # patch -> (3, 64, 64)
    patch = np.stack(
        [
            axial,
            coronal,
            sagital
        ],
        axis=0
    )


    # A Sprint pede float16.
    return patch.astype(
        np.float16
    )
    
def extrair_patch_2_5d(
    volume,
    centro,
    espacamento
):
    """
    Executa todo o pipeline de transformação de um nódulo.

    Ordem obrigatória:

    volume original
        ↓
    ROI de 96 mm
        ↓
    reamostragem 1 mm
        ↓
    cubo de 64 mm
        ↓
    janela HU
        ↓
    axial + coronal + sagital
    """

    # --------------------------------------------------------
    # 1. RECORTA 96 MM NO ESPAÇAMENTO ORIGINAL
    # --------------------------------------------------------
    roi, centro_roi = recortar_roi_mm(
        volume,
        centro,
        espacamento
    )


    # --------------------------------------------------------
    # 2. REAMOSTRA SOMENTE A ROI
    # --------------------------------------------------------
    roi_1mm = reamostrar_roi_1mm(
        roi,
        centro_roi,
        espacamento
    )


    # --------------------------------------------------------
    # 3. RECORTA OS 64 MM CENTRAIS
    # --------------------------------------------------------
    cubo = recortar_cubo_64mm(
        roi_1mm
    )


    # --------------------------------------------------------
    # 4. APLICA A JANELA DE HU
    # --------------------------------------------------------
    cubo = janelar_hu(
        cubo
    )


    # --------------------------------------------------------
    # 5. GERA OS 3 CANAIS 2.5D
    # --------------------------------------------------------
    patch = criar_patch_2_5d(
        cubo
    )


    return patch

def teste_sintetico():
    """
    Cria um volume 3D artificial para testar
    todo o pipeline sem precisar dos DICOM reais.
    """

    print(
        "Iniciando teste sintético..."
    )


    # --------------------------------------------------------
    # ESPAÇAMENTO FALSO
    # --------------------------------------------------------
    #
    # Simula um exame onde:
    #
    # X = 0.7 mm
    # Y = 0.7 mm
    # Z = 2.0 mm
    #
    # Isso é propositalmente anisotrópico.
    espacamento = np.array(
        [
            0.7,
            0.7,
            2.0
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # TAMANHO DO VOLUME FALSO
    # --------------------------------------------------------
    shape = (
        180,
        180,
        80
    )


    # --------------------------------------------------------
    # CRIA O "PULMÃO"
    # --------------------------------------------------------
    #
    # Preenche tudo com -700 HU.
    #
    # É apenas um valor artificial parecido com
    # uma região contendo ar.
    volume = np.full(
        shape,
        -700,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # DEFINE O CENTRO DO NÓDULO
    # --------------------------------------------------------
    centro = np.array(
        [
            90.0,
            90.0,
            40.0
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # CRIA UMA ESFERA ARTIFICIAL
    # --------------------------------------------------------
    #
    # np.indices cria as coordenadas de todos os voxels.
    i, j, k = np.indices(
        shape
    )


    # Calcula a distância física de cada voxel
    # até o centro.
    #
    # Perceba que multiplicamos pelo spacing.
    #
    # Assim a distância é calculada em MILÍMETROS,
    # e não em quantidade de pixels.
    distancia_mm = np.sqrt(
        (
            (i - centro[0])
            * espacamento[0]
        ) ** 2
        +
        (
            (j - centro[1])
            * espacamento[1]
        ) ** 2
        +
        (
            (k - centro[2])
            * espacamento[2]
        ) ** 2
    )


    # --------------------------------------------------------
    # CRIA O NÓDULO
    # --------------------------------------------------------
    #
    # Todos os voxels que estiverem a até
    # 6 mm do centro recebem 100 HU.
    #
    # Portanto criamos uma esfera de aproximadamente
    # 12 mm de diâmetro.
    volume[
        distancia_mm <= 6
    ] = 100


    # --------------------------------------------------------
    # EXECUTA NOSSO PIPELINE
    # --------------------------------------------------------
    patch = extrair_patch_2_5d(
        volume,
        centro,
        espacamento
    )


    # --------------------------------------------------------
    # TESTES AUTOMÁTICOS
    # --------------------------------------------------------

    # O formato obrigatoriamente deve ser:
    #
    # 3 canais
    # 64 pixels
    # 64 pixels
    assert patch.shape == (
        3,
        64,
        64
    )


    # A Sprint pede float16.
    assert patch.dtype == np.float16


    # Depois da janela HU,
    # nenhum valor pode ser menor que 0.
    assert patch.min() >= 0


    # E nenhum pode ultrapassar 1.
    assert patch.max() <= 1


    print(
        "Teste sintético OK!"
    )

    print(
        "Shape:",
        patch.shape
    )

    print(
        "Tipo:",
        patch.dtype
    )

    print(
        "Valor mínimo:",
        patch.min()
    )

    print(
        "Valor máximo:",
        patch.max()
    )
    
if __name__ == "__main__":
    # Por enquanto executamos somente o teste artificial.
    #
    # Assim conseguimos desenvolver toda a lógica
    # sem precisar baixar o LIDC-IDRI.
    teste_sintetico()