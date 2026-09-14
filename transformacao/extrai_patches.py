from pathlib import Path
import configparser
import numpy as np
import pandas as pd

from scipy.ndimage import map_coordinates


# ============================================================
# COMPATIBILIDADE COM PYTHON / NUMPY RECENTES
# ============================================================

# O pylidc usa nomes antigos do NumPy.
if not hasattr(np, "int"):
    np.int = int

if not hasattr(np, "float"):
    np.float = float

if not hasattr(np, "bool"):
    np.bool = bool


# O pylidc usa SafeConfigParser(),
# que foi removido no Python 3.12.
#
# ConfigParser é o substituto atual equivalente.
if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser


# IMPORTANTE:
# pylidc deve ser importado DEPOIS das correções acima.
import pylidc as pl

from pylidc.utils import consensus
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

def obter_scan(scan_id):
    """
    Busca um exame no banco do pylidc pelo ID.

    O scan_id armazenado no nosso CSV corresponde
    ao campo Scan.id do pylidc.
    """

    # Faz uma consulta no banco SQLite interno do pylidc.
    scan = (
        pl.query(pl.Scan)
        .filter(pl.Scan.id == int(scan_id))
        .first()
    )


    # Se nada for encontrado, alguma coisa está errada
    # com o scan_id ou com o banco local do pylidc.
    if scan is None:
        raise ValueError(
            f"Scan id={scan_id} não encontrado no pylidc."
        )


    return scan

def obter_cluster(scan, cluster_idx):
    """
    Recupera exatamente o cluster correspondente
    ao nódulo selecionado anteriormente.
    """

    # Reexecutamos exatamente o mesmo agrupamento usado
    # durante a etapa de seleção dos nódulos.
    clusters = scan.cluster_annotations(
        verbose=False
    )


    # Converte para inteiro porque o valor veio do CSV.
    cluster_idx = int(
        cluster_idx
    )


    # Verificação de segurança.
    #
    # Exemplo:
    #
    # se existem 4 clusters:
    #
    # índices válidos = 0, 1, 2, 3
    if (
        cluster_idx < 0
        or cluster_idx >= len(clusters)
    ):
        raise IndexError(
            f"cluster_idx={cluster_idx} inválido. "
            f"O scan possui {len(clusters)} clusters."
        )


    return clusters[
        cluster_idx
    ]

def calcular_centroide_consenso(
    anns
):
    """
    Calcula o centro do nódulo usando o consenso
    de pelo menos 50% das anotações.

    anns é uma lista de Annotation do pylidc
    referentes ao mesmo nódulo.
    """


    # ========================================================
    # CONSENSO ENTRE OS RADIOLOGISTAS
    # ========================================================
    #
    # clevel=0.5 significa:
    #
    # um voxel pertence ao nódulo quando pelo menos
    # 50% das segmentações dos radiologistas incluem
    # aquele voxel.
    #
    # ret_masks=False:
    # não precisamos guardar as máscaras individuais
    # dos radiologistas.
    #
    # Recebemos:
    #
    # cmask = máscara 3D de consenso
    # cbbox = posição dessa máscara dentro do exame
    cmask, cbbox = consensus(
        anns,
        clevel=0.5,
        ret_masks=False,
    )


    # ========================================================
    # COORDENADAS DOS VOXELS DO NÓDULO
    # ========================================================
    #
    # cmask é uma matriz booleana:
    #
    # False = não pertence ao nódulo
    # True  = pertence ao nódulo
    #
    # np.argwhere retorna todas as coordenadas True.
    coordenadas = np.argwhere(
        cmask
    )


    # É extremamente improvável que isso aconteça,
    # mas fazemos a validação por segurança.
    if coordenadas.size == 0:
        raise ValueError(
            "A máscara de consenso ficou vazia."
        )


    # ========================================================
    # CENTROIDE LOCAL
    # ========================================================
    #
    # Calculamos a média das coordenadas de todos
    # os voxels pertencentes ao consenso.
    #
    # Exemplo:
    #
    # [12.3, 14.7, 3.8]
    #
    # Porém essas coordenadas ainda são relativas
    # à pequena bounding box do consenso.
    centro_local = coordenadas.mean(
        axis=0
    )


    # ========================================================
    # OFFSET DA BOUNDING BOX
    # ========================================================
    #
    # O consensus() não cria uma máscara do tamanho
    # do tórax inteiro.
    #
    # Ele cria apenas uma pequena região contendo
    # o nódulo.
    #
    # cbbox informa onde essa região fica no exame.
    #
    # Exemplo:
    #
    # cbbox[0] = slice(180, 210)
    # cbbox[1] = slice(240, 270)
    # cbbox[2] = slice(50, 60)
    #
    # Portanto precisamos somar:
    #
    # [180, 240, 50]
    offset = np.array(
        [
            cbbox[0].start,
            cbbox[1].start,
            cbbox[2].start
        ],
        dtype=np.float32
    )


    # ========================================================
    # CENTROIDE NO VOLUME COMPLETO
    # ========================================================
    #
    # Agora convertemos:
    #
    # coordenada dentro da máscara
    #
    # para:
    #
    # coordenada dentro do exame inteiro.
    centro_volume = (
        centro_local
        + offset
    )


    return centro_volume

def extrair_nodulo_real(
    linha
):
    """
    Recebe uma linha do nodulos_com_split.csv
    e gera o patch 2.5D correspondente.
    """


    print(
        f"Processando {linha.nodule_id}..."
    )


    # ========================================================
    # 1. LOCALIZA O EXAME
    # ========================================================

    scan = obter_scan(
        linha.scan_id
    )


    # Verificação adicional:
    #
    # o paciente encontrado no pylidc deve ser
    # exatamente o mesmo paciente informado no CSV.
    if scan.patient_id != linha.patient_id:
        raise ValueError(
            "Paciente do CSV não corresponde ao Scan. "
            f"CSV={linha.patient_id}, "
            f"pylidc={scan.patient_id}"
        )


    print(
        "Paciente:",
        scan.patient_id
    )

    print(
        "Scan:",
        scan.id
    )


    # ========================================================
    # 2. LOCALIZA O NÓDULO
    # ========================================================

    anns = obter_cluster(
        scan,
        linha.cluster_idx
    )


    print(
        "Cluster:",
        linha.cluster_idx
    )

    print(
        "Anotações:",
        len(anns)
    )


    # ========================================================
    # 3. CONFERE A QUANTIDADE DE RADIOLOGISTAS
    # ========================================================
    #
    # Essa é uma verificação muito importante.
    #
    # Se o CSV dizia que o nódulo possuía 4 anotações,
    # esperamos encontrar exatamente 4 novamente.
    if (
        len(anns)
        != int(linha.n_radiologistas)
    ):
        raise ValueError(
            "Quantidade de anotações diferente "
            "da registrada no CSV. "
            f"CSV={linha.n_radiologistas}, "
            f"agora={len(anns)}"
        )


    # ========================================================
    # 4. CONFERE OS ESCORES DE MALIGNIDADE
    # ========================================================
    #
    # Isso serve como uma segunda proteção contra
    # selecionar o cluster errado.
    escores_atuais = sorted(
        int(ann.malignancy)
        for ann in anns
    )


    # No CSV temos algo como:
    #
    # "5,5,5,4"
    escores_csv = sorted(
        int(valor)
        for valor
        in str(
            linha.escores_individuais
        ).split(",")
    )


    if (
        escores_atuais
        != escores_csv
    ):
        raise ValueError(
            "Os escores do cluster não correspondem "
            "aos escores salvos no CSV. "
            f"CSV={escores_csv}, "
            f"pylidc={escores_atuais}"
        )


    print(
        "Cluster confirmado."
    )


    # ========================================================
    # 5. CARREGA O VOLUME
    # ========================================================
    #
    # Aqui os DICOMs finalmente são acessados.
    #
    # ATENÇÃO:
    #
    # não fazemos RescaleSlope / RescaleIntercept
    # novamente.
    #
    # Seguimos a regra definida para a Sprint.
    volume = scan.to_volume(
        verbose=False
    ).astype(
        np.float32
    )


    print(
        "Volume:",
        volume.shape
    )


    # ========================================================
    # 6. ESPAÇAMENTO FÍSICO
    # ========================================================
    #
    # O pylidc usa um único pixel_spacing porque
    # no LIDC os dois eixos do plano transversal
    # possuem a mesma resolução.
    #
    # No eixo Z usamos slice_spacing.
    espacamento = np.array(
        [
            scan.pixel_spacing,
            scan.pixel_spacing,
            scan.slice_spacing
        ],
        dtype=np.float32
    )


    print(
        "Espaçamento:",
        espacamento
    )


    # ========================================================
    # 7. CONSENSO + CENTROIDE
    # ========================================================

    centro = calcular_centroide_consenso(
        anns
    )


    print(
        "Centroide:",
        centro
    )


    # ========================================================
    # 8. EXECUTA O PIPELINE QUE JÁ TESTAMOS
    # ========================================================
    #
    # Aqui reaproveitamos exatamente a função
    # validada com o dado sintético.
    patch = extrair_patch_2_5d(
        volume,
        centro,
        espacamento
    )


    return patch

def salvar_teste_patch_real(patch, linha):
    """
    Salva temporariamente um único patch real.

    Serve apenas para validar o formato do .npz e do manifesto
    antes de processarmos o dataset completo.
    """

    # --------------------------------------------------------
    # CRIA A PASTA DE PATCHES
    # --------------------------------------------------------

    pasta_patches = Path("patches")

    pasta_patches.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # ADICIONA A DIMENSÃO DE AMOSTRAS
    # --------------------------------------------------------
    #
    # Atualmente:
    #
    # patch.shape = (3, 64, 64)
    #
    # Para salvar vários nódulos, queremos:
    #
    # (N, 3, 64, 64)
    #
    # Como temos apenas um:
    #
    # (1, 3, 64, 64)
    patches = np.expand_dims(
        patch,
        axis=0
    ).astype(
        np.float16
    )


    # --------------------------------------------------------
    # SALVA O NPZ
    # --------------------------------------------------------

    caminho_npz = (
        pasta_patches
        / "teste_patch_2_5d.npz"
    )

    np.savez_compressed(
        caminho_npz,
        patches=patches,
        nodule_id=np.array(
            [linha.nodule_id]
        )
    )


    # --------------------------------------------------------
    # CRIA O MANIFESTO
    # --------------------------------------------------------

    manifesto = pd.DataFrame(
        [
            {
                "nodule_id": linha.nodule_id,
                "patient_id": linha.patient_id,
                "split": linha.split,
                "label": linha.rotulo_binario
            }
        ]
    )


    caminho_manifesto = Path(
        "transformacao/"
        "teste_manifest_patches.csv"
    )


    manifesto.to_csv(
        caminho_manifesto,
        index=False
    )


    print()
    print("Arquivos de teste salvos!")
    print("NPZ:", caminho_npz)
    print(
        "Shape salvo:",
        patches.shape
    )
    print(
        "Manifesto:",
        caminho_manifesto
    )

def teste_nodulo_real():
    """
    Testa a extração usando somente o primeiro
    nódulo do dataset.

    Isso evita processar os 751 de uma vez enquanto
    ainda estamos validando o pipeline.
    """


    print()
    print(
        "Iniciando teste com nódulo real..."
    )


    # Abre o CSV que contém os nódulos
    # e os respectivos splits.
    df = pd.read_csv(
        "selecao/nodulos_com_split.csv"
    )


    # ========================================================
    # CONFERE AS COLUNAS NECESSÁRIAS
    # ========================================================

    colunas_necessarias = {
        "nodule_id",
        "cluster_idx",
        "patient_id",
        "scan_id",
        "n_radiologistas",
        "escores_individuais",
        "rotulo_binario",
        "split"
    }


    faltantes = (
        colunas_necessarias
        - set(df.columns)
    )


    if faltantes:
        raise ValueError(
            "O CSV está sem as seguintes colunas: "
            f"{sorted(faltantes)}"
        )


    # ============================================================
    # ESCOLHE UM NÓDULO REAL ESPECÍFICO PARA O TESTE
    # ============================================================
    #
    # Estamos usando o LIDC-IDRI-0082 porque é um dos pacientes
    # disponíveis no conjunto de DICOM que temos acesso.

    nodule_id_teste = "LIDC-IDRI-0082_scan90_cluster000"


    # Procura exatamente esse nódulo no CSV.
    df_teste = df[
        df["nodule_id"] == nodule_id_teste
    ]


    # Se não encontrar, interrompe o programa.
    if df_teste.empty:
        raise ValueError(
            f"Nódulo {nodule_id_teste} não encontrado no CSV."
        )


    # Como nodule_id deve ser único, pegamos a única linha encontrada.
    linha = next(
        df_teste.itertuples(
            index=False
        )
    )


    print(
        "Nódulo:",
        linha.nodule_id
    )

    print(
        "Paciente:",
        linha.patient_id
    )

    print(
        "Rótulo:",
        linha.rotulo_binario
    )

    print(
        "Split:",
        linha.split
    )


    # Executa a extração.
    patch = extrair_nodulo_real(
        linha
    )


    # ========================================================
    # VERIFICAÇÕES
    # ========================================================

    assert patch.shape == (
        3,
        64,
        64
    )

    assert (
        patch.dtype
        == np.float16
    )

    assert (
        patch.min()
        >= 0
    )

    assert (
        patch.max()
        <= 1
    )


    print()
    print(
        "Teste real OK!"
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
        "Mínimo:",
        patch.min()
    )

    print(
        "Máximo:",
        patch.max()
    )
    # Testa também o salvamento do patch
    # e a criação do manifesto.
    salvar_teste_patch_real(
        patch,
        linha
)   

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

    # Sempre podemos executar o teste sintético.
    teste_sintetico()


    # Quando os arquivos DICOM estiverem disponíveis
    # e configurados no pylidc, descomente:
    #
    teste_nodulo_real()