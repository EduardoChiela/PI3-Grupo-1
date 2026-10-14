"""
Issue #17 — ResNet com transfer learning sobre patches 2.5D.

Arquitetura:
    ResNet18 pré-treinada no ImageNet.

Entrada:
    3 x 64 x 64

Os três canais NÃO são RGB:
    canal 0 = axial
    canal 1 = coronal
    canal 2 = sagital

Decisão sobre tamanho:
    Os patches permanecem em 64 x 64.
    Não são interpolados para 224 x 224 porque isso aumentaria
    o custo sem criar nova informação anatômica.

    A ResNet utiliza AdaptiveAvgPool2d antes da camada final,
    portanto aceita normalmente entradas 64 x 64.

Normalização:
    É aplicada a normalização do ImageNet porque são utilizados
    pesos pré-treinados do ImageNet.

    Isso é apenas uma compatibilização com os pesos pré-treinados.
    Os canais continuam representando planos anatômicos, não RGB.

Transfer learning:
    O backbone permanece congelado.
    Apenas a camada final de classificação é treinada.

Regra de avaliação:
    Arquitetura, épocas, threshold e demais decisões são tomadas
    usando somente treino e validação.

    O teste é utilizado uma única vez após essas decisões.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8"
)

os.environ.setdefault(
    "MPLBACKEND",
    "Agg"
)

import argparse
import logging
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from mineracao.dataset import (
    AugmentacaoFlip3D,
    carregar_particoes
)

from mineracao.avaliacao import (
    avaliar_com_threshold,
    bootstrap_por_paciente,
    calcular_metricas,
    escolher_threshold_validacao,
    gerar_curva_roc,
    validar_com_manifesto
)


SEED = 42

PESOS_REFERENCIA = {
    0: 0.966667,
    1: 1.035714
}

MEDIA_IMAGENET = (
    0.485,
    0.456,
    0.406
)

DESVIO_IMAGENET = (
    0.229,
    0.224,
    0.225
)

log = logging.getLogger(
    "resnet"
)


def fixar_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(
        True,
        warn_only=True
    )


class ResNet2_5D(nn.Module):

    def __init__(self):

        super().__init__()

        pesos = (
            ResNet18_Weights.IMAGENET1K_V1
        )

        self.rede = resnet18(
            weights=pesos
        )

        # ----------------------------------------------------
        # CONGELA TODO O EXTRATOR
        # ----------------------------------------------------

        for parametro in self.rede.parameters():
            parametro.requires_grad = False

        # ----------------------------------------------------
        # SUBSTITUI A CABEÇA ORIGINAL
        # ----------------------------------------------------
        #
        # ImageNet:
        # 512 features -> 1000 classes
        #
        # Nosso problema:
        # 512 features -> 1 logit
        #

        entrada_fc = (
            self.rede.fc.in_features
        )

        self.rede.fc = nn.Linear(
            entrada_fc,
            1
        )

        # ----------------------------------------------------
        # NORMALIZAÇÃO IMAGENET
        # ----------------------------------------------------

        media = torch.tensor(
            MEDIA_IMAGENET,
            dtype=torch.float32
        ).view(
            1,
            3,
            1,
            1
        )

        desvio = torch.tensor(
            DESVIO_IMAGENET,
            dtype=torch.float32
        ).view(
            1,
            3,
            1,
            1
        )

        self.register_buffer(
            "media",
            media
        )

        self.register_buffer(
            "desvio",
            desvio
        )

    def forward(self, x):

        # x originalmente está entre 0 e 1.
        x = (
            x - self.media
        ) / self.desvio

        return self.rede(x)


def manter_backbone_congelado(modelo):

    # Quando chamamos modelo.train(),
    # BatchNorm normalmente atualizaria suas estatísticas.
    #
    # Como estamos dizendo que o extrator está congelado,
    # também mantemos os BatchNorm em modo de avaliação.

    for modulo in modelo.rede.modules():

        if isinstance(
            modulo,
            nn.BatchNorm2d
        ):
            modulo.eval()


def pesos_de_classe(y_treino):

    n = len(y_treino)

    pesos = {
        classe: round(
            n
            /
            (
                2
                * int(
                    (
                        y_treino
                        == classe
                    ).sum()
                )
            ),
            6
        )
        for classe in (
            0,
            1
        )
    }

    for classe in (
        0,
        1
    ):

        if abs(
            pesos[classe]
            - PESOS_REFERENCIA[classe]
        ) > 1e-6:

            raise ValueError(
                f"Pesos calculados {pesos} "
                f"diferem da referência "
                f"{PESOS_REFERENCIA}."
            )

    return pesos


def prever(
    modelo,
    dataset,
    dispositivo,
    batch=128
):

    modelo.eval()

    saidas = []

    with torch.no_grad():

        for inicio in range(
            0,
            len(dataset),
            batch
        ):

            fim = min(
                inicio + batch,
                len(dataset)
            )

            x = np.stack(
                [
                    dataset[i][0]
                    for i in range(
                        inicio,
                        fim
                    )
                ]
            )

            x = torch.from_numpy(
                x
            ).to(
                dispositivo
            )

            logits = modelo(
                x
            ).squeeze(1)

            probabilidades = torch.sigmoid(
                logits
            )

            saidas.append(
                probabilidades
                .cpu()
                .numpy()
            )

    return np.concatenate(
        saidas
    ).astype(
        float
    )


def tabela_predicoes(
    dataset,
    split,
    probabilidades
):

    return pd.DataFrame(
        {
            "nodule_id":
                dataset.nodule_ids,

            "patient_id":
                dataset.patient_ids,

            "split":
                split,

            "y_true":
                dataset.y.astype(int),

            "y_prob":
                probabilidades
        }
    )


def perda_validacao(
    y,
    probabilidades,
    pesos
):

    probabilidades = np.clip(
        probabilidades,
        1e-7,
        1 - 1e-7
    )

    w = np.where(
        y == 1,
        pesos[1],
        pesos[0]
    )

    perda = (
        -w
        * (
            y
            * np.log(probabilidades)
            +
            (
                1 - y
            )
            * np.log(
                1 - probabilidades
            )
        )
    )

    return float(
        np.mean(perda)
    )


def main():

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]
    )

    parser.add_argument(
        "--npz",
        type=Path,
        default=Path(
            "patches/patches_2_5d.npz"
        )
    )

    parser.add_argument(
        "--manifesto",
        type=Path,
        default=Path(
            "transformacao/"
            "manifest_patches.csv"
        )
    )

    parser.add_argument(
        "--saida",
        type=Path,
        default=Path(
            "mineracao/"
            "resultados/"
            "resnet"
        )
    )

    parser.add_argument(
        "--epocas",
        type=int,
        default=30
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    parser.add_argument(
        "--sem-aug",
        action="store_true"
    )

    parser.add_argument(
        "--sem-pesos",
        action="store_true"
    )

    parser.add_argument(
        "--avaliar-teste",
        action="store_true",
        help=(
            "Avalia o teste uma única vez, "
            "depois das decisões serem "
            "congeladas."
        )
    )

    args = parser.parse_args()

    args.saida.mkdir(
        parents=True,
        exist_ok=True
    )

    logging.basicConfig(
        level=logging.INFO,

        format=(
            "%(asctime)s "
            "%(message)s"
        ),

        handlers=[
            logging.StreamHandler(),

            logging.FileHandler(
                args.saida
                / "log_treino.txt",
                "w"
            )
        ]
    )

    fixar_seed(
        args.seed
    )

    dispositivo = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Augmentation idêntica à baseline.
    augmentacao = (
        None
        if args.sem_aug
        else AugmentacaoFlip3D(
            args.seed
        )
    )

    particoes, sha = carregar_particoes(
        args.npz,
        args.manifesto,
        transform_treino=augmentacao
    )

    treino = particoes[
        "treino"
    ]

    validacao = particoes[
        "validacao"
    ]

    teste = particoes[
        "teste"
    ]

    pesos_calculados = pesos_de_classe(
        treino.y
    )

    pesos = (
        {
            0: 1.0,
            1: 1.0
        }
        if args.sem_pesos
        else pesos_calculados
    )

    modelo = ResNet2_5D().to(
        dispositivo
    )

    parametros_treinaveis = [
        parametro
        for parametro
        in modelo.parameters()
        if parametro.requires_grad
    ]

    total_parametros = sum(
        parametro.numel()
        for parametro
        in modelo.parameters()
    )

    total_treinaveis = sum(
        parametro.numel()
        for parametro
        in parametros_treinaveis
    )

    log.info(
        "python=%s torch=%s torchvision=%s dispositivo=%s",
        platform.python_version(),
        torch.__version__,
        torchvision.__version__,
        dispositivo
    )

    log.info(
        "seed=%d epocas=%d lr=%g batch=%d",
        args.seed,
        args.epocas,
        args.lr,
        args.batch
    )

    log.info(
        "arquitetura=ResNet18 "
        "pesos=ImageNet1K_V1"
    )

    log.info(
        "entrada=3x64x64 "
        "sem resize para 224"
    )

    log.info(
        "canais=axial,coronal,sagital "
        "(nao RGB)"
    )

    log.info(
        "normalizacao=ImageNet "
        "mean=%s std=%s",
        MEDIA_IMAGENET,
        DESVIO_IMAGENET
    )

    log.info(
        "backbone=congelado "
        "cabeca_fc=treinavel"
    )

    log.info(
        "parametros_total=%d "
        "parametros_treinaveis=%d",
        total_parametros,
        total_treinaveis
    )

    log.info(
        "augmentation=%s",
        (
            "flip3d_so_treino"
            if augmentacao
            else "nenhuma"
        )
    )

    log.info(
        "pesos_de_classe=%s",
        (
            pesos
            if not args.sem_pesos
            else "NAO (1 e 1)"
        )
    )

    log.info(
        "npz=%s",
        args.npz
    )

    log.info(
        "npz_sha256=%s",
        sha
    )

    log.info(
        "manifesto=%s",
        args.manifesto
    )

    log.info(
        "n treino=%d validacao=%d teste=%d",
        len(treino),
        len(validacao),
        len(teste)
    )

    otimizador = torch.optim.Adam(
        parametros_treinaveis,
        lr=args.lr
    )

    gerador = torch.Generator().manual_seed(
        args.seed
    )

    loader = DataLoader(
        treino,
        batch_size=args.batch,
        shuffle=True,
        generator=gerador,
        num_workers=0
    )

    pesos_tensor = torch.tensor(
        [
            pesos[0],
            pesos[1]
        ],
        dtype=torch.float32,
        device=dispositivo
    )

    melhor_perda = float(
        "inf"
    )

    melhor_estado = None

    melhor_epoca = -1

    for epoca in range(
        1,
        args.epocas + 1
    ):

        modelo.train()

        manter_backbone_congelado(
            modelo
        )

        soma_perda = 0.0

        quantidade = 0

        for x, y in loader:

            x = x.to(
                dispositivo
            )

            y = y.to(
                dispositivo
            )

            logits = modelo(
                x
            ).squeeze(1)

            perdas = (
                F.binary_cross_entropy_with_logits(
                    logits,
                    y,
                    reduction="none"
                )
            )

            perda = (
                perdas
                * pesos_tensor[
                    y.long()
                ]
            ).mean()

            otimizador.zero_grad()

            perda.backward()

            otimizador.step()

            soma_perda += (
                perda.item()
                * len(y)
            )

            quantidade += len(y)

        probabilidades_val = prever(
            modelo,
            validacao,
            dispositivo
        )

        perda_val = perda_validacao(
            validacao.y,
            probabilidades_val,
            pesos
        )

        auc_val = calcular_metricas(
            validacao.y,
            probabilidades_val
        )["auc"]

        log.info(
            "epoca=%02d "
            "perda_treino=%.4f "
            "perda_val=%.4f "
            "auc_val=%.4f",
            epoca,
            soma_perda / quantidade,
            perda_val,
            auc_val
        )

        if perda_val < melhor_perda:

            melhor_perda = perda_val

            melhor_epoca = epoca

            melhor_estado = {
                chave:
                    valor.detach()
                    .cpu()
                    .clone()

                for chave, valor
                in modelo.state_dict().items()
            }

    modelo.load_state_dict(
        melhor_estado
    )

    log.info(
        "melhor_epoca_validacao=%d",
        melhor_epoca
    )

    probabilidades_val = prever(
        modelo,
        validacao,
        dispositivo
    )

    df_val = tabela_predicoes(
        validacao,
        "validacao",
        probabilidades_val
    )

    manifesto = pd.read_csv(
        args.manifesto
    )

    df_val = validar_com_manifesto(
        df_val,
        manifesto
    )

    escolha = escolher_threshold_validacao(
        df_val
    )

    threshold = escolha[
        "threshold"
    ]

    log.info(
        "threshold_validacao=%s",
        escolha
    )

    metricas_val = avaliar_com_threshold(
        df_val,
        threshold
    )

    log.info(
        "metricas_validacao=%s",
        metricas_val
    )

    df_val.to_csv(
        args.saida
        / "predicoes_validacao.csv",
        index=False
    )

    gerar_curva_roc(
        df_val,
        args.saida
        / "roc_validacao.png",
        threshold=threshold
    )

    torch.save(
        modelo.state_dict(),
        args.saida
        / "modelo_resnet.pt"
    )

    # --------------------------------------------------------
    # TESTE FINAL
    # --------------------------------------------------------
    #
    # Só execute quando arquitetura, épocas, pesos,
    # normalização e threshold estiverem congelados.
    #

    if args.avaliar_teste:

        log.warning(
            "AVALIACAO FINAL DO TESTE INICIADA"
        )

        probabilidades_teste = prever(
            modelo,
            teste,
            dispositivo
        )

        df_teste = tabela_predicoes(
            teste,
            "teste",
            probabilidades_teste
        )

        df_teste = validar_com_manifesto(
            df_teste,
            manifesto
        )

        df_teste.to_csv(
            args.saida
            / "predicoes_teste.csv",
            index=False
        )

        gerar_curva_roc(
            df_teste,
            args.saida
            / "roc_teste.png",
            threshold=threshold
        )

        resultado_teste = (
            avaliar_com_threshold(
                df_teste,
                threshold
            )
        )

        log.info(
            "TESTE=%s",
            resultado_teste
        )

        intervalos = bootstrap_por_paciente(
            df_teste,
            threshold,
            n_repeticoes=2000,
            seed=args.seed
        )

        log.info(
            "IC95_bootstrap_por_paciente=%s",
            intervalos
        )


if __name__ == "__main__":
    main()