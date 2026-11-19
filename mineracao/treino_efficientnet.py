"""Issue #44 — EfficientNet-B0 com transfer learning em patches 2.5D.

Na raiz do repositório, execute:

    python -m mineracao.treino_efficientnet --configuracao congelado --saida mineracao/resultados/efficientnet_s6/congelado
    python -m mineracao.treino_efficientnet --configuracao ultimo_bloco --saida mineracao/resultados/efficientnet_s6/ultimo_bloco
    python -m mineracao.treino_efficientnet --configuracao dois_ultimos_blocos --saida mineracao/resultados/efficientnet_s6/dois_ultimos_blocos
    python -m mineracao.resumo_efficientnet

Confira `escolha_validacao.json` antes de executar a avaliação final única:

    python -m mineracao.avaliar_efficientnet --confirmar-teste-final

Este módulo não avalia o conjunto de teste. Os três comandos de treino usam
pastas fixas porque o resumo lê esses caminhos.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")

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
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from mineracao.dataset import AugmentacaoFlip3D, carregar_particoes
from mineracao.avaliacao import (
    avaliar_com_threshold,
    calcular_metricas,
    escolher_threshold_validacao,
    gerar_curva_roc,
    validar_com_manifesto,
)


SEED = 42
PESOS_REFERENCIA = {0: 0.966667, 1: 1.035714}
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)
CONFIGURACOES = ("congelado", "ultimo_bloco", "dois_ultimos_blocos")
log = logging.getLogger("efficientnet")


def fixar_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class EfficientNetB0_2_5D(nn.Module):
    """EfficientNet-B0 ImageNet; entrada 3x64x64 dos planos 2.5D."""

    def __init__(self, configuracao: str):
        super().__init__()
        if configuracao not in CONFIGURACOES:
            raise ValueError(f"Configuração inválida: {configuracao}")
        self.rede = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for parametro in self.rede.parameters():
            parametro.requires_grad = False
        entrada = self.rede.classifier[-1].in_features
        self.rede.classifier[-1] = nn.Linear(entrada, 1)
        # features[8] é a projeção convolucional comum após os blocos MBConv.
        # Ela permanece congelada nas três configurações deste experimento.
        if configuracao == "ultimo_bloco":
            blocos = (7,)
        elif configuracao == "dois_ultimos_blocos":
            blocos = (6, 7)
        else:
            blocos = ()
        for indice in blocos:
            for parametro in self.rede.features[indice].parameters():
                parametro.requires_grad = True
        self.register_buffer("media", torch.tensor(MEDIA_IMAGENET).view(1, 3, 1, 1))
        self.register_buffer("desvio", torch.tensor(DESVIO_IMAGENET).view(1, 3, 1, 1))

    def forward(self, x):
        return self.rede((x - self.media) / self.desvio)


def manter_batchnorm_em_avaliacao(modelo) -> None:
    for modulo in modelo.rede.modules():
        if isinstance(modulo, nn.BatchNorm2d):
            modulo.eval()


def pesos_de_classe(y_treino):
    n = len(y_treino)
    pesos = {
        classe: round(n / (2 * int((y_treino == classe).sum())), 6)
        for classe in (0, 1)
    }
    if any(abs(pesos[c] - PESOS_REFERENCIA[c]) > 1e-6 for c in (0, 1)):
        raise ValueError(f"Pesos calculados {pesos} diferem da referência {PESOS_REFERENCIA}.")
    return pesos


def prever(modelo, dataset, dispositivo, batch=128):
    modelo.eval()
    saidas = []
    with torch.no_grad():
        for inicio in range(0, len(dataset), batch):
            fim = min(inicio + batch, len(dataset))
            x = np.stack([dataset[i][0] for i in range(inicio, fim)])
            logits = modelo(torch.from_numpy(x).to(dispositivo)).squeeze(1)
            saidas.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(saidas).astype(float)


def tabela_predicoes(dataset, split, probabilidades):
    return pd.DataFrame({
        "nodule_id": dataset.nodule_ids,
        "patient_id": dataset.patient_ids,
        "split": split,
        "y_true": dataset.y.astype(int),
        "y_prob": probabilidades,
    })


def perda_validacao(y, probabilidades, pesos):
    probabilidades = np.clip(probabilidades, 1e-7, 1 - 1e-7)
    w = np.where(y == 1, pesos[1], pesos[0])
    perda = -w * (y * np.log(probabilidades) + (1 - y) * np.log(1 - probabilidades))
    return float(np.mean(perda))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=Path("patches/patches_2_5d.npz"))
    parser.add_argument("--manifesto", type=Path, default=Path("transformacao/manifest_patches.csv"))
    parser.add_argument("--saida", type=Path, required=True, help="Diretório exclusivo da configuração.")
    parser.add_argument("--configuracao", choices=CONFIGURACOES, required=True)
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.saida.exists() and any(args.saida.iterdir()):
        raise FileExistsError(f"A saída já contém arquivos; use outro diretório para preservar o log: {args.saida}")
    args.saida.mkdir(parents=True, exist_ok=True)
    log_path = args.saida / "log_treino.txt"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(log_path, "w", encoding="utf-8")
    ])
    fixar_seed(args.seed)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augmentacao = AugmentacaoFlip3D(args.seed)
    particoes, sha = carregar_particoes(args.npz, args.manifesto, transform_treino=augmentacao)
    treino, validacao = particoes["treino"], particoes["validacao"]
    pesos = pesos_de_classe(treino.y)
    modelo = EfficientNetB0_2_5D(args.configuracao).to(dispositivo)
    parametros_treinaveis = [p for p in modelo.parameters() if p.requires_grad]
    log.info("python=%s torch=%s torchvision=%s dispositivo=%s", platform.python_version(), torch.__version__, torchvision.__version__, dispositivo)
    log.info("seed=%d epocas=%d lr=%g batch=%d", args.seed, args.epocas, args.lr, args.batch)
    log.info("arquitetura=EfficientNet_B0 pesos=ImageNet1K_V1")
    log.info("configuracao_finetuning=%s", args.configuracao)
    log.info("lr_cabeca=%g lr_backbone=%g", args.lr, args.lr_backbone)
    log.info("entrada=3x64x64 sem resize; canais=axial,coronal,sagital (nao RGB)")
    log.info("normalizacao=ImageNet mean=%s std=%s", MEDIA_IMAGENET, DESVIO_IMAGENET)
    log.info("blocos_liberados=%s", {"congelado": [], "ultimo_bloco": [7], "dois_ultimos_blocos": [6, 7]}[args.configuracao])
    log.info("parametros_total=%d parametros_treinaveis=%d", sum(p.numel() for p in modelo.parameters()), sum(p.numel() for p in parametros_treinaveis))
    log.info("augmentation=flip3d_so_treino pesos_de_classe=%s", pesos)
    log.info("npz=%s npz_sha256=%s manifesto=%s", args.npz, sha, args.manifesto)
    log.info("n treino=%d validacao=%d teste=%d", len(treino), len(validacao), len(particoes["teste"]))

    cabeca = list(modelo.rede.classifier.parameters())
    backbone = [p for nome, p in modelo.rede.named_parameters() if p.requires_grad and not nome.startswith("classifier.")]
    grupos = [{"params": cabeca, "lr": args.lr}]
    if backbone:
        grupos.append({"params": backbone, "lr": args.lr_backbone})
    otimizador = torch.optim.Adam(grupos)
    loader = DataLoader(treino, batch_size=args.batch, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    pesos_tensor = torch.tensor([pesos[0], pesos[1]], dtype=torch.float32, device=dispositivo)
    melhor_perda, melhor_estado, melhor_epoca = float("inf"), None, -1
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        manter_batchnorm_em_avaliacao(modelo)
        soma_perda, quantidade = 0.0, 0
        for x, y in loader:
            x, y = x.to(dispositivo), y.to(dispositivo)
            logits = modelo(x).squeeze(1)
            perdas = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            perda = (perdas * pesos_tensor[y.long()]).mean()
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            soma_perda += perda.item() * len(y)
            quantidade += len(y)
        p_val = prever(modelo, validacao, dispositivo)
        perda_val = perda_validacao(validacao.y, p_val, pesos)
        auc_val = calcular_metricas(validacao.y, p_val)["auc"]
        log.info("epoca=%02d perda_treino=%.4f perda_val=%.4f auc_val=%.4f", epoca, soma_perda / quantidade, perda_val, auc_val)
        if perda_val < melhor_perda:
            melhor_perda, melhor_epoca = perda_val, epoca
            melhor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}

    modelo.load_state_dict(melhor_estado)
    log.info("melhor_epoca_validacao=%d", melhor_epoca)
    df_val = tabela_predicoes(validacao, "validacao", prever(modelo, validacao, dispositivo))
    df_val = validar_com_manifesto(df_val, pd.read_csv(args.manifesto))
    escolha = escolher_threshold_validacao(df_val)
    metricas = avaliar_com_threshold(df_val, escolha["threshold"])
    log.info("threshold_validacao=%s", escolha)
    log.info("metricas_validacao=%s", metricas)
    df_val.to_csv(args.saida / "predicoes_validacao.csv", index=False)
    gerar_curva_roc(df_val, args.saida / "roc_validacao.png", threshold=escolha["threshold"])
    torch.save(modelo.state_dict(), args.saida / "modelo_efficientnet.pt")
    log.info("teste_nao_avaliado=true")


if __name__ == "__main__":
    main()
