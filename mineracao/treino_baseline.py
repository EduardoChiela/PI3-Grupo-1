"""Baseline da Issue #14: CNN rasa (3 camadas convolucionais) em PyTorch.

Por que CNN rasa e não SVM+HOG: é a mesma família da ResNet com que será
comparada (mesma entrada 3x64x64, mesma perda com pesos de classe, mesma
augmentation, mesmo log), então a comparação isola a profundidade da rede.
HOG exigiria outro pipeline de features e outra família de modelo.

Regras que o script impõe:
  * split vem do manifesto (mineracao/dataset.py);
  * augmentation só no treino; validação e teste sem transformação;
  * SHA-256 do .npz no log;
  * threshold escolhido só na validação; teste só com --avaliar-teste,
    UMA vez, depois de congelar arquitetura/épocas/threshold;
  * dado real só com --liberado-11 (depois de a #11 fechar).

Uso, na raiz do repositório:
  python -m mineracao.treino_baseline                      # sintético
  python -m mineracao.treino_baseline --liberado-11 --npz patches/patches_2_5d.npz
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # antes de importar torch
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
from torch.utils.data import DataLoader

from mineracao.avaliacao import (avaliar_com_threshold, bootstrap_por_paciente,
                                 calcular_metricas, escolher_threshold_validacao,
                                 gerar_curva_roc, validar_com_manifesto)
from mineracao.dataset import AugmentacaoFlip3D, carregar_particoes

SEED = 42
PESOS_REFERENCIA = {0: 0.966667, 1: 1.035714}  # benigno, maligno (issue #14)
log = logging.getLogger("baseline")


def fixar_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class CNNRasa(nn.Module):
    """3 x [Conv3x3 -> ReLU -> MaxPool2] (16/32/64 filtros) -> Linear -> logit."""

    def __init__(self):
        super().__init__()
        def bloco(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.conv = nn.Sequential(bloco(3, 16), bloco(16, 32), bloco(32, 64))
        self.cabeca = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                    nn.Linear(64 * 8 * 8, 1))

    def forward(self, x):
        return self.cabeca(self.conv(x))


def pesos_de_classe(y_treino: np.ndarray) -> dict[int, float]:
    """w = N / (2 * n_classe), só no treino (mesma fórmula de pesos_classe.py)."""
    n = len(y_treino)
    pesos = {c: round(n / (2 * int((y_treino == c).sum())), 6) for c in (0, 1)}
    if any(abs(pesos[c] - PESOS_REFERENCIA[c]) > 1e-6 for c in (0, 1)):
        raise ValueError(f"Pesos {pesos} diferem da referência {PESOS_REFERENCIA}.")
    return pesos


def prever(modelo, ds, dispositivo, batch=128) -> np.ndarray:
    modelo.eval()
    saida = []
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            x = np.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))])
            logit = modelo(torch.from_numpy(x).to(dispositivo)).squeeze(1)
            saida.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(saida).astype(float)


def bce_ponderada(y, p, pesos) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    w = np.where(y == 1, pesos[1], pesos[0])
    return float(np.mean(-w * (y * np.log(p) + (1 - y) * np.log(1 - p))))


def tabela(ds, nome_split, prob) -> pd.DataFrame:
    return pd.DataFrame({"nodule_id": ds.nodule_ids, "patient_id": ds.patient_ids,
                         "split": nome_split, "y_true": ds.y.astype(int), "y_prob": prob})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", type=Path, default=Path("sintetico/patches_sintetico.npz"))
    ap.add_argument("--manifesto", type=Path,
                    default=Path("transformacao/manifest_patches.csv"))
    ap.add_argument("--saida", type=Path, default=Path("mineracao/resultados/baseline"))
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sem-aug", action="store_true", help="desliga a augmentation")
    ap.add_argument("--liberado-11", action="store_true",
                    help="confirma que a #11 fechou; obrigatório para .npz real")
    ap.add_argument("--avaliar-teste", action="store_true",
                    help="avalia o teste UMA vez, com o threshold congelado")
    args = ap.parse_args()

    if "sintetico" not in args.npz.name and not args.liberado_11:
        raise SystemExit("Dado real bloqueado até a #11 fechar (use --liberado-11).")

    args.saida.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.saida / "log_treino.txt", "w")])
    fixar_seed(args.seed)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aug = None if args.sem_aug else AugmentacaoFlip3D(args.seed)
    particoes, sha = carregar_particoes(args.npz, args.manifesto, transform_treino=aug)
    treino, val, teste = (particoes[k] for k in ("treino", "validacao", "teste"))
    pesos = pesos_de_classe(treino.y)

    log.info("python=%s numpy=%s pandas=%s torch=%s so=%s dispositivo=%s",
             platform.python_version(), np.__version__, pd.__version__,
             torch.__version__, platform.platform(), dispositivo)
    log.info("seed=%d epocas=%d lr=%g batch=%d augmentation=%s",
             args.seed, args.epocas, args.lr, args.batch,
             "flip3d_so_treino" if aug else "nenhuma")
    log.info("npz=%s", args.npz)
    log.info("npz_sha256=%s", sha)
    log.info("manifesto=%s", args.manifesto)
    log.info("n treino=%d val=%d teste=%d | pesos benigno=%.6f maligno=%.6f",
             len(treino), len(val), len(teste), pesos[0], pesos[1])

    modelo = CNNRasa().to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    gerador = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(treino, batch_size=args.batch, shuffle=True,
                        generator=gerador, num_workers=0)
    w = torch.tensor([pesos[0], pesos[1]], dtype=torch.float32, device=dispositivo)

    melhor = (float("inf"), None, -1)  # (perda_val, estado, época)
    for ep in range(1, args.epocas + 1):
        modelo.train()
        soma, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(dispositivo), y.to(dispositivo)
            logit = modelo(x).squeeze(1)
            perda = (F.binary_cross_entropy_with_logits(logit, y, reduction="none")
                     * w[y.long()]).mean()
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            soma, n = soma + perda.item() * len(y), n + len(y)
        pv = prever(modelo, val, dispositivo)
        perda_val = bce_ponderada(val.y, pv, pesos)
        auc_val = calcular_metricas(val.y, pv)["auc"]
        log.info("epoca %02d  perda_treino=%.4f  perda_val=%.4f  auc_val=%.4f",
                 ep, soma / n, perda_val, auc_val)
        if perda_val < melhor[0]:
            melhor = (perda_val, {k: v.clone() for k, v in modelo.state_dict().items()}, ep)

    modelo.load_state_dict(melhor[1])
    log.info("melhor época (menor perda de validação): %d", melhor[2])

    manifesto = pd.read_csv(args.manifesto)
    df_val = validar_com_manifesto(tabela(val, "validacao", prever(modelo, val, dispositivo)),
                                   manifesto)
    escolha = escolher_threshold_validacao(df_val)
    log.info("threshold (Youden, validação): %s", escolha)
    gerar_curva_roc(df_val, args.saida / "roc_validacao.png", threshold=escolha["threshold"])
    df_val.to_csv(args.saida / "predicoes_validacao.csv", index=False)
    log.info("métricas validação: %s", avaliar_com_threshold(df_val, escolha["threshold"]))
    torch.save(modelo.state_dict(), args.saida / "modelo.pt")

    if args.avaliar_teste:
        df_te = validar_com_manifesto(tabela(teste, "teste", prever(modelo, teste, dispositivo)),
                                      manifesto)
        df_te.to_csv(args.saida / "predicoes_teste.csv", index=False)
        gerar_curva_roc(df_te, args.saida / "roc_teste.png", threshold=escolha["threshold"])
        log.info("TESTE: %s", avaliar_com_threshold(df_te, escolha["threshold"]))
        log.info("IC95 bootstrap por paciente: %s",
                 bootstrap_por_paciente(df_te, escolha["threshold"], seed=args.seed))


if __name__ == "__main__":
    main()
