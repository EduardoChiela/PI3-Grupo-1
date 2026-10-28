"""Testes de estresse do feedback do Checkpoint 2: rótulos embaralhados e faixas de tamanho.

Parte 1 — rótulos embaralhados (validação, nunca teste):
  * o split vem do manifesto (dataset.carregar_particoes), igual à #14;
  * os rótulos do TREINO são permutados DEPOIS da divisão por paciente
    (permutação preserva as contagens, logo os pesos de classe não mudam);
  * todo o resto é o baseline da #14: CNNRasa, seed 42, 30 épocas, lr 1e-3,
    batch 32, flip 3D só no treino, pesos de classe, época escolhida pela menor
    perda de validação, CPU;
  * 3 embaralhamentos (seeds abaixo) + 1 controle com os rótulos verdadeiros;
  * previsão registrada antes de rodar: AUC de validação dentro de
    0,5 ± 1,96 * EP_nulo para cada embaralhamento; controle perto de 0,90.

Parte 2 — composição e desempenho por faixa de diâmetro (teste congelado):
  * só lê predicoes_teste_pareadas.csv (#18); não treina nem reabre o teste;
  * cortes fixados antes de olhar resultados: 8,4 e 14,4 mm (tercis do teste);
  * composição das três faixas é o resultado principal;
  * AUC e recall só na faixa do meio, IC 95% por bootstrap por paciente, no
    limiar congelado do baseline (Youden na validação, #14). Leitura descritiva.

Uso, na raiz do repositório:
  python -m mineracao.teste_estresse --npz patches/patches_2_5d.npz
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # antes de importar torch

import argparse
import json
import logging
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mineracao.avaliacao import bootstrap_por_paciente, calcular_metricas
from mineracao.dataset import AugmentacaoFlip3D, carregar_particoes
from mineracao.treino_baseline import (SEED, CNNRasa, bce_ponderada, fixar_seed,
                                       pesos_de_classe, prever)

SEEDS_EMBARALHAMENTO = (1, 2, 3)
CORTES_MM = (8.4, 14.4)  # registrados na issue antes de olhar os resultados; "<=" no corte
LIMIAR_BASELINE = 0.4304952919483185  # Youden na validação, log da #14 (0,4305)
N_REPETICOES = 2000
log = logging.getLogger("estresse")


# ------------------------------------------------- parte 1: embaralhamento ----

def ep_auc_nula(n_pos: int, n_neg: int) -> float:
    """Erro-padrão da AUC sob H0 (Mann-Whitney sem empates)."""
    return math.sqrt((n_pos + n_neg + 1) / (12 * n_pos * n_neg))


def treinar(treino, val, y_treino, args, dispositivo) -> dict:
    """Mesmo laço do treino_baseline.py; só o vetor de rótulos do treino muda."""
    fixar_seed(SEED)
    treino.y = y_treino
    treino.transform = AugmentacaoFlip3D(SEED)  # mesmo estado inicial em toda rodada
    pesos = pesos_de_classe(treino.y)

    modelo = CNNRasa().to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr)
    loader = DataLoader(treino, batch_size=args.batch, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), num_workers=0)
    w = torch.tensor([pesos[0], pesos[1]], dtype=torch.float32, device=dispositivo)

    melhor = (float("inf"), None, -1)  # (perda_val, auc_val, época)
    aucs = []
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
        aucs.append(auc_val)
        log.info("  epoca %02d  perda_treino=%.4f  perda_val=%.4f  auc_val=%.4f",
                 ep, soma / n, perda_val, auc_val)
        if perda_val < melhor[0]:
            melhor = (perda_val, auc_val, ep)
    return {"epoca_escolhida": melhor[2], "auc_val_epoca_escolhida": melhor[1],
            "auc_val_ultima_epoca": aucs[-1], "auc_val_por_epoca": aucs}


def parte_rotulos(args) -> dict:
    dispositivo = torch.device("cpu")  # o log da #14 é de CPU
    particoes, sha = carregar_particoes(args.npz, args.manifesto)
    treino, val = particoes["treino"], particoes["validacao"]
    log.info("npz=%s", args.npz)
    log.info("npz_sha256=%s", sha)
    log.info("manifesto=%s", args.manifesto)
    log.info("n treino=%d val=%d | malignos treino=%d val=%d",
             len(treino), len(val), int(treino.y.sum()), int(val.y.sum()))

    n_pos, n_neg = int(val.y.sum()), int((val.y == 0).sum())
    ep = ep_auc_nula(n_pos, n_neg)
    faixa = (0.5 - 1.96 * ep, 0.5 + 1.96 * ep)
    log.info("PREVISAO: AUC de validacao embaralhada em [%.3f; %.3f] (0,5 +- 1,96 * %.4f)",
             *faixa, ep)

    y_original = treino.y.copy()
    rodadas = {}
    for nome, seed in [("controle", None)] + [(f"embaralhado_seed{s}", s)
                                             for s in SEEDS_EMBARALHAMENTO]:
        if seed is None:
            y = y_original.copy()
        else:  # embaralha só o treino, já separado por paciente
            y = np.random.default_rng(seed).permutation(y_original)
        iguais = float((y == y_original).mean())
        log.info("rodada %s: seed_embaralhamento=%s malignos=%d rotulos_inalterados=%.3f",
                 nome, seed, int(y.sum()), iguais)
        r = treinar(treino, val, y, args, dispositivo)
        dentro = None if seed is None else bool(  # a previsão nula não vale para o controle
            faixa[0] <= r["auc_val_epoca_escolhida"] <= faixa[1])
        r.update(seed_embaralhamento=seed, fracao_rotulos_inalterados=iguais,
                 dentro_da_previsao=dentro)
        log.info("RESULTADO %s: epoca=%d auc_val=%.4f auc_val_ultima=%.4f dentro_da_previsao=%s",
                 nome, r["epoca_escolhida"], r["auc_val_epoca_escolhida"],
                 r["auc_val_ultima_epoca"], r["dentro_da_previsao"])
        rodadas[nome] = r
    return {"npz_sha256": sha, "n_val": len(val), "malignos_val": n_pos,
            "ep_auc_nula": ep, "faixa_prevista": faixa, "rodadas": rodadas}


# -------------------------------------------------- parte 2: faixas ----

def parte_faixas(args) -> dict:
    pred = pd.read_csv(args.predicoes)
    nod = pd.read_csv(args.nodulos)
    df = pred.merge(nod[["nodule_id", "patient_id", "split", "rotulo_binario", "diametro_mm"]],
                    on="nodule_id", how="left", suffixes=("", "_sel"), validate="one_to_one")
    if (len(df) != 111 or df.diametro_mm.isna().any() or not df.split.eq("teste").all()
            or not df.patient_id.eq(df.patient_id_sel).all()
            or not df.y_true.eq(df.rotulo_binario.map({"benigno": 0, "maligno": 1})).all()):
        raise ValueError("Predições de teste e nodulos_com_split.csv não batem.")

    tercis = np.quantile(df.diametro_mm, [1 / 3, 2 / 3])
    log.info("tercis observados do diametro (teste): %.3f e %.3f mm | cortes usados: %s",
             *tercis, CORTES_MM)
    df["faixa"] = pd.cut(df.diametro_mm, [-np.inf, *CORTES_MM, np.inf],
                         labels=["pequeno", "medio", "grande"])

    composicao = []
    for faixa, g in df.groupby("faixa", observed=True):
        linha = {"faixa": faixa, "n": len(g), "pacientes": g.patient_id.nunique(),
                 "malignos": int(g.y_true.sum()), "benignos": int((g.y_true == 0).sum()),
                 "fracao_malignos": float(g.y_true.mean()),
                 "diametro_min": float(g.diametro_mm.min()),
                 "diametro_max": float(g.diametro_mm.max())}
        log.info("COMPOSICAO %s", linha)
        composicao.append(linha)

    meio = df[df.faixa == "medio"].rename(columns={"p_baseline": "y_prob"})
    meio = meio[["nodule_id", "patient_id", "split", "y_true", "y_prob"]]
    ic = bootstrap_por_paciente(meio, LIMIAR_BASELINE, n_repeticoes=N_REPETICOES, seed=SEED)
    pontual = calcular_metricas(meio.y_true, meio.y_prob, LIMIAR_BASELINE)
    log.info("FAIXA DO MEIO (baseline, limiar %.4f): %s", LIMIAR_BASELINE, pontual)
    log.info("IC95 bootstrap por paciente (faixa do meio): %s", ic)
    return {"tercis_observados_mm": tercis.tolist(), "cortes_mm": CORTES_MM,
            "composicao": composicao, "limiar_baseline": LIMIAR_BASELINE,
            "faixa_meio": {"pacientes": meio.patient_id.nunique(), "pontual": pontual,
                           "bootstrap": ic}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", type=Path, default=Path("patches/patches_2_5d.npz"))
    ap.add_argument("--manifesto", type=Path,
                    default=Path("transformacao/manifest_patches.csv"))
    ap.add_argument("--predicoes", type=Path,
                    default=Path("mineracao/resultados/comparacao/predicoes_teste_pareadas.csv"))
    ap.add_argument("--nodulos", type=Path, default=Path("selecao/nodulos_com_split.csv"))
    ap.add_argument("--saida", type=Path, default=Path("mineracao/resultados/estresse"))
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    args.saida.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(args.saida / "log_estresse.txt", "w",
                                                      encoding="utf-8")])
    log.info("python=%s numpy=%s pandas=%s torch=%s so=%s",
             platform.python_version(), np.__version__, pd.__version__,
             torch.__version__, platform.platform())
    log.info("seed_treino=%d seeds_embaralhamento=%s epocas=%d lr=%g batch=%d",
             SEED, SEEDS_EMBARALHAMENTO, args.epocas, args.lr, args.batch)

    resultados = {"rotulos_embaralhados": parte_rotulos(args),
                  "faixas_de_tamanho": parte_faixas(args)}
    (args.saida / "resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
