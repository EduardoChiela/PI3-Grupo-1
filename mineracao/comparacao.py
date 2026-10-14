"""Issue #18: compara baseline (#14) e ResNet (#17) no teste, com teste pareado.

Não treina, não prevê e não escolhe threshold: lê as predições de teste já
gravadas pelos dois scripts de treino e os thresholds congelados nos logs.
O teste, portanto, não é reaberto aqui; cada modelo foi avaliado nele uma vez.

Conferências antes de qualquer número (aborta se falhar):
  * SHA-256 do .npz idêntico nos dois logs (e, se o arquivo existir, igual ao atual);
  * cada log tem exatamente um threshold de Youden e uma avaliação de TESTE;
  * as duas tabelas passam por validar_com_manifesto e cobrem exatamente o split
    de teste do manifesto (mesmos nódulos, pacientes e rótulos);
  * as métricas recalculadas do CSV batem com a linha TESTE do log.

Decisão sobre independência (ameaça T7): a análise AGREGA POR PACIENTE.
  * IC 95%: bootstrap por paciente (avaliacao.bootstrap_por_paciente) e, para
    as diferenças, bootstrap pareado por paciente (mesmos sorteios nos dois modelos);
  * McNemar: versão para dados pareados em cluster (Durkalski et al., 2003),
    com o paciente como cluster; a versão exata por nódulo sai só como referência;
  * DeLong: versão em cluster (Obuchowski, 1997), que reduz ao DeLong quando
    cada paciente tem um nódulo; o DeLong por nódulo sai só como referência.

Uso, na raiz do repositório:
  python -m mineracao.comparacao
Dependências: numpy, pandas, matplotlib (sem scipy).
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mineracao.avaliacao import (METRICAS, _amostrar_indices_por_paciente, _roc,
                                 bootstrap_por_paciente, calcular_metricas,
                                 validar_com_manifesto)
from mineracao.dataset import sha256_arquivo

SEED = 42
N_REPETICOES = 2000
MODELOS = ("baseline", "resnet")  # diferenças sempre resnet − baseline
NOMES = {"baseline": "CNN rasa (baseline)", "resnet": "ResNet18 (transfer learning)"}
CORES = {"baseline": "#2a78d6", "resnet": "#eb6834"}
ESTILOS = {"baseline": ("-", "o"), "resnet": ("--", "s")}


# ----------------------------------------------------------------- logs ----

def _ler_texto(caminho: Path) -> str:
    bruto = Path(caminho).read_bytes()
    try:
        return bruto.decode("utf-8")
    except UnicodeDecodeError:  # logging no Windows grava em cp1252
        return bruto.decode("cp1252")


def ler_log(caminho: Path) -> dict:
    """Extrai sha256 do .npz, threshold de Youden e a linha TESTE de um log."""
    texto = _ler_texto(caminho)
    shas = re.findall(r"npz_sha256=([0-9a-f]{64})", texto)
    # Só o dicionário do Youden tem a chave 'j'.
    youden = [l for l in texto.splitlines() if "'threshold'" in l and "'j'" in l]
    testes = [l for l in texto.splitlines() if re.search(r"\bTESTE\s*[:=]", l)]
    if len(set(shas)) != 1 or len(youden) != 1 or len(testes) != 1:
        raise ValueError(f"{caminho}: esperava 1 sha256, 1 threshold e 1 TESTE; "
                         f"achou {len(set(shas))}, {len(youden)}, {len(testes)}.")
    def valor(linha, chave):
        return float(re.search(rf"'{chave}': ([-0-9.eE]+|nan)", linha).group(1))
    teste = {k: valor(testes[0], k) for k in ("threshold", *METRICAS, "tp", "tn", "fp", "fn")}
    manifesto = re.search(r"manifesto=(\S+)", texto)
    return {"npz_sha256": shas[0], "threshold": valor(youden[0], "threshold"),
            "teste_log": teste, "manifesto": manifesto.group(1) if manifesto else None}


# -------------------------------------------------------- conferências ----

def carregar_e_conferir(predicoes: dict, logs: dict, manifesto: pd.DataFrame, npz=None):
    """Aplica todas as conferências e devolve uma tabela alinhada por nódulo."""
    shas = {m: logs[m]["npz_sha256"] for m in MODELOS}
    if len(set(shas.values())) != 1:
        raise ValueError(f"Hash do .npz diverge entre os logs: {shas}. Retreinar.")
    if npz is not None and Path(npz).exists() and sha256_arquivo(npz) != shas[MODELOS[0]]:
        raise ValueError("O .npz atual não é o usado nos treinos.")

    esperado = manifesto[manifesto.split == "teste"].set_index("nodule_id")
    tabelas = {}
    for m in MODELOS:
        df = validar_com_manifesto(predicoes[m], manifesto)
        if not df.split.eq("teste").all() or set(df.nodule_id) != set(esperado.index):
            raise ValueError(f"{m}: predições não cobrem exatamente o teste do manifesto.")
        recalculado = calcular_metricas(df.y_true, df.y_prob, logs[m]["threshold"])
        for chave, v in logs[m]["teste_log"].items():
            if not math.isclose(recalculado[chave], v, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{m}: {chave} do CSV ({recalculado[chave]}) "
                                 f"≠ log ({v}). CSV e log não são da mesma execução.")
        tabelas[m] = df.set_index("nodule_id")

    a, b = (tabelas[m] for m in MODELOS)
    alinhada = a[["patient_id", "y_true"]].join(
        pd.DataFrame({f"p_{m}": tabelas[m].y_prob for m in MODELOS}))
    if not a.y_true.equals(b.y_true.loc[a.index]) or not a.patient_id.equals(
            b.patient_id.loc[a.index]):
        raise ValueError("Rótulos ou pacientes divergem entre as duas tabelas.")
    return alinhada.reset_index()


# ------------------------------------------------------------- McNemar ----

def _binomial_bilateral(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    cauda = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * cauda)


def mcnemar(acerto_a, acerto_b, grupos=None) -> dict:
    """b: só A acerta; c: só B acerta.

    Sem grupos: McNemar exato por nódulo (assume nódulos independentes).
    Com grupos: Durkalski et al. (2003), X² = (Σ(b_k − c_k))² / Σ(b_k − c_k)²,
    qui-quadrado com 1 g.l., paciente como cluster.
    """
    acerto_a, acerto_b = np.asarray(acerto_a, bool), np.asarray(acerto_b, bool)
    so_a, so_b = acerto_a & ~acerto_b, ~acerto_a & acerto_b
    b, c = int(so_a.sum()), int(so_b.sum())
    if grupos is None:
        return {"b_so_baseline_acerta": b, "c_so_resnet_acerta": c,
                "p_valor": _binomial_bilateral(b, b + c), "teste": "McNemar exato por nódulo"}
    d = pd.Series(so_a.astype(int) - so_b.astype(int)).groupby(np.asarray(grupos)).sum()
    denominador = float((d ** 2).sum())
    x2 = float(d.sum() ** 2 / denominador) if denominador else 0.0
    return {"b_so_baseline_acerta": b, "c_so_resnet_acerta": c,
            "pacientes_discordantes": int((d != 0).sum()), "estatistica_x2": x2,
            "p_valor": math.erfc(math.sqrt(x2 / 2)),
            "teste": "McNemar em cluster por paciente (Durkalski et al., 2003)"}


# -------------------------------------------------------------- DeLong ----

def _componentes(y, p):
    pos, neg = p[y == 1], p[y == 0]
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    return psi.mean(), psi.mean(axis=1), psi.mean(axis=0)  # AUC, V10, V01


def delong(y, probs: list, grupos=None) -> dict:
    """Compara duas AUC correlacionadas (mesmos casos): probs = [p_a, p_b].

    Sem grupos: DeLong et al. (1988). Com grupos: Obuchowski (1997), em que os
    componentes estruturais são somados por paciente antes da covariância.
    Diferença reportada: AUC_b − AUC_a, com IC 95% normal.
    """
    y = np.asarray(y, int)
    comps = [_componentes(y, np.asarray(p, float)) for p in probs]
    theta = np.array([c[0] for c in comps])
    m, n = int((y == 1).sum()), int((y == 0).sum())
    v10 = np.vstack([c[1] for c in comps])  # (2, m)
    v01 = np.vstack([c[2] for c in comps])  # (2, n)
    if grupos is None:
        cov = np.cov(v10) / m + np.cov(v01) / n
    else:
        grupos = np.asarray(grupos)
        ids = pd.unique(grupos)
        g_pos, g_neg = grupos[y == 1], grupos[y == 0]
        def por_cluster(v, g):
            return np.vstack([[v[r, g == i].sum() for i in ids] for r in range(len(probs))])
        m_i = np.array([(g_pos == i).sum() for i in ids])
        n_i = np.array([(g_neg == i).sum() for i in ids])
        e10 = por_cluster(v10, g_pos) - np.outer(theta, m_i)
        e01 = por_cluster(v01, g_neg) - np.outer(theta, n_i)
        k = len(ids) / (len(ids) - 1)
        s10, s01, s11 = k * e10 @ e10.T / m, k * e01 @ e01.T / n, k * e10 @ e01.T
        cov = s10 / m + s01 / n + (s11 + s11.T) / (m * n)
    dif = float(theta[1] - theta[0])
    ep = float(math.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    z = dif / ep if ep else float("nan")
    return {"auc": theta.tolist(), "diferenca": dif, "erro_padrao": ep,
            "ic95": (dif - 1.959964 * ep, dif + 1.959964 * ep), "z": z,
            "p_valor": math.erfc(abs(z) / math.sqrt(2)) if ep else float("nan"),
            "correlacao_auc": float(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1])),
            "teste": ("DeLong em cluster por paciente (Obuchowski, 1997)" if grupos is not None
                      else "DeLong por nódulo (DeLong et al., 1988)")}


# ------------------------------------------- bootstrap da diferença ----

def bootstrap_diferenca(alinhada, thresholds, n_repeticoes=N_REPETICOES, seed=SEED):
    """IC 95% percentil de (resnet − baseline), com o MESMO sorteio de pacientes
    nos dois modelos em cada réplica; thresholds congelados."""
    grupos = list(alinhada.groupby("patient_id", sort=False).indices.values())
    rng = np.random.default_rng(seed)
    y = alinhada.y_true.to_numpy()
    p = {m: alinhada[f"p_{m}"].to_numpy() for m in MODELOS}
    valores = {nome: [] for nome in METRICAS}
    for _ in range(n_repeticoes):
        idx = _amostrar_indices_por_paciente(grupos, rng)
        ma, mb = (calcular_metricas(y[idx], p[m][idx], thresholds[m]) for m in MODELOS)
        for nome in METRICAS:
            d = mb[nome] - ma[nome]
            if np.isfinite(d):
                valores[nome].append(d)
    ponto = {m: calcular_metricas(y, p[m], thresholds[m]) for m in MODELOS}
    return {nome: {"diferenca": ponto["resnet"][nome] - ponto["baseline"][nome],
                   "ic95": tuple(float(v) for v in np.percentile(valores[nome], [2.5, 97.5])),
                   "replicas_validas": len(valores[nome]), "replicas_totais": n_repeticoes}
            for nome in METRICAS}


# -------------------------------------------------------------- figura ----

def figura_roc(alinhada, thresholds, caminho):
    import matplotlib.pyplot as plt

    y = alinhada.y_true.to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=200)
    try:
        ax.plot([0, 1], [0, 1], ":", color="#8a8984", linewidth=1)
        for m in MODELOS:
            p = alinhada[f"p_{m}"].to_numpy()
            fpr, tpr = _roc(y, p)
            met = calcular_metricas(y, p, thresholds[m])
            linha, marcador = ESTILOS[m]
            ax.plot(fpr, tpr, linha, color=CORES[m], linewidth=2,
                    label=f"{NOMES[m]} — AUC {met['auc']:.3f}".replace(".", ","))
            x, yy = 1 - met["especificidade"], met["recall"]
            ax.plot(x, yy, marcador, markersize=9, color=CORES[m],
                    markeredgecolor="white", markeredgewidth=2, zorder=3)
            ax.annotate(f"limiar {thresholds[m]:.3f}".replace(".", ","), (x, yy),
                        xytext=(10, -14), textcoords="offset points", fontsize=8,
                        color="#52514e")
        ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="1 − especificidade",
               ylabel="Sensibilidade (classe maligna)",
               title="ROC no teste (111 nódulos, 69 pacientes)")
        ax.grid(color="#e6e5e0", linewidth=0.6)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.legend(loc="lower right", fontsize=8, frameon=False,
                  title="Marcador = ponto de operação (Youden na validação)",
                  title_fontsize=7)
        fig.savefig(caminho, bbox_inches="tight")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------- main ----

def _fmt(v, casas=3):
    return f"{v:.{casas}f}".replace(".", ",")


def tabela_markdown(intervalos, diferencas, thresholds, n_tp):
    nomes = {"recall": "Recall (maligno)", "especificidade": "Especificidade",
             "f1": "F1", "auc": "AUC-ROC"}
    linhas = ["| Métrica | " + " | ".join(NOMES[m] for m in MODELOS) +
              " | Diferença (ResNet − baseline) |", "|---|---|---|---|"]
    for nome in METRICAS:
        celulas = [f"{_fmt(intervalos[m][nome]['estimativa'])} "
                   f"[{_fmt(intervalos[m][nome]['ic95'][0])}; "
                   f"{_fmt(intervalos[m][nome]['ic95'][1])}]" for m in MODELOS]
        d = diferencas[nome]
        celulas.append(f"{_fmt(d['diferenca'])} [{_fmt(d['ic95'][0])}; {_fmt(d['ic95'][1])}]")
        linhas.append(f"| {nomes[nome]} | " + " | ".join(celulas) + " |")
    linhas.append("| Limiar (Youden, validação) | " + " | ".join(
        _fmt(thresholds[m], 4) for m in MODELOS) + " | — |")
    linhas.append("| TP / FN / TN / FP | " + " | ".join(
        "{tp} / {fn} / {tn} / {fp}".format(**n_tp[m]) for m in MODELOS) + " | — |")
    return "\n".join(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    raiz = Path("mineracao/resultados")
    ap.add_argument("--baseline", type=Path, default=raiz / "baseline")
    ap.add_argument("--resnet", type=Path, default=raiz / "resnet")
    ap.add_argument("--manifesto", type=Path, default=Path("transformacao/manifest_patches.csv"))
    ap.add_argument("--npz", type=Path, default=Path("patches/patches_2_5d.npz"))
    ap.add_argument("--saida", type=Path, default=raiz / "comparacao")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    pastas = {"baseline": args.baseline, "resnet": args.resnet}

    manifesto = pd.read_csv(args.manifesto)
    logs = {m: ler_log(pastas[m] / "log_treino.txt") for m in MODELOS}
    predicoes = {m: pd.read_csv(pastas[m] / "predicoes_teste.csv") for m in MODELOS}
    alinhada = carregar_e_conferir(predicoes, logs, manifesto, args.npz)
    thresholds = {m: logs[m]["threshold"] for m in MODELOS}
    y, grupos = alinhada.y_true.to_numpy(), alinhada.patient_id.to_numpy()

    intervalos, pontos = {}, {}
    for m in MODELOS:
        df = alinhada.assign(split="teste", y_prob=alinhada[f"p_{m}"])
        intervalos[m] = bootstrap_por_paciente(df, thresholds[m], N_REPETICOES, args.seed)
        pontos[m] = calcular_metricas(y, alinhada[f"p_{m}"], thresholds[m])
    diferencas = bootstrap_diferenca(alinhada, thresholds, seed=args.seed)

    acerto = {m: (alinhada[f"p_{m}"].to_numpy() >= thresholds[m]) == (y == 1) for m in MODELOS}
    probs = [alinhada[f"p_{m}"].to_numpy() for m in MODELOS]
    testes = {"mcnemar_cluster_paciente": mcnemar(acerto["baseline"], acerto["resnet"], grupos),
              "mcnemar_nodulo_referencia": mcnemar(acerto["baseline"], acerto["resnet"]),
              "delong_cluster_paciente": delong(y, probs, grupos),
              "delong_nodulo_referencia": delong(y, probs)}

    args.saida.mkdir(parents=True, exist_ok=True)
    figura_roc(alinhada, thresholds, args.saida / "roc_comparacao.png")
    tabela = tabela_markdown(intervalos, diferencas, thresholds, pontos)
    (args.saida / "tabela_comparativa.md").write_text(tabela + "\n", encoding="utf-8")
    alinhada.to_csv(args.saida / "predicoes_teste_pareadas.csv", index=False)

    conferencia = {
        "npz_sha256_baseline": logs["baseline"]["npz_sha256"],
        "npz_sha256_resnet": logs["resnet"]["npz_sha256"],
        "npz_sha256_arquivo_atual": sha256_arquivo(args.npz) if args.npz.exists() else None,
        "manifesto_nos_logs": {m: logs[m]["manifesto"] for m in MODELOS},
        "manifesto_sha256_atual": sha256_arquivo(args.manifesto),
        "n_nodulos_teste": len(alinhada), "n_pacientes_teste": int(alinhada.patient_id.nunique()),
        "malignos": int(y.sum()), "benignos": int((y == 0).sum()),
    }
    resultado = {"conferencia": conferencia, "thresholds": thresholds,
                 "metricas_ic95_por_paciente": intervalos,
                 "diferenca_resnet_menos_baseline": diferencas, "testes_pareados": testes,
                 "bootstrap": {"repeticoes": N_REPETICOES, "seed": args.seed}}
    (args.saida / "resultados.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print("Conferência:", json.dumps(conferencia, indent=2, ensure_ascii=False))
    print("\n" + tabela + "\n")
    for nome, t in testes.items():
        print(f"{nome}: {t['teste']} -> p = {t['p_valor']:.3g}  "
              + ", ".join(f"{k}={v}" for k, v in t.items() if k not in ("teste", "p_valor")))
    print(f"\nSaídas em {args.saida}")


if __name__ == "__main__":
    main()
