"""Comparação pareada da EfficientNet selecionada com baseline e ResNet.

Usa predições já versionadas; não carrega imagens nem repete inferência no teste.
"""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mineracao.avaliacao import (
    METRICAS,
    _amostrar_indices_por_paciente,
    bootstrap_por_paciente,
    calcular_metricas,
    validar_com_manifesto,
)
from mineracao.comparacao import delong, ler_log, mcnemar


RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "mineracao/resultados"
N_REPETICOES = 2000
SEED = 42
PASTAS = {
    "baseline": RESULTADOS / "baseline",
    "resnet": RESULTADOS / "resnet_s5/final",
    "efficientnet": RESULTADOS / "efficientnet_s6",
}


def mcnemar_nomeado(acerto_a, acerto_b, grupos, nomes):
    resultado = mcnemar(acerto_a, acerto_b, grupos)
    return {
        "modelo_a": nomes[0],
        "modelo_b": nomes[1],
        "somente_a_acerta": resultado["b_so_baseline_acerta"],
        "somente_b_acerta": resultado["c_so_resnet_acerta"],
        "p_valor": resultado["p_valor"],
        "teste": resultado["teste"],
        **({"pacientes_discordantes": resultado["pacientes_discordantes"],
           "estatistica_x2": resultado["estatistica_x2"]}
          if "pacientes_discordantes" in resultado else {}),
    }


def main() -> None:
    manifesto = pd.read_csv(RAIZ / "transformacao/manifest_patches.csv")
    eficiencia = json.loads((PASTAS["efficientnet"] / "avaliacao_final.json").read_text(encoding="utf-8"))
    logs = {nome: ler_log(PASTAS[nome] / "log_treino.txt") for nome in ("baseline", "resnet")}
    hashes = {logs[m]["npz_sha256"] for m in logs} | {eficiencia["npz_sha256"]}
    if len(hashes) != 1:
        raise ValueError(f"Hash do NPZ diverge entre os modelos: {hashes}")

    thresholds = {
        "baseline": logs["baseline"]["threshold"],
        "resnet": logs["resnet"]["threshold"],
        "efficientnet": eficiencia["threshold_youden_validacao"],
    }
    predicoes = {
        "baseline": pd.read_csv(PASTAS["baseline"] / "predicoes_teste.csv"),
        "resnet": pd.read_csv(PASTAS["resnet"] / "predicoes_teste.csv"),
        "efficientnet": pd.read_csv(PASTAS["efficientnet"] / "predicoes_teste.csv"),
    }
    tabelas = {}
    for nome, df in predicoes.items():
        df = validar_com_manifesto(df, manifesto)
        esperado = set(manifesto.loc[manifesto.split == "teste", "nodule_id"])
        if not df.split.eq("teste").all() or set(df.nodule_id) != esperado:
            raise ValueError(f"{nome}: predições não cobrem exatamente o teste do manifesto.")
        recalculadas = calcular_metricas(df.y_true, df.y_prob, thresholds[nome])
        referencia_metricas = (
            logs[nome]["teste_log"] if nome in logs else eficiencia["metricas_youden_teste"]
        )
        for chave, valor in referencia_metricas.items():
            if chave in recalculadas and not math.isclose(
                recalculadas[chave], valor, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(f"{nome}: {chave} recalculada diverge do registro final.")
        tabelas[nome] = df.set_index("nodule_id").sort_index()
    referencia = tabelas["baseline"]
    for nome in ("resnet", "efficientnet"):
        atual = tabelas[nome]
        if not referencia.patient_id.equals(atual.patient_id) or not referencia.y_true.equals(atual.y_true):
            raise ValueError(f"{nome}: nódulos, pacientes ou rótulos divergem da baseline.")

    alinhada = pd.DataFrame({
        "patient_id": referencia.patient_id,
        "y_true": referencia.y_true,
        **{f"p_{nome}": tabelas[nome].y_prob for nome in PASTAS},
    }).reset_index(names="nodule_id")
    y = alinhada.y_true.to_numpy()
    grupos = alinhada.patient_id.to_numpy()

    metricas, pontos = {}, {}
    for nome in PASTAS:
        df = alinhada.assign(split="teste", y_prob=alinhada[f"p_{nome}"])
        metricas[nome] = bootstrap_por_paciente(df, thresholds[nome], N_REPETICOES, SEED)
        pontos[nome] = calcular_metricas(y, alinhada[f"p_{nome}"], thresholds[nome])

    diferencas, testes = {}, {}
    grupos_indices = list(alinhada.groupby("patient_id", sort=False).indices.values())
    for a, b in combinations(PASTAS, 2):
        chave = f"{b}_menos_{a}"
        rng = np.random.default_rng(SEED)
        valores = {m: [] for m in METRICAS}
        pa, pb = alinhada[f"p_{a}"].to_numpy(), alinhada[f"p_{b}"].to_numpy()
        for _ in range(N_REPETICOES):
            idx = _amostrar_indices_por_paciente(grupos_indices, rng)
            ma = calcular_metricas(y[idx], pa[idx], thresholds[a])
            mb = calcular_metricas(y[idx], pb[idx], thresholds[b])
            for m in METRICAS:
                d = mb[m] - ma[m]
                if np.isfinite(d):
                    valores[m].append(d)
        diferencas[chave] = {
            m: {
                "diferenca": pontos[b][m] - pontos[a][m],
                "ic95": [float(v) for v in np.percentile(valores[m], [2.5, 97.5])],
                "replicas_validas": len(valores[m]),
                "replicas_totais": N_REPETICOES,
            }
            for m in METRICAS
        }
        acerto_a = (pa >= thresholds[a]) == (y == 1)
        acerto_b = (pb >= thresholds[b]) == (y == 1)
        testes[chave] = {
            "mcnemar_cluster_paciente": mcnemar_nomeado(acerto_a, acerto_b, grupos, (a, b)),
            "mcnemar_nodulo_referencia": mcnemar_nomeado(acerto_a, acerto_b, None, (a, b)),
            "delong_cluster_paciente": delong(y, [pa, pb], grupos),
            "delong_nodulo_referencia": delong(y, [pa, pb]),
        }

    saida = {
        "conferencia": {
            "npz_sha256": next(iter(hashes)),
            "n_teste": len(alinhada),
            "n_pacientes_teste": int(alinhada.patient_id.nunique()),
            "malignos": int(y.sum()),
            "benignos": int((y == 0).sum()),
        },
        "thresholds_youden_validacao": thresholds,
        "metricas_ic95_bootstrap_paciente": metricas,
        "metricas_pontuais_teste": pontos,
        "diferencas_ic95_bootstrap_paciente": diferencas,
        "testes_pareados": testes,
        "bootstrap": {"repeticoes": N_REPETICOES, "seed": SEED, "unidade": "patient_id"},
    }
    caminho = PASTAS["efficientnet"] / "comparacao_pareada.json"
    caminho.write_text(json.dumps(saida, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(saida, indent=2, ensure_ascii=False, default=float))
    print(f"\nComparação gravada em {caminho}")


if __name__ == "__main__":
    main()
