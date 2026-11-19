"""Seleciona EfficientNet-B0 somente pelo recall na validação (Issue #44)."""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from pathlib import Path

import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "mineracao/resultados/efficientnet_s6"
CONFIGURACOES = ("congelado", "ultimo_bloco", "dois_ultimos_blocos")


def ler_log(configuracao: str) -> dict:
    caminho = BASE / configuracao / "log_treino.txt"
    texto = caminho.read_text(encoding="utf-8")

    def ultimo(padrao):
        achados = re.findall(padrao, texto)
        if not achados:
            raise ValueError(f"Campo ausente no log {caminho}: {padrao}")
        return achados[-1]

    metricas = ast.literal_eval(ultimo(r"metricas_validacao=(\{[^\n]+\})"))
    threshold = ast.literal_eval(ultimo(r"threshold_validacao=(\{[^\n]+\})"))
    predicoes_path = BASE / configuracao / "predicoes_validacao.csv"
    predicoes = pd.read_csv(predicoes_path)
    obrigatorias = {"nodule_id", "patient_id", "split", "y_true", "y_prob"}
    if not obrigatorias.issubset(predicoes.columns) or not predicoes.split.eq("validacao").all():
        raise ValueError(f"Predições inválidas ou não exclusivamente de validação: {predicoes_path}")
    sha = ultimo(r"npz_sha256=([0-9a-fA-F]{64})")
    n_treino, n_validacao, n_teste = map(int, ultimo(r"n treino=(\d+) validacao=(\d+) teste=(\d+)"))
    return {
        "nome": configuracao,
        "configuracao": ultimo(r"configuracao_finetuning=(\S+)"),
        "arquitetura": ultimo(r"arquitetura=(\S+)"),
        "parametros_treinaveis": int(ultimo(r"parametros_treinaveis=(\d+)")),
        "melhor_epoca": int(ultimo(r"melhor_epoca_validacao=(\d+)")),
        "threshold_youden_validacao": float(threshold["threshold"]),
        "recall": float(metricas["recall"]),
        "especificidade": float(metricas["especificidade"]),
        "f1": float(metricas["f1"]),
        "auc": float(metricas["auc"]),
        "npz_sha256": sha,
        "n_treino": n_treino,
        "n_validacao": n_validacao,
        "n_teste": n_teste,
        "predicoes_validacao": predicoes,
        "log_origem": caminho.relative_to(RAIZ).as_posix(),
    }


def main() -> None:
    resultados = [ler_log(nome) for nome in CONFIGURACOES]
    if len({r["npz_sha256"] for r in resultados}) != 1:
        raise ValueError("As configurações usaram arquivos NPZ diferentes.")
    if len({(r["n_treino"], r["n_validacao"], r["n_teste"]) for r in resultados}) != 1:
        raise ValueError("As configurações usaram splits de tamanhos diferentes.")
    referencia = resultados[0]["predicoes_validacao"]
    for resultado in resultados[1:]:
        atual = resultado["predicoes_validacao"]
        colunas = ["nodule_id", "patient_id", "y_true"]
        if not referencia[colunas].sort_values("nodule_id").reset_index(drop=True).equals(
            atual[colunas].sort_values("nodule_id").reset_index(drop=True)
        ):
            raise ValueError("As configurações não avaliaram os mesmos nódulos de validação.")
    maior_recall = max(r["recall"] for r in resultados)
    candidatos = [r for r in resultados if maior_recall - r["recall"] < 0.01]
    if len(candidatos) > 1:
        maior_especificidade = max(r["especificidade"] for r in candidatos)
        candidatos = [r for r in candidatos if math.isclose(
            r["especificidade"], maior_especificidade, rel_tol=0, abs_tol=1e-12
        )]
        maior_auc = max(r["auc"] for r in candidatos)
        candidatos = [r for r in candidatos if math.isclose(
            r["auc"], maior_auc, rel_tol=0, abs_tol=1e-12
        )]
    if len(candidatos) != 1:
        raise ValueError("Empate após aplicar o desempate pré-registrado; teste não avaliado.")
    vencedor = candidatos[0]
    for resultado in resultados:
        resultado.pop("predicoes_validacao")

    BASE.mkdir(parents=True, exist_ok=True)
    campos = ["nome", "configuracao", "arquitetura", "parametros_treinaveis", "melhor_epoca",
              "threshold_youden_validacao", "recall", "especificidade", "f1", "auc", "npz_sha256",
              "n_treino", "n_validacao", "n_teste", "log_origem"]
    with (BASE / "resumo_validacao.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)
    linhas = [
        "# EfficientNet-B0 — seleção na validação", "",
        "Critério pré-registrado: maior recall maligno na validação; se a diferença entre os melhores for < 0,01, maior especificidade e depois maior AUC.",
        "", "| Configuração | Parâmetros treináveis | Melhor época | Recall | Especificidade | F1 | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    linhas.extend(
        f"| {r['nome']} | {r['parametros_treinaveis']} | {r['melhor_epoca']} | {r['recall']:.4f} | "
        f"{r['especificidade']:.4f} | {r['f1']:.4f} | {r['auc']:.4f} |" for r in resultados
    )
    linhas.extend(["", f"Configuração escolhida: `{vencedor['nome']}`.",
                   "O conjunto de teste não foi usado na escolha."])
    (BASE / "resumo_validacao.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    escolha = {
        "criterio": "maior recall da classe maligna na validacao",
        "regra_desempate": "se diferenca de recall < 0.01: maior especificidade; depois maior AUC",
        "teste_utilizado_na_escolha": False,
        "configuracao_escolhida": vencedor,
        "npz_sha256": vencedor["npz_sha256"],
        "experimentos_considerados": [r["nome"] for r in resultados],
    }
    (BASE / "escolha_validacao.json").write_text(
        json.dumps(escolha, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Escolhida na validação: {vencedor['nome']} (recall={vencedor['recall']:.4f})")
    print(f"Resumo: {BASE / 'resumo_validacao.md'}")
    print("Teste_utilizado_na_escolha: false")


if __name__ == "__main__":
    main()
