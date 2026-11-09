"""Issue #45: relatório da CV do baseline e da ResNet18, sem carregar o NPZ/teste.

Lê apenas as predições de validação das cinco dobras, calcula AUC,
sensibilidade e especificidade com threshold fixo 0,5, média/DP entre dobras
e IC95% por bootstrap de pacientes nas predições out-of-fold.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from mineracao.avaliacao import bootstrap_por_paciente, calcular_metricas, validar_predicoes

RAIZ = Path(__file__).resolve().parents[1]
S5 = RAIZ / "mineracao/resultados/sprint5"
S6 = RAIZ / "mineracao/resultados/sprint6"
THRESHOLD = 0.5
N_BOOTSTRAP = 2000
SEED_BOOTSTRAP = 42
METRICAS = ("auc", "recall", "especificidade")
NOMES = {
    "baseline": "CNN rasa (baseline)",
    "resnet": "ResNet18 (layer3+layer4)",
}
INICIO = "<!-- inicio: issue45 -->"
FIM = "<!-- fim: issue45 -->"


def _fmt(v):
    return f"{v:.6f}"


def gravar_json(caminho: Path, valor) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(valor, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def carregar_referencia():
    man = pd.read_csv(S5 / "manifesto_desenvolvimento.csv")
    dobras = pd.read_csv(S5 / "dobras.csv")
    obrig_man = {"nodule_id", "patient_id", "label"}
    obrig_dob = {"nodule_id", "patient_id", "dobra_validacao"}
    if obrig_man - set(man.columns) or obrig_dob - set(dobras.columns):
        raise ValueError("Manifesto/dobras da Sprint 5 incompletos.")
    ref = man[list(obrig_man)].merge(
        dobras[list(obrig_dob)],
        on=["nodule_id", "patient_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(ref) != len(man) or len(ref) != len(dobras):
        raise ValueError("Manifesto e dobras da Sprint 5 não cobrem os mesmos nódulos.")
    ref["y_true"] = ref.label.map({"benigno": 0, "maligno": 1})
    if ref.y_true.isna().any():
        raise ValueError("Rótulo inválido no manifesto de desenvolvimento.")
    if ref.groupby("patient_id").dobra_validacao.nunique().gt(1).any():
        raise ValueError("Paciente atribuído a mais de uma dobra.")
    return ref


def conferir_predicoes(caminho: Path, esperado: pd.DataFrame):
    df = validar_predicoes(pd.read_csv(caminho))
    if not df.split.eq("validacao").all():
        raise ValueError(f"{caminho}: esperado apenas split=validacao.")
    if set(df.nodule_id) != set(esperado.nodule_id):
        raise ValueError(f"{caminho}: nódulos não correspondem à dobra congelada.")
    chk = df.merge(
        esperado[["nodule_id", "patient_id", "y_true"]],
        on="nodule_id",
        how="left",
        suffixes=("", "_esperado"),
        validate="one_to_one",
    )
    if not chk.patient_id.eq(chk.patient_id_esperado).all():
        raise ValueError(f"{caminho}: patient_id diverge da dobra congelada.")
    if not chk.y_true.eq(chk.y_true_esperado).all():
        raise ValueError(f"{caminho}: y_true diverge do manifesto.")
    return df


def coletar():
    ref = carregar_referencia()
    linhas = []
    oof = {"baseline": [], "resnet": []}

    for dobra in range(1, 6):
        esperado = ref[ref.dobra_validacao == dobra]
        caminhos = {
            "baseline": S5 / "cv" / f"dobra_{dobra}" / "predicoes_validacao.csv",
            "resnet": S6 / "resnet18" / "cv" / f"dobra_{dobra}" / "predicoes_validacao.csv",
        }
        for modelo, caminho in caminhos.items():
            if not caminho.is_file():
                raise FileNotFoundError(f"Predições ausentes: {caminho}")
            df = conferir_predicoes(caminho, esperado)
            m = calcular_metricas(df.y_true, df.y_prob, THRESHOLD)
            linhas.append(
                {
                    "modelo": modelo,
                    "dobra": dobra,
                    "auc": m["auc"],
                    "sensibilidade": m["recall"],
                    "especificidade": m["especificidade"],
                    "tp": m["tp"],
                    "tn": m["tn"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                    "threshold": THRESHOLD,
                }
            )
            oof[modelo].append(df)

    tabela = pd.DataFrame(linhas)
    resumo = {}
    for modelo in ("baseline", "resnet"):
        todas = pd.concat(oof[modelo], ignore_index=True)
        todas = validar_predicoes(todas)
        if len(todas) != len(ref) or set(todas.nodule_id) != set(ref.nodule_id):
            raise ValueError(f"{modelo}: OOF não cobre exatamente o desenvolvimento.")
        boot = bootstrap_por_paciente(
            todas,
            THRESHOLD,
            n_repeticoes=N_BOOTSTRAP,
            seed=SEED_BOOTSTRAP,
        )
        resumo[modelo] = {}
        for metrica, coluna in (
            ("auc", "auc"),
            ("recall", "sensibilidade"),
            ("especificidade", "especificidade"),
        ):
            vals = tabela.loc[tabela.modelo == modelo, coluna].to_numpy(float)
            resumo[modelo][metrica] = {
                "media_5_dobras": float(vals.mean()),
                "desvio_padrao_amostral": float(vals.std(ddof=1)),
                "estimativa_oof": float(boot[metrica]["estimativa"]),
                "ic95_bootstrap_paciente": [
                    float(boot[metrica]["ic95"][0]),
                    float(boot[metrica]["ic95"][1]),
                ],
                "replicas_validas": int(boot[metrica]["replicas_validas"]),
                "replicas_totais": int(boot[metrica]["replicas_totais"]),
            }
    return tabela, resumo


def tabela_por_dobra_md(df: pd.DataFrame):
    linhas = [
        "| Modelo | Dobra | AUC | Sensibilidade | Especificidade |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        linhas.append(
            f"| {NOMES[r.modelo]} | {int(r.dobra)} | "
            f"{_fmt(r.auc)} | {_fmt(r.sensibilidade)} | {_fmt(r.especificidade)} |"
        )
    return "\n".join(linhas) + "\n"


def resumo_md(resumo):
    labels = {
        "auc": "AUC",
        "recall": "Sensibilidade",
        "especificidade": "Especificidade",
    }
    linhas = [
        "| Modelo | Métrica | Média das 5 dobras | DP amostral | Estimativa OOF | IC 95% bootstrap por paciente |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for modelo in ("baseline", "resnet"):
        for metrica in METRICAS:
            r = resumo[modelo][metrica]
            lo, hi = r["ic95_bootstrap_paciente"]
            linhas.append(
                f"| {NOMES[modelo]} | {labels[metrica]} | "
                f"{_fmt(r['media_5_dobras'])} | {_fmt(r['desvio_padrao_amostral'])} | "
                f"{_fmt(r['estimativa_oof'])} | [{_fmt(lo)}; {_fmt(hi)}] |"
            )
    return "\n".join(linhas) + "\n"


def gerar_secao(df, resumo):
    return f"""{INICIO}
### Extensão da validação cruzada — Sprint 6 (Issue #45)

A validação cruzada usa exatamente as cinco dobras agrupadas por paciente
congeladas na Sprint 5 (`mineracao/resultados/sprint5/dobras.csv`). Para os dois
modelos, a seed de treino é 42 e a avaliação ocorre somente após a época 30,
fixada antes das execuções. A dobra externa não seleciona checkpoint nem
threshold.

A ResNet18 usa a configuração escolhida na Issue #32, com `layer3` e `layer4`
liberadas para fine-tuning. O conjunto de teste não é carregado nem avaliado
nesta rotina.

Sensibilidade e especificidade exigem ponto de operação. Como escolher um
threshold pela própria dobra externa introduziria viés, usa-se **threshold fixo
0,5 para ambos os modelos**. A AUC independe desse threshold.

#### Métricas por dobra e por modelo

{tabela_por_dobra_md(df)}
#### Resumo das cinco dobras e IC 95%

{resumo_md(resumo)}
A média e o desvio-padrão são calculados sobre as cinco dobras. O IC 95% é
percentil por bootstrap de pacientes (2000 réplicas, seed 42) sobre as
predições out-of-fold concatenadas; portanto, o intervalo é do desempenho OOF
agregado, e não um intervalo t que trate as cinco dobras como independentes.
Cada paciente é reamostrado como unidade para preservar o agrupamento dos
nódulos.

Todos os números acima são regenerados por
`python -m mineracao.relatorio_sprint6`; o script lê somente predições de
validação versionadas e não acessa imagens nem o conjunto de teste.
{FIM}
"""


def atualizar_documentacao(secao: str):
    doc = RAIZ / "docs/resultados_comparacao.md"
    texto = doc.read_text(encoding="utf-8")
    if INICIO in texto:
        antes, resto = texto.split(INICIO, 1)
        if FIM not in resto:
            raise ValueError("Marcador issue45 incompleto na documentação.")
        _, depois = resto.split(FIM, 1)
        novo = antes + secao.rstrip() + depois
    else:
        marcador = "### Comparação com e sem pesos"
        if marcador not in texto:
            raise ValueError(
                "Não encontrei o ponto de inserção da Issue #45 dentro da seção 11."
            )
        novo = texto.replace(marcador, secao.rstrip() + "\n\n" + marcador, 1)
    doc.write_text(novo, encoding="utf-8")


def main():
    df, resumo = coletar()
    S6.mkdir(parents=True, exist_ok=True)
    df.to_csv(S6 / "metricas_por_dobra.csv", index=False)
    gravar_json(S6 / "resumo_metricas.json", resumo)
    secao = gerar_secao(df, resumo)
    (S6 / "secao_resultados.md").write_text(secao, encoding="utf-8")
    atualizar_documentacao(secao)
    print("Issue #45: métricas recalculadas e seção 11 atualizada.")
    print(f"Threshold fixo: {THRESHOLD}; bootstrap: {N_BOOTSTRAP} por paciente.")


if __name__ == "__main__":
    main()
