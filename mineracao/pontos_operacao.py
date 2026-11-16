"""Sprint 6: pontos de operação clínicos e F1 no limiar de máximo F1 da validação.

Métricas primárias do documento de instruções que faltavam:
  * sensibilidade com especificidade fixada em 0,90 (espec >= 0,90);
  * especificidade com sensibilidade fixada em 0,85 (sens >= 0,85);
  * F1 no limiar que maximiza o F1 na VALIDAÇÃO (ao lado do F1 no Youden).

Não treina e não prevê. Lê só arquivos versionados:
  * predições de teste pareadas da comparação S5 (comparacao_s5/);
  * logs de treino (limiar de Youden e linha TESTE) do baseline e da ResNet final;
  * predições de validação de cada modelo, quando existirem.

Regra dos pontos clínicos (curva ROC empírica do teste, sem interpolação):
  os limiares candidatos são as probabilidades previstas (mais 0 e 1), com
  "probabilidade >= limiar é maligno". Entre os limiares que cumprem a
  restrição, escolhe-se o que maximiza a métrica-alvo; no empate, o de melhor
  métrica acompanhante (ponto não dominado da ROC). No bootstrap, o ponto é
  reescolhido em cada réplica (pacientes reamostrados, seed 42, 2000 réplicas).
  A interpolação linear entre os vértices vizinhos sai só como referência.

Conferências (aborta se falhar): predições de teste batem com a linha TESTE de
cada log; predições de validação reproduzem o limiar de Youden do mesmo log
(prova de que vêm da mesma execução); tudo passa por validar_com_manifesto.

Uso, na raiz do repositório:
  python -m mineracao.pontos_operacao
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mineracao.avaliacao import (_amostrar_indices_por_paciente, _vetores,
                                 bootstrap_por_paciente, calcular_metricas,
                                 escolher_threshold_validacao, validar_com_manifesto,
                                 validar_predicoes)
from mineracao.comparacao import ler_log

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "mineracao/resultados"
SAIDA = RESULTADOS / "pontos_operacao_s6"
DOC = RAIZ / "docs/resultados_comparacao.md"
INICIO = "<!-- inicio: pontos_operacao -->"
FIM = "<!-- fim: pontos_operacao -->"

SEED = 42
N_REPETICOES = 2000
ESPEC_MINIMA = 0.90
SENS_MINIMA = 0.85
MODELOS = ("baseline", "resnet")
NOMES = {"baseline": "CNN rasa (baseline)", "resnet": "ResNet18 (`layer3` + `layer4`)"}
PASTAS = {"baseline": RESULTADOS / "baseline", "resnet": RESULTADOS / "resnet_s5/final"}


# ------------------------------------------------------- curva e pontos ----

def _contagens(y, p):
    """TP, FP e limiar para cada limiar candidato (probabilidades, 0 e 1)."""
    candidatos = np.unique(np.r_[0.0, p, 1.0])
    pos, neg = np.sort(p[y == 1]), np.sort(p[y == 0])
    tp = len(pos) - np.searchsorted(pos, candidatos, side="left")
    fp = len(neg) - np.searchsorted(neg, candidatos, side="left")
    return candidatos, tp, fp, len(pos), len(neg)


def ponto_operacao(y_true, y_prob, alvo: str, minimo: float) -> dict:
    """Ponto da ROC empírica que maximiza `alvo` sob a restrição da outra métrica.

    alvo="sensibilidade": maior sensibilidade com especificidade >= minimo.
    alvo="especificidade": maior especificidade com sensibilidade >= minimo.
    Empate na métrica-alvo: melhor métrica acompanhante; persistindo (mesma
    matriz de confusão), o maior limiar. Sem interpolação: o ponto é sempre
    atingível por um limiar real.
    """
    if alvo not in ("sensibilidade", "especificidade"):
        raise ValueError("alvo deve ser 'sensibilidade' ou 'especificidade'.")
    y, p = _vetores(y_true, y_prob)
    t, tp, fp, n_pos, n_neg = _contagens(y, p)
    if not n_pos or not n_neg:
        return {k: float("nan") for k in ("sensibilidade", "especificidade", "limiar")}
    tn = n_neg - fp
    # Restrição em contagens inteiras, para não depender de arredondamento.
    if alvo == "sensibilidade":
        ok = tn >= math.ceil(minimo * n_neg - 1e-9)
        chave = np.lexsort((t, tn, tp))  # último = maior tp, depois maior tn, maior t
    else:
        ok = tp >= math.ceil(minimo * n_pos - 1e-9)
        chave = np.lexsort((t, tp, tn))
    escolhido = [i for i in chave if ok[i]][-1]
    return {"sensibilidade": float(tp[escolhido] / n_pos),
            "especificidade": float(tn[escolhido] / n_neg),
            "limiar": float(t[escolhido]),
            "tp": int(tp[escolhido]), "fn": int(n_pos - tp[escolhido]),
            "tn": int(tn[escolhido]), "fp": int(fp[escolhido])}


def ponto_interpolado(y_true, y_prob, alvo: str, minimo: float) -> float:
    """Referência: interpolação linear na ROC exatamente em espec/sens = minimo.

    Percorre a curva na ordem dos limiares (decrescente) e interpola entre os
    dois vértices vizinhos que cruzam o valor fixado. Não corresponde a um
    limiar real; por isso não é a regra adotada.
    """
    y, p = _vetores(y_true, y_prob)
    t, tp, fp, n_pos, n_neg = _contagens(y, p)
    ordem = np.argsort(-t, kind="stable")
    sens, fpr = tp[ordem] / n_pos, fp[ordem] / n_neg
    if alvo == "sensibilidade":  # sensibilidade em fpr = 1 - minimo
        x = 1 - minimo
        k = int(np.flatnonzero(fpr <= x + 1e-12)[-1])
        if math.isclose(fpr[k], x, abs_tol=1e-12) or k == len(fpr) - 1:
            return float(sens[k])
        return float(sens[k] + (sens[k + 1] - sens[k]) * (x - fpr[k]) / (fpr[k + 1] - fpr[k]))
    x = minimo  # especificidade em sens = minimo
    k = int(np.flatnonzero(sens >= x - 1e-12)[0])
    if math.isclose(sens[k], x, abs_tol=1e-12) or k == 0:
        return float(1 - fpr[k])
    f = fpr[k - 1] + (fpr[k] - fpr[k - 1]) * (x - sens[k - 1]) / (sens[k] - sens[k - 1])
    return float(1 - f)


def bootstrap_ponto(dados, alvo, minimo, n_repeticoes=N_REPETICOES, seed=SEED) -> dict:
    """IC 95% percentil reamostrando pacientes; o ponto é reescolhido em cada réplica."""
    dados = validar_predicoes(dados).reset_index(drop=True)
    y, p = _vetores(dados.y_true, dados.y_prob)
    grupos = list(dados.groupby("patient_id", sort=False).indices.values())
    rng = np.random.default_rng(seed)
    valores = {"sensibilidade": [], "especificidade": []}
    for _ in range(n_repeticoes):
        idx = _amostrar_indices_por_paciente(grupos, rng)
        r = ponto_operacao(y[idx], p[idx], alvo, minimo)
        if np.isfinite(r[alvo]):
            for k in valores:
                valores[k].append(r[k])
    ponto = ponto_operacao(y, p, alvo, minimo)
    return {**ponto,
            "ic95": {k: tuple(float(v) for v in np.percentile(valores[k], [2.5, 97.5]))
                     for k in valores},
            "interpolado_referencia": ponto_interpolado(y, p, alvo, minimo),
            "replicas_validas": len(valores[alvo]), "replicas_totais": n_repeticoes}


def limiar_max_f1_validacao(dados) -> dict:
    """Maximiza F1 exclusivamente na validação; empate: maior limiar (como no Youden)."""
    dados = validar_predicoes(dados)
    if not dados.split.eq("validacao").all():
        raise ValueError("Escolha de limiar aceita exclusivamente validação.")
    y, p = _vetores(dados.y_true, dados.y_prob)
    candidatos = np.unique(np.r_[0.0, p, 1.0])
    melhor = max((calcular_metricas(y, p, t) for t in candidatos),
                 key=lambda m: (m["f1"] if np.isfinite(m["f1"]) else -1, m["threshold"]))
    return melhor


# ------------------------------------------------------- conferências ----

def carregar_teste(logs, manifesto) -> pd.DataFrame:
    pareadas = pd.read_csv(RESULTADOS / "comparacao_s5/predicoes_teste_pareadas.csv")
    for m in MODELOS:
        df = validar_com_manifesto(pareadas.assign(split="teste", y_prob=pareadas[f"p_{m}"]),
                                   manifesto)
        recalculado = calcular_metricas(df.y_true, df.y_prob, logs[m]["threshold"])
        for chave, v in logs[m]["teste_log"].items():
            if not math.isclose(recalculado[chave], v, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{m}: {chave} das predições pareadas ≠ linha TESTE do log.")
    esperado = set(manifesto.loc[manifesto.split == "teste", "nodule_id"])
    if set(pareadas.nodule_id) != esperado:
        raise ValueError("Predições pareadas não cobrem exatamente o teste do manifesto.")
    return pareadas


def carregar_validacao(modelo, log, manifesto):
    """Devolve as predições de validação ou None se o CSV não estiver versionado."""
    caminho = PASTAS[modelo] / "predicoes_validacao.csv"
    if not caminho.exists():
        return None, caminho
    df = validar_com_manifesto(pd.read_csv(caminho), manifesto)
    youden = escolher_threshold_validacao(df)["threshold"]
    if not math.isclose(youden, log["threshold"], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{modelo}: Youden recalculado de {caminho} ({youden}) ≠ log "
                         f"({log['threshold']}). O CSV não é da mesma execução.")
    return df, caminho


# ------------------------------------------------------------- texto ----

def _f(v, casas=3):
    return f"{v:.{casas}f}".replace(".", ",")


def _ic(est, ic):
    return f"{_f(est)} [{_f(ic[0])}; {_f(ic[1])}]"


def _rel(caminho: Path) -> str:
    return caminho.relative_to(RAIZ).as_posix()


def gerar_secao(r) -> str:
    pc, f1 = r["pontos_clinicos"], r["f1"]
    se = {m: pc[m]["sens_com_espec_minima"] for m in MODELOS}
    es = {m: pc[m]["espec_com_sens_minima"] for m in MODELOS}
    yd = r["youden_teste"]
    b, rn = "baseline", "resnet"

    # Guardas: o texto abaixo afirma estas relações; se os números mudarem, o
    # script para em vez de publicar uma frase falsa.
    assert se[b]["sensibilidade"] > se[rn]["sensibilidade"], "ordem em sens@espec mudou"
    assert es[rn]["especificidade"] > es[b]["especificidade"], "ordem em espec@sens mudou"
    assert yd[b]["especificidade"] < ESPEC_MINIMA, "Youden da CNN já cumpre espec >= 0,90"
    assert yd[b]["recall"] >= SENS_MINIMA, "Youden da CNN deixou de cumprir sens >= 0,85"
    assert all(math.isclose(d["interpolado_referencia"], d[alvo])
               for m in MODELOS for d, alvo in ((se[m], "sensibilidade"),
                                                (es[m], "especificidade"))), \
        "interpolação deixou de coincidir com a regra adotada; revisar texto da 12.1"

    def linha_ponto(rotulo, d, alvo):
        return (f"| {rotulo} | " + " | ".join(
            f"{_ic(d[m][alvo], d[m]['ic95'][alvo])}" for m in MODELOS) + " |")

    def linha_acomp(d, outra):
        return (f"| ↳ {outra} no ponto (limiar) | " + " | ".join(
            f"{_f(d[m][outra])} ({_f(d[m]['limiar'], 4)})" for m in MODELOS) + " |")

    def linha_matriz(d):
        return ("| ↳ TP / FN / TN / FP | " + " | ".join(
            "{tp} / {fn} / {tn} / {fp}".format(**d[m]) for m in MODELOS) + " |")

    tabela_pontos = "\n".join([
        "| Ponto de operação (teste) | " + " | ".join(NOMES[m] for m in MODELOS) + " |",
        "|---|---|---|",
        linha_ponto("**Sensibilidade com especificidade ≥ 0,90**", se, "sensibilidade"),
        linha_acomp(se, "especificidade"),
        linha_matriz(se),
        linha_ponto("**Especificidade com sensibilidade ≥ 0,85**", es, "especificidade"),
        linha_acomp(es, "sensibilidade"),
        linha_matriz(es),
        "| Referência: Youden da validação (§3), sens / espec (limiar) | " + " | ".join(
            f"{_f(yd[m]['recall'])} / {_f(yd[m]['especificidade'])} "
            f"({_f(yd[m]['threshold'], 4)})" for m in MODELOS) + " |",
    ])

    def celula_f1(m, chave):
        d = f1[m].get(chave)
        if d is None:
            return ("**pendente** — `predicoes_validacao.csv` da rodada final "
                    "não versionado (ver abaixo)")
        return f"{_ic(d['estimativa'], d['ic95'])}"

    def celula_limiar(m):
        d = f1[m].get("max_f1_validacao")
        return "pendente" if d is None else _f(d["limiar"], 4)

    def celula_val(m):
        d = f1[m].get("max_f1_validacao")
        return "pendente" if d is None else _f(d["f1_validacao"])

    def celula_matriz(m):
        d = f1[m].get("max_f1_validacao")
        return "pendente" if d is None else "{tp} / {fn} / {tn} / {fp}".format(**d["teste"])

    tabela_f1 = "\n".join([
        "| F1 no teste (IC 95% por paciente) | " + " | ".join(NOMES[m] for m in MODELOS) + " |",
        "|---|---|---|",
        "| F1 no limiar de **Youden** (o reportado na §3) | " + " | ".join(
            celula_f1(m, "youden") for m in MODELOS) + " |",
        "| ↳ limiar (Youden, validação) | " + " | ".join(
            _f(yd[m]["threshold"], 4) for m in MODELOS) + " |",
        "| F1 no limiar de **máximo F1 da validação** (pedido no documento) | " + " | ".join(
            celula_f1(m, "max_f1_validacao") for m in MODELOS) + " |",
        "| ↳ limiar (máximo F1, validação) | " + " | ".join(
            celula_limiar(m) for m in MODELOS) + " |",
        "| ↳ F1 na própria validação nesse limiar | " + " | ".join(
            celula_val(m) for m in MODELOS) + " |",
        "| ↳ TP / FN / TN / FP no teste nesse limiar | " + " | ".join(
            celula_matriz(m) for m in MODELOS) + " |",
    ])

    pendentes = [m for m in MODELOS if f1[m].get("max_f1_validacao") is None]
    nota_pendente = ""
    if pendentes:
        nota_pendente = "\n" + "\n".join(
            f"> **Pendência — {NOMES[m]}.** O arquivo `{f1[m]['caminho_validacao']}` não está "
            "no repositório: a rodada final (#32, commit `d468daf`) versionou só o log e as "
            "predições de teste, e o `modelo_resnet.pt` também não foi guardado. Sem as "
            "probabilidades da validação não há como achar o limiar de máximo F1. Retreinar "
            "não serve: duas execuções com a mesma seed divergem (§9), então o limiar seria "
            "de outro modelo. Escolher o limiar no teste também não serve, porque seria "
            "escolher olhando o teste. Quando o CSV original for commitado, basta rodar "
            "`python -m mineracao.pontos_operacao` de novo: o script confere se ele "
            "reproduz o limiar de Youden do log e preenche esta célula."
            for m in pendentes) + "\n"

    coincide = [m for m in MODELOS if f1[m].get("max_f1_validacao") is not None
                and math.isclose(f1[m]["max_f1_validacao"]["limiar"], yd[m]["threshold"])]
    nota_coincide = "".join(
        f"\nNa {NOMES[m]}, o máximo de F1 na validação "
        f"({_f(f1[m]['max_f1_validacao']['f1_validacao'], 4)}) é atingido num único limiar, "
        "e ele é **o mesmo do Youden**. Por isso os dois F1 do teste são idênticos. "
        "Os dois limiares são calculados separadamente, cada um pelo seu critério, "
        "sobre as mesmas predições de validação.\n" for m in coincide)

    queda_fn = se[b]["fn"] - yd[b]["fn"]
    ganho_fp = yd[b]["fp"] - se[b]["fp"]
    return f"""{INICIO}
## 12. Sprint 6 — Pontos de operação clínicos

Esta seção é gerada por `python -m mineracao.pontos_operacao` (seed {SEED},
{N_REPETICOES} réplicas de bootstrap por paciente). Saídas em
`{_rel(SAIDA)}/`: `resultados.json` e `secao_resultados.md`. O script não
treina nem prevê. Ele lê as predições de teste pareadas da §3
(`comparacao_s5/predicoes_teste_pareadas.csv`) e confere que elas reproduzem a
linha `TESTE` de cada log. Também confere que cada CSV de validação reproduz o
limiar de Youden do mesmo log, o que prova que vem da mesma execução.

O documento de instruções define três métricas primárias: AUC-ROC (§3),
**sensibilidade com especificidade fixada em 0,90** e **especificidade com
sensibilidade fixada em 0,85**. As duas últimas estão aqui, junto com o F1 no
limiar que o documento pede.

### 12.1 Regra de escolha do ponto

Com {r['conferencia']['n_nodulos_teste']} nódulos, a ROC do teste é escalonada e
quase nunca passa exatamente por 0,90 ou 0,85. A regra adotada é:

1. **Limiares candidatos:** as probabilidades previstas no teste (mais 0 e 1),
   com "probabilidade ≥ limiar é maligno", a mesma convenção da §3.
2. **Restrição como piso:** especificidade **≥ 0,90** (ou sensibilidade
   **≥ 0,85**). Entre os limiares que cumprem o piso, fica o que dá a **maior**
   métrica-alvo. Como a curva anda em degraus, o ponto escolhido fica com a
   métrica fixada acima do piso (CNN: especificidade
   {_f(se[b]['especificidade'])}), e não exatamente em 0,90.
3. **Empate:** vários limiares dão a mesma métrica-alvo. Fica o de **melhor
   métrica acompanhante** (o ponto não dominado da ROC: mesma especificidade e
   mais malignos detectados, por exemplo).
4. **Sem interpolação.** Interpolar entre os dois vértices vizinhos dá um ponto
   que nenhum limiar real produz. Como referência, a interpolação daria
   sensibilidade {_f(se[b]['interpolado_referencia'])} (CNN) e
   {_f(se[rn]['interpolado_referencia'])} (ResNet) em especificidade 0,90, e
   especificidade {_f(es[b]['interpolado_referencia'])} (CNN) e
   {_f(es[rn]['interpolado_referencia'])} (ResNet) em sensibilidade 0,85. Nos
   quatro casos o cruzamento cai num trecho plano da escada, então as duas
   regras coincidem nestes dados. Em outra amostra poderiam divergir.
5. **IC 95%:** bootstrap reamostrando pacientes, como na §3. Em cada réplica o
   ponto é **reescolhido** pela mesma regra, porque o que se estima é "a
   sensibilidade quando a especificidade é fixada", e não o desempenho de um
   limiar fixo.

Os limiares destes pontos foram **lidos na curva do teste**. Eles descrevem a
curva e não são limiares congelados na validação como o de Youden. Usá-los
para operar o modelo daria um resultado otimista.

### 12.2 Tabela (teste, IC 95% por paciente)

{tabela_pontos}

Réplicas válidas: no mínimo {min(pc[m][k]['replicas_validas'] for m in MODELOS for k in pc[m])} de {N_REPETICOES} em cada linha.

**A ordem entre os modelos muda conforme o critério.** Com especificidade fixada
em 0,90, a CNN tem a sensibilidade maior ({_f(se[b]['sensibilidade'])} contra
{_f(se[rn]['sensibilidade'])}). Com sensibilidade fixada em 0,85, a ResNet tem
a especificidade maior ({_f(es[rn]['especificidade'])} contra
{_f(es[b]['especificidade'])}). As curvas se cruzam: nenhum modelo domina o
outro em todos os pontos de operação, apesar de a CNN ter a AUC maior. As
diferenças são pequenas perto da largura dos intervalos e não foram testadas
estatisticamente nesta seção.

### 12.3 F1 no limiar de máximo F1 da validação

A §3 reporta o F1 no limiar de Youden. O documento pede o F1 no limiar que
**maximiza o F1 na validação**. Esse limiar é escolhido só na validação
(empate: maior limiar, a mesma convenção do Youden) e aplicado congelado no
teste. O IC usa o limiar fixo, como na §3.

{tabela_f1}
{nota_coincide}{nota_pendente}
### 12.4 Ponto de operação adotado pelo grupo (atividade 4.4)

**O grupo mantém o limiar de Youden escolhido na validação** (CNN rasa:
{_f(yd[b]['threshold'], 4)}), com sensibilidade {_f(yd[b]['recall'])} e
especificidade {_f(yd[b]['especificidade'])} no teste. Isso fica **abaixo** dos
0,90 de especificidade que o documento trata como o ponto clinicamente
relevante, e a escolha é consciente. No ponto exigido (especificidade ≥ 0,90),
a sensibilidade cai de {_f(yd[b]['recall'])} para {_f(se[b]['sensibilidade'])}.
Nos {r['conferencia']['malignos']} malignos do teste, isso significa
{se[b]['fn']} cânceres não detectados em vez de {yd[b]['fn']}
({queda_fn} a mais), para evitar {ganho_fp} falsos positivos
({se[b]['fp']} em vez de {yd[b]['fp']} nos {r['conferencia']['benignos']}
benignos).

O trade-off que o grupo aceita é este: **errar um maligno custa mais que errar
um benigno.** O falso negativo atrasa o diagnóstico de um câncer. O falso
positivo leva a exames de seguimento (nova tomografia, eventualmente biópsia),
que têm custo e risco, mas são recuperáveis. Por isso o grupo prioriza a
sensibilidade e aceita especificidade abaixo de 0,90. O Youden também cumpre o
piso de sensibilidade de 0,85 com folga.

O limiar de Youden foi escolhido na validação e aplicado congelado. Os pontos
da §12.2 foram lidos na própria curva do teste. Então a comparação favorece os
pontos da §12.2, e a perda de sensibilidade real ao exigir especificidade 0,90
tende a ser pelo menos a mostrada aqui. A decisão vale para este conjunto, mais
fácil que o LIDC completo (§6 e §10). Em uso clínico real, o limiar teria de
ser recalibrado numa população de referência.
{FIM}
"""


# ---------------------------------------------------------------- main ----

def main() -> None:
    manifesto = pd.read_csv(RAIZ / "transformacao/manifest_patches.csv")
    logs = {m: ler_log(PASTAS[m] / "log_treino.txt") for m in MODELOS}
    if len({logs[m]["npz_sha256"] for m in MODELOS}) != 1:
        raise ValueError("Hash do .npz diverge entre os logs.")
    pareadas = carregar_teste(logs, manifesto)
    y = pareadas.y_true.to_numpy()

    pontos, youden, f1 = {}, {}, {}
    for m in MODELOS:
        dados = pareadas.assign(split="teste", y_prob=pareadas[f"p_{m}"])
        pontos[m] = {
            "sens_com_espec_minima": bootstrap_ponto(dados, "sensibilidade", ESPEC_MINIMA),
            "espec_com_sens_minima": bootstrap_ponto(dados, "especificidade", SENS_MINIMA),
        }
        youden[m] = calcular_metricas(y, dados.y_prob, logs[m]["threshold"])
        ic_youden = bootstrap_por_paciente(dados, logs[m]["threshold"], N_REPETICOES, SEED)
        f1[m] = {"youden": {"limiar": logs[m]["threshold"], **ic_youden["f1"]}}

        val, caminho = carregar_validacao(m, logs[m], manifesto)
        f1[m]["caminho_validacao"] = _rel(caminho)
        if val is None:
            f1[m]["max_f1_validacao"] = None
            continue
        escolha = limiar_max_f1_validacao(val)
        ic = bootstrap_por_paciente(dados, escolha["threshold"], N_REPETICOES, SEED)
        f1[m]["max_f1_validacao"] = {
            "limiar": escolha["threshold"], "f1_validacao": escolha["f1"],
            **ic["f1"], "teste": calcular_metricas(y, dados.y_prob, escolha["threshold"])}

    # O F1 no Youden tem de reproduzir exatamente o da §3 (mesma seed e método).
    s5 = json.loads((RESULTADOS / "comparacao_s5/resultados.json").read_text(encoding="utf-8"))
    for m in MODELOS:
        ref = s5["metricas_ic95_por_paciente"][m]["f1"]
        if not (math.isclose(ref["estimativa"], f1[m]["youden"]["estimativa"])
                and np.allclose(ref["ic95"], f1[m]["youden"]["ic95"])):
            raise ValueError(f"{m}: F1 no Youden não reproduz comparacao_s5/resultados.json.")

    resultado = {
        "conferencia": {"n_nodulos_teste": len(pareadas),
                        "n_pacientes_teste": int(pareadas.patient_id.nunique()),
                        "malignos": int(y.sum()), "benignos": int((y == 0).sum()),
                        "npz_sha256": logs["baseline"]["npz_sha256"]},
        "regra": {"especificidade_minima": ESPEC_MINIMA, "sensibilidade_minima": SENS_MINIMA,
                  "restricao": "piso (>=), sem interpolacao",
                  "empate": "melhor metrica acompanhante; depois maior limiar",
                  "bootstrap": "pacientes reamostrados; ponto reescolhido em cada replica",
                  "f1_limiar": "maximo F1 na validacao; empate: maior limiar"},
        "bootstrap": {"repeticoes": N_REPETICOES, "seed": SEED},
        "pontos_clinicos": pontos, "youden_teste": youden, "f1": f1,
    }
    secao = gerar_secao(resultado)

    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "resultados.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (SAIDA / "secao_resultados.md").write_text(secao, encoding="utf-8")

    texto = DOC.read_text(encoding="utf-8")
    if INICIO in texto:
        antes, resto = texto.split(INICIO, 1)
        _, depois = resto.split(FIM, 1)
        texto = antes + secao.rstrip() + depois
    else:
        marcador = "## Referências"
        if marcador not in texto:
            raise ValueError("Marcador de referências ausente; revisar integração.")
        texto = texto.replace(marcador, secao + "\n" + marcador, 1)
    DOC.write_text(texto, encoding="utf-8")

    for m in MODELOS:
        for nome, d in pontos[m].items():
            print(f"{m:8s} {nome}: sens {d['sensibilidade']:.3f} espec {d['especificidade']:.3f} "
                  f"limiar {d['limiar']:.4f} IC sens {d['ic95']['sensibilidade']} "
                  f"IC espec {d['ic95']['especificidade']}")
        mf = f1[m]["max_f1_validacao"]
        print(f"{m:8s} F1 Youden {f1[m]['youden']['estimativa']:.3f}; F1 máx-F1-val "
              + ("PENDENTE" if mf is None else f"{mf['estimativa']:.3f} (limiar {mf['limiar']:.4f})"))
    print(f"\nSeção atualizada em {_rel(DOC)}; saídas em {_rel(SAIDA)}")


if __name__ == "__main__":
    main()
