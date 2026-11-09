"""Sprint 6: o modelo enxerga algo além do tamanho do nódulo?

Parte A — faixas de tamanho exigidas pelo documento de instruções
  (< 6 mm, 6 a 20 mm, > 20 mm), no teste congelado: composição das três
  faixas e, na faixa do meio, AUC da CNN rasa com IC 95% por paciente, recall
  e especificidade no limiar congelado (Youden na validação). A AUC do
  diâmetro sozinho na faixa do meio sai só como referência descritiva.

Parte B — o teste pré-registrado na §8: DeLong em cluster por paciente
  (Obuchowski, 1997; a mesma função da §4) comparando a AUC da CNN rasa com a
  AUC do diâmetro sozinho nos mesmos 111 nódulos de teste. Regra registrada
  antes do resultado: se o IC 95% de (CNN − diâmetro) não incluir o zero, há
  evidência de que a CNN usa algo além do diâmetro; se incluir, não há.

Não treina, não prevê e não escolhe nada. Lê só arquivos versionados:
  * mineracao/resultados/comparacao_s5/predicoes_teste_pareadas.csv;
  * mineracao/resultados/baseline/log_treino.txt (limiar e linha TESTE);
  * selecao/nodulos_com_split.csv (diametro_mm, rótulo e paciente);
  * transformacao/manifest_patches.csv (quais nódulos têm patch).
O diâmetro entra direto como nota (maior = mais suspeito), sem treino.

Uso, na raiz do repositório:
  python -m mineracao.faixas_tamanho
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mineracao.avaliacao import bootstrap_por_paciente, calcular_metricas, validar_com_manifesto
from mineracao.comparacao import delong, ler_log

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "mineracao/resultados"
SAIDA = RESULTADOS / "faixas_tamanho_s6"
DOC = RAIZ / "docs/resultados_comparacao.md"
MARCAS = {"delong": ("<!-- inicio: delong_diametro -->", "<!-- fim: delong_diametro -->"),
          "faixas": ("<!-- inicio: faixas_documento -->", "<!-- fim: faixas_documento -->")}

SEED = 42
N_REPETICOES = 2000
FAIXAS = ("< 6 mm", "6 a 20 mm", "> 20 mm")


def faixa(diametro: float) -> str:
    """Cortes do documento de instruções; 6 e 20 mm entram na faixa do meio."""
    if diametro < 6:
        return FAIXAS[0]
    return FAIXAS[1] if diametro <= 20 else FAIXAS[2]


# ---------------------------------------------------------- carregamento ----

def carregar(manifesto: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pred = pd.read_csv(RESULTADOS / "comparacao_s5/predicoes_teste_pareadas.csv")
    nod = pd.read_csv(RAIZ / "selecao/nodulos_com_split.csv")
    df = pred.merge(nod[["nodule_id", "patient_id", "split", "rotulo_binario", "diametro_mm"]],
                    on="nodule_id", how="left", suffixes=("", "_sel"), validate="one_to_one")
    rotulo = df.rotulo_binario.map({"benigno": 0, "maligno": 1})
    if (df.diametro_mm.isna().any() or not df.split.eq("teste").all()
            or not df.patient_id.eq(df.patient_id_sel).all() or not df.y_true.eq(rotulo).all()):
        raise ValueError("Predições de teste e selecao/nodulos_com_split.csv não batem.")

    # As predições da CNN têm de reproduzir a linha TESTE do log do baseline.
    log = ler_log(RESULTADOS / "baseline/log_treino.txt")
    cnn = validar_com_manifesto(
        df[["nodule_id", "patient_id", "y_true"]].assign(split="teste", y_prob=df.p_baseline),
        manifesto)
    recalculado = calcular_metricas(cnn.y_true, cnn.y_prob, log["threshold"])
    for chave, v in log["teste_log"].items():
        if not math.isclose(recalculado[chave], v, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"CNN: {chave} das predições pareadas ≠ linha TESTE do log.")
    if set(df.nodule_id) != set(manifesto.loc[manifesto.split == "teste", "nodule_id"]):
        raise ValueError("Predições não cobrem exatamente o teste do manifesto.")
    df["faixa"] = df.diametro_mm.map(faixa)
    return df, log


def composicao(df: pd.DataFrame) -> list[dict]:
    linhas = []
    for nome in FAIXAS:
        g = df[df.faixa == nome]
        linhas.append({"faixa": nome, "n": len(g), "pacientes": int(g.patient_id.nunique()),
                       "malignos": int(g.y_true.sum()), "benignos": int((g.y_true == 0).sum()),
                       "diametro_min": float(g.diametro_mm.min()),
                       "diametro_max": float(g.diametro_mm.max())})
    return linhas


def composicao_conjunto(manifesto: pd.DataFrame) -> list[dict]:
    """Mesmas faixas nos 747 nódulos com patch (treino + validação + teste)."""
    nod = pd.read_csv(RAIZ / "selecao/nodulos_com_split.csv")
    nod = nod[nod.nodule_id.isin(manifesto.nodule_id)]
    if len(nod) != len(manifesto):
        raise ValueError("nodulos_com_split.csv não cobre todos os nódulos do manifesto.")
    nod = nod.assign(faixa=nod.diametro_mm.map(faixa),
                     y=nod.rotulo_binario.map({"benigno": 0, "maligno": 1}))
    return [{"faixa": nome, "n": int((nod.faixa == nome).sum()),
             "malignos": int(nod.y[nod.faixa == nome].sum()),
             "benignos": int((nod.y[nod.faixa == nome] == 0).sum())} for nome in FAIXAS]


def tabela_bootstrap(g: pd.DataFrame, coluna: str, limiar: float) -> dict:
    dados = g[["nodule_id", "patient_id", "y_true"]].assign(split="teste", y_prob=g[coluna])
    return bootstrap_por_paciente(dados, limiar, N_REPETICOES, SEED)


# ---------------------------------------------------------------- texto ----

def _f(v, casas=3):
    return f"{v:.{casas}f}".replace(".", ",")


def _ic(d, casas=3):
    return f"{_f(d['estimativa'], casas)} [{_f(d['ic95'][0], casas)}; {_f(d['ic95'][1], casas)}]"


def _n(qtd, palavra):
    return f"{qtd} {palavra}" + ("" if qtd == 1 else "s")


def _pct(a, b):
    return f"{round(100 * a / b)}%"


def texto_delong(r) -> str:
    dc, dn = r["delong"]["cluster_paciente"], r["delong"]["nodulo_referencia"]
    inclui_zero = dc["ic95"][0] <= 0 <= dc["ic95"][1]
    # O texto abaixo foi escrito para o caso em que o IC inclui o zero.
    assert inclui_zero, "IC não inclui o zero: reescrever a conclusão da §8"
    m = r["faixa_meio"]
    return f"""{MARCAS['delong'][0]}
**Teste pré-registrado: CNN contra o diâmetro.** Gerado por
`python -m mineracao.faixas_tamanho` (saídas em `mineracao/resultados/faixas_tamanho_s6/`).
O DeLong em cluster por paciente (Obuchowski, 1997), a mesma função da §4,
compara as duas AUC nos mesmos {r['conferencia']['n_nodulos']} nódulos de teste
({r['conferencia']['n_pacientes']} pacientes). O diâmetro entra direto como
nota, sem treino.

| Teste | AUC CNN | AUC diâmetro | Diferença (CNN − diâmetro) | IC 95% | p-valor |
|---|---|---|---|---|---|
| **DeLong em cluster por paciente** (principal) | {_f(dc['auc'][1], 4)} | {_f(dc['auc'][0], 4)} | {_f(dc['diferenca'], 4)} (EP {_f(dc['erro_padrao'], 4)}) | **[{_f(dc['ic95'][0], 4)}; {_f(dc['ic95'][1], 4)}]** | {_f(dc['p_valor'])} |
| DeLong por nódulo (referência) | {_f(dn['auc'][1], 4)} | {_f(dn['auc'][0], 4)} | {_f(dn['diferenca'], 4)} (EP {_f(dn['erro_padrao'], 4)}) | [{_f(dn['ic95'][0], 4)}; {_f(dn['ic95'][1], 4)}] | {_f(dn['p_valor'])} |

**Conclusão pela regra pré-registrada.** O IC 95% da diferença
([{_f(dc['ic95'][0], 4)}; {_f(dc['ic95'][1], 4)}]) **inclui o zero**. Pela regra
fixada antes do teste, **não há evidência de que a CNN use algo além do
diâmetro**. O ganho de {_f(dc['diferenca'], 3)} na AUC não se distingue do
acaso nesta amostra. Isso não prova que a CNN só olha o tamanho: com
{r['conferencia']['n_nodulos']} nódulos e um piso já em {_f(dc['auc'][0], 4)},
sobra pouco espaço para medir um ganho. O que os dados sustentam é que o
desempenho da CNN no teste é **compatível com o de um preditor só de
diâmetro**. Esta é a resposta à pergunta central do trabalho e deve ser
apresentada assim.

Na faixa de 6 a 20 mm, onde o tamanho decide menos (§10.2), a CNN tem AUC
{_f(m['cnn']['auc']['estimativa'])} e o diâmetro sozinho
{_f(m['diametro_auc']['estimativa'])}. Essa diferença é só descritiva: a
comparação não foi pré-registrada e não foi testada.
{MARCAS['delong'][1]}"""


def texto_faixas(r) -> str:
    comp, conj, m = r["composicao_teste"], r["composicao_conjunto"], r["faixa_meio"]
    peq, meio, gde = comp
    cpeq, _, cgde = conj
    # Guardas: o texto afirma estas contagens; se mudarem, o script para.
    assert peq["malignos"] == 0 and peq["benignos"] > 0, "faixa < 6 mm mudou"
    assert gde["benignos"] == 0 and gde["malignos"] > 0, "faixa > 20 mm mudou"
    assert meio["malignos"] > 0 and meio["benignos"] > 0, "faixa do meio sem as duas classes"
    total_n, total_m = sum(c["n"] for c in comp), sum(c["malignos"] for c in comp)
    total_pac = r["conferencia"]["n_pacientes"]

    linhas = "\n".join(
        f"| {c['faixa']} | {_f(c['diametro_min'], 2)} – {_f(c['diametro_max'], 2)} | {c['n']} "
        f"| {c['pacientes']} | {c['malignos']} | {c['benignos']} | {_pct(c['malignos'], c['n'])} |"
        for c in comp)
    linhas_conj = "\n".join(
        f"| {c['faixa']} | {c['n']} | {c['malignos']} | {c['benignos']} |" for c in conj)
    cnn, pt = m["cnn"], m["cnn_pontual"]
    return f"""{MARCAS['faixas'][0]}
### 10.2 Desempenho nas faixas exigidas: < 6, 6 a 20 e > 20 mm (teste)

O documento de instruções exige desempenho separado em **< 6 mm, 6 a 20 mm e
> 20 mm** e chama a análise de obrigatória. Esta é a análise exigida. Os
tercis usados antes ficam na §10.3, como análise adicional. Gerado por
`python -m mineracao.faixas_tamanho`, que só lê as predições de teste da §3
(`comparacao_s5/predicoes_teste_pareadas.csv`) e o `diametro_mm` de
`selecao/nodulos_com_split.csv`. O script não treina, não prevê e não escolhe
limiar. Os cortes vêm do documento, não dos dados. Os nódulos com exatamente 6
ou 20 mm entrariam na faixa do meio, mas no teste não há nenhum. O limiar é o
oficial da CNN rasa ({_f(r['limiar_cnn'], 4)}, Youden na validação).

**Composição (teste):**

| Faixa | Diâmetro (mm) | Nódulos | Pacientes | Malignos | Benignos | % malignos |
|---|---|---|---|---|---|---|
{linhas}
| Total | | {total_n} | {total_pac} | {total_m} | {total_n - total_m} | {_pct(total_m, total_n)} |

A soma de pacientes por faixa passa de {total_pac} porque alguns pacientes têm
nódulos em mais de uma faixa.

**Por que as faixas extremas não permitem medir desempenho.** **Não existe
nenhum nódulo maligno abaixo de 6 mm no teste** ({peq['n']} nódulos, todos
benignos). Sensibilidade e AUC nessa faixa seriam 0/0. Elas não podem ser
calculadas, nem com este modelo nem com um modelo melhor. Acima de 20 mm,
**todos os {gde['n']} nódulos são malignos**. Ali a especificidade e a AUC
também são indefinidas, e o tamanho sozinho acerta a faixa inteira. A pergunta
do documento, "o modelo erra mais em nódulos pequenos (< 6 mm)?", **não pode
ser respondida nesta amostra**. Isso é uma limitação declarada, e não um
resultado ruim do modelo.

O padrão não é acaso do sorteio do teste. Nos {sum(c['n'] for c in conj)}
nódulos com patch (treino, validação e teste), a composição é esta:

| Faixa | Nódulos | Malignos | Benignos |
|---|---|---|---|
{linhas_conj}

Abaixo de 6 mm há {_n(cpeq['malignos'], 'maligno')} em {cpeq['n']} nódulos.
Acima de 20 mm há {_n(cgde['benignos'], 'benigno')} em {cgde['n']}. O padrão vem dos
critérios de seleção. O conjunto exige que pelo menos três radiologistas
tenham marcado o nódulo e exclui a malignidade mediana 3 ("indeterminado").
O segundo filtro tira os casos que os radiologistas consideraram ambíguos.
É provável que seja ali que estariam os nódulos pequenos suspeitos e os
grandes de aparência benigna. O que sobra é quase separado pelo tamanho. Os arquivos versionados já trazem o conjunto
depois desse filtro, então não dá para contar aqui quantos nódulos de cada
faixa ele removeu.

**Desempenho na faixa de 6 a 20 mm** ({meio['n']} nódulos, {meio['pacientes']}
pacientes, {meio['malignos']} malignos), a única com as duas classes em número
suficiente. IC 95% por bootstrap por paciente, {N_REPETICOES} réplicas, seed
{SEED} (réplicas válidas: {min(v['replicas_validas'] for v in cnn.values())}/{N_REPETICOES}).

| Métrica | CNN rasa (faixa 6 a 20 mm) | Diâmetro sozinho (referência, faixa 6 a 20 mm) |
|---|---|---|
| AUC-ROC | {_ic(cnn['auc'])} | {_ic(m['diametro_auc'])} |
| Recall (maligno) | {_ic(cnn['recall'])} | — |
| Especificidade | {_ic(cnn['especificidade'])} | — |
| TP / FN / TN / FP | {pt['tp']} / {pt['fn']} / {pt['tn']} / {pt['fp']} | — |

**O que significa.** Mesmo dentro de 6 a 20 mm, o diâmetro sozinho ainda
separa bem as classes, porque a faixa é larga. A CNN fica acima dele nessa
faixa, mas os intervalos se sobrepõem e essa comparação não foi testada. O
teste formal contra o diâmetro, pré-registrado, é o do teste inteiro (§8), e
ele não mostrou diferença. A faixa do meio é onde o problema é de fato
difícil, e é nela que a AUC deve ser lida como medida de dificuldade.
{MARCAS['faixas'][1]}"""


def substituir(texto: str, chave: str, bloco: str) -> str:
    inicio, fim = MARCAS[chave]
    if texto.count(inicio) != 1 or texto.count(fim) != 1:
        raise ValueError(f"Marcadores de '{chave}' ausentes ou duplicados no documento.")
    antes, resto = texto.split(inicio, 1)
    _, depois = resto.split(fim, 1)
    return antes + bloco + depois


# ----------------------------------------------------------------- main ----

def main() -> None:
    manifesto = pd.read_csv(RAIZ / "transformacao/manifest_patches.csv")
    df, log = carregar(manifesto)
    y, grupos = df.y_true.to_numpy(), df.patient_id.to_numpy()
    probs = [df.diametro_mm.to_numpy(float), df.p_baseline.to_numpy()]  # dif = CNN − diâmetro

    meio = df[df.faixa == FAIXAS[1]]
    # AUC não muda com transformação monotônica; dividir pelo máximo só põe
    # o diâmetro em [0, 1], que é o que a função de bootstrap aceita.
    meio = meio.assign(diametro_escala=meio.diametro_mm / df.diametro_mm.max())
    resultado = {
        "conferencia": {"n_nodulos": len(df), "n_pacientes": int(df.patient_id.nunique()),
                        "malignos": int(y.sum()), "npz_sha256": log["npz_sha256"]},
        "limiar_cnn": log["threshold"],
        "auc_diametro_teste": calcular_metricas(y, df.diametro_mm / df.diametro_mm.max())["auc"],
        "delong": {"cluster_paciente": delong(y, probs, grupos),
                   "nodulo_referencia": delong(y, probs)},
        "regra_pre_registrada": "IC 95% de (CNN - diametro) sem o zero => evidencia de algo "
                                "alem do diametro; com o zero => nao ha",
        "composicao_teste": composicao(df),
        "composicao_conjunto": composicao_conjunto(manifesto),
        "faixa_meio": {
            "cnn": tabela_bootstrap(meio, "p_baseline", log["threshold"]),
            "cnn_pontual": calcular_metricas(meio.y_true, meio.p_baseline, log["threshold"]),
            "diametro_auc": tabela_bootstrap(meio, "diametro_escala", 0.5)["auc"],
        },
        "bootstrap": {"repeticoes": N_REPETICOES, "seed": SEED},
    }
    dc = resultado["delong"]["cluster_paciente"]
    if not math.isclose(dc["auc"][0], resultado["auc_diametro_teste"]):
        raise ValueError("AUC do diâmetro no DeLong diverge do cálculo direto.")
    resultado["regra_conclusao"] = ("ha evidencia" if not dc["ic95"][0] <= 0 <= dc["ic95"][1]
                                    else "nao ha evidencia")

    blocos = {"delong": texto_delong(resultado), "faixas": texto_faixas(resultado)}
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "resultados.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    texto = DOC.read_text(encoding="utf-8")
    for chave, bloco in blocos.items():
        texto = substituir(texto, chave, bloco)
    DOC.write_text(texto, encoding="utf-8")

    print(f"DeLong cluster CNN - diametro: {dc['diferenca']:.4f} "
          f"IC [{dc['ic95'][0]:.4f}; {dc['ic95'][1]:.4f}] p={dc['p_valor']:.3f} "
          f"-> {resultado['regra_conclusao']}")
    for c in resultado["composicao_teste"]:
        print(c)
    print("Faixa do meio, AUC CNN:", resultado["faixa_meio"]["cnn"]["auc"],
          "| diametro:", resultado["faixa_meio"]["diametro_auc"])


if __name__ == "__main__":
    main()
