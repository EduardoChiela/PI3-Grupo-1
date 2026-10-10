"""Avaliação de predições prontas; não treina nem seleciona modelos no teste."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path

import numpy as np
import pandas as pd


METRICAS = ("recall", "especificidade", "f1", "auc")
COLUNAS = {"nodule_id", "patient_id", "split", "y_true", "y_prob"}


def _threshold(threshold):
    if not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold deve ser finito e estar entre 0 e 1.")


def _vetores(y_true, y_prob):
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Vetores devem ser unidimensionais, não vazios e do mesmo tamanho.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true deve conter apenas 0/1.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("y_prob deve ser finito e estar entre 0 e 1.")
    return y.astype(int), p


def validar_predicoes(dados: pd.DataFrame) -> pd.DataFrame:
    """Valida uma linha por nódulo e ausência de vazamento entre splits."""
    faltantes = COLUNAS - set(dados.columns)
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(faltantes)}")
    _vetores(dados.y_true, dados.y_prob)
    for coluna in ("patient_id", "nodule_id", "split"):
        if dados[coluna].isna().any() or dados[coluna].astype(str).str.strip().eq("").any():
            raise ValueError(f"{coluna} não pode estar ausente ou vazio.")
    if dados.nodule_id.duplicated().any():
        raise ValueError("nodule_id deve ser único.")
    if not dados.split.isin(["treino", "validacao", "teste"]).all():
        raise ValueError("split deve ser treino, validacao ou teste.")
    if dados.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("Paciente presente em mais de um split.")
    return dados.copy()


def validar_com_manifesto(dados, manifesto):
    """Confere identidade, split e rótulo; permite avaliar subconjuntos explícitos."""
    dados = validar_predicoes(dados)
    obrigatorias = {"nodule_id", "patient_id", "split", "label"}
    if obrigatorias - set(manifesto.columns):
        raise ValueError("Manifesto sem as colunas obrigatórias.")
    if manifesto.nodule_id.isna().any() or manifesto.nodule_id.duplicated().any():
        raise ValueError("Manifesto com nodule_id ausente ou duplicado.")
    if not manifesto.label.isin(["benigno", "maligno"]).all():
        raise ValueError("Manifesto com label inválido.")
    if manifesto.patient_id.isna().any() or manifesto.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("Manifesto com paciente ausente ou vazamento entre splits.")
    cruzado = dados.merge(manifesto[list(obrigatorias)], on="nodule_id", how="left",
                          suffixes=("", "_manifesto"), validate="one_to_one", indicator=True)
    if not cruzado._merge.eq("both").all():
        raise ValueError("Predições contêm nódulos ausentes do manifesto.")
    for coluna in ("patient_id", "split"):
        if not cruzado[coluna].eq(cruzado[coluna + "_manifesto"]).all():
            raise ValueError(f"{coluna} diverge do manifesto.")
    if not cruzado.y_true.eq(cruzado.label.map({"benigno": 0, "maligno": 1})).all():
        raise ValueError("y_true diverge do rótulo do manifesto.")
    return dados


def _roc(y, p):
    # Agrupa probabilidades empatadas antes de acumular TP/FP.
    ordem = np.argsort(-p, kind="stable")
    ys, ps = y[ordem], p[ordem]
    fins = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]
    tp = np.r_[0, np.cumsum(ys)[fins]]
    fp = np.r_[0, (np.arange(1, len(ys) + 1) - np.cumsum(ys))[fins]]
    return fp / (y == 0).sum(), tp / (y == 1).sum()


def calcular_metricas(y_true, y_prob, threshold=0.5):
    """Regra: probabilidade >= threshold é maligna. Indefinidos são NaN.

    Recall e especificidade são sempre retornados juntos, no mesmo threshold.
    AUC é independente do threshold; requer as duas classes.
    """
    _threshold(threshold)
    y, p = _vetores(y_true, y_prob)
    previsto = p >= threshold
    tp = int(((y == 1) & previsto).sum())
    tn = int(((y == 0) & ~previsto).sum())
    fp = int(((y == 0) & previsto).sum())
    fn = int(((y == 1) & ~previsto).sum())
    def dividir(a, b):
        return a / b if b else float("nan")
    auc = float("nan")
    if len(np.unique(y)) == 2:
        fpr, tpr = _roc(y, p)
        auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return {"threshold": float(threshold), "recall": dividir(tp, tp + fn),
            "especificidade": dividir(tn, tn + fp), "f1": dividir(2 * tp, 2 * tp + fp + fn),
            "auc": auc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def escolher_threshold_validacao(dados):
    """Maximiza Youden exclusivamente na validação; empate: maior threshold.

    Alterar o critério exige decisão metodológica ANTES do teste final.
    Não filtra silenciosamente entradas contendo treino ou teste.
    """
    dados = validar_predicoes(dados)
    if not dados.split.eq("validacao").all():
        raise ValueError("Escolha de threshold aceita exclusivamente validação.")
    y, p = _vetores(dados.y_true, dados.y_prob)
    if len(np.unique(y)) != 2:
        raise ValueError("Youden requer ambas as classes na validação.")
    candidatos = np.unique(np.r_[0.0, p, 1.0])
    resultados = [calcular_metricas(y, p, t) for t in candidatos]
    melhor = max(resultados, key=lambda m: (m["recall"] + m["especificidade"] - 1,
                                            m["threshold"]))
    return {"threshold": melhor["threshold"], "sensibilidade": melhor["recall"],
            "especificidade": melhor["especificidade"],
            "j": melhor["recall"] + melhor["especificidade"] - 1}


def avaliar_com_threshold(dados, threshold):
    """Aplica um valor já congelado, sem recalculá-lo."""
    dados = validar_predicoes(dados)
    if dados.split.nunique() != 1:
        raise ValueError("Avalie um split por vez.")
    return calcular_metricas(dados.y_true, dados.y_prob, threshold)


def gerar_curva_roc(dados, caminho=None, threshold=None):
    """Retorna AUC e opcionalmente salva ROC; marca threshold na validação.

    Fecha a figura inclusive em caso de falha ao salvar.
    """
    import matplotlib.pyplot as plt

    dados = validar_predicoes(dados)
    if dados.split.nunique() != 1:
        raise ValueError("Gere ROC de um split por vez.")
    y, p = _vetores(dados.y_true, dados.y_prob)
    if len(np.unique(y)) != 2:
        raise ValueError("ROC requer ambas as classes.")
    if dados.split.eq("validacao").all() and threshold is None:
        raise ValueError("Informe o threshold escolhido para marcar a ROC de validação.")
    metricas = calcular_metricas(y, p, 0.5 if threshold is None else threshold)
    fpr, tpr = _roc(y, p)
    fig, ax = plt.subplots()
    try:
        ax.plot(fpr, tpr, label=f"AUC = {metricas['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        if threshold is not None:
            ax.scatter(1 - metricas["especificidade"], metricas["recall"],
                       label=f"Threshold = {threshold:.4g}", zorder=3)
        ax.set(xlabel="1 − especificidade", ylabel="Sensibilidade (maligno)",
               title=f"ROC — {dados.split.iloc[0]}", xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        if caminho is not None:
            fig.savefig(Path(caminho), bbox_inches="tight")
    finally:
        plt.close(fig)
    return metricas["auc"]


def _amostrar_indices_por_paciente(grupos, rng):
    # Concatenar preserva TODOS os nódulos e a multiplicidade de cada sorteio.
    sorteados = rng.integers(0, len(grupos), size=len(grupos))
    return np.concatenate([grupos[i] for i in sorteados])


def bootstrap_por_paciente(dados, threshold, n_repeticoes=2000, seed=42):
    """IC percentil 95% com threshold fixo; informa réplicas válidas por métrica.

    Réplicas sem a classe necessária geram NaN e são excluídas apenas do IC
    daquela métrica. Nenhuma réplica escolhe um novo threshold.
    """
    dados = validar_predicoes(dados).reset_index(drop=True)
    ponto = avaliar_com_threshold(dados, threshold)
    if isinstance(n_repeticoes, bool) or not isinstance(n_repeticoes, (int, np.integer)) or n_repeticoes < 1:
        raise ValueError("n_repeticoes deve ser inteiro positivo.")
    grupos = list(dados.groupby("patient_id", sort=False).indices.values())
    rng = np.random.default_rng(seed)
    y, p = _vetores(dados.y_true, dados.y_prob)
    valores = {nome: [] for nome in METRICAS}
    for _ in range(n_repeticoes):
        indices = _amostrar_indices_por_paciente(grupos, rng)
        m = calcular_metricas(y[indices], p[indices], threshold)
        for nome in METRICAS:
            if np.isfinite(m[nome]):
                valores[nome].append(m[nome])
    return {nome: {"estimativa": ponto[nome],
                   "ic95": tuple(float(v) for v in np.percentile(valores[nome], [2.5, 97.5]))
                   if valores[nome] else (float("nan"), float("nan")),
                   "replicas_validas": len(valores[nome]), "replicas_totais": n_repeticoes}
            for nome in METRICAS}


def analisar_faixa_preta(dados, threshold, nodule_ids: Collection[str] | None = None):
    """Análise de subgrupo; identificação definitiva depende da Issue #13.

    Manifesto atual não possui tem_faixa_preta. Lista explícita é alternativa
    à coluna booleana; ausência de identificação nunca é inferida da imagem.
    """
    dados = validar_predicoes(dados)
    avaliar_com_threshold(dados, threshold)
    if nodule_ids is not None:
        if "tem_faixa_preta" in dados:
            raise ValueError("Forneça coluna booleana OU coleção de IDs, não ambas.")
        if isinstance(nodule_ids, str):
            raise ValueError("Forneça uma coleção de IDs, não uma string.")
        faixa = dados.nodule_id.isin(set(nodule_ids))
    else:
        if "tem_faixa_preta" not in dados:
            raise ValueError("Identificação de faixa preta ausente; depende da Issue #13.")
        faixa = dados.tem_faixa_preta
        if faixa.isna().any() or not pd.api.types.is_bool_dtype(faixa):
            raise ValueError("tem_faixa_preta deve ser booleana e sem ausentes.")
    resultado = {}
    for nome, mascara in (("com_faixa_preta", faixa), ("sem_faixa_preta", ~faixa)):
        parte = dados.loc[mascara]
        resultado[nome] = {"n": len(parte), "metricas":
                           avaliar_com_threshold(parte, threshold) if len(parte) else None}
    return resultado


def comparar_configuracoes(predicoes: Mapping[str, pd.DataFrame], thresholds: Mapping[str, float]):
    """Organiza métricas na validação, sem treinamento ou escolha de vencedor.

    Exige os mesmos nódulos, pacientes e rótulos para comparação pareada.
    """
    if len(predicoes) < 2 or set(predicoes) != set(thresholds):
        raise ValueError("Forneça duas ou mais configurações e seus thresholds.")
    referencia = None
    linhas = []
    for nome, dados in predicoes.items():
        dados = validar_predicoes(dados)
        if not dados.split.eq("validacao").all():
            raise ValueError("Comparação metodológica aceita exclusivamente validação.")
        identidade = dados.set_index("nodule_id")[["patient_id", "y_true"]].sort_index()
        if referencia is not None and not identidade.equals(referencia):
            raise ValueError("Configurações devem avaliar os mesmos nódulos e rótulos.")
        referencia = identidade
        linhas.append({"configuracao": nome, **avaliar_com_threshold(dados, thresholds[nome])})
    return pd.DataFrame(linhas).set_index("configuracao")
