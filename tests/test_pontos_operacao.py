"""Testes sintéticos da regra de pontos de operação; não abrem dados reais."""
import unittest

import numpy as np
import pandas as pd

from mineracao import pontos_operacao as po


# 5 malignos e 10 benignos. Ordenado por probabilidade decrescente:
# M .95, M .90, B .85, M .80, B .70, B .70, M .60, B .50, M .40, B .30 ... B .05
Y = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
P = np.array([.95, .90, .85, .80, .70, .70, .60, .50, .40, .30, .25, .20, .15, .10, .05])


class PontoOperacao(unittest.TestCase):
    def test_sens_com_espec_minima_usa_piso_e_limiar_real(self):
        r = po.ponto_operacao(Y, P, "sensibilidade", 0.90)
        # espec >= 0,90 com 10 benignos => no máximo 1 FP => limiar .80 (3 TP, 1 FP)
        self.assertEqual((r["tp"], r["fp"]), (3, 1))
        self.assertAlmostEqual(r["limiar"], 0.80)
        self.assertGreaterEqual(r["especificidade"], 0.90)

    def test_espec_com_sens_minima_desempata_pela_melhor_acompanhante(self):
        # sens >= 0,6 => 3 TP; o limiar .80 dá 3 TP com 1 FP; .70 já tem 3 FP
        r = po.ponto_operacao(Y, P, "especificidade", 0.60)
        self.assertEqual((r["tp"], r["fp"]), (3, 1))
        r2 = po.ponto_operacao(Y, P, "especificidade", 0.40)
        # 2 TP com 0 FP (limiar .90) é melhor que 3 TP com 1 FP.
        self.assertEqual((r2["tp"], r2["fp"]), (2, 0))

    def test_empate_na_meta_escolhe_ponto_nao_dominado(self):
        # B .9, M .8, M .4, B .3, B .2: limiares .8 e .4 dão espec 2/3,
        # com sens 0,5 e 1,0 respectivamente.
        y = np.array([0, 1, 1, 0, 0])
        p = np.array([.9, .8, .4, .3, .2])
        r = po.ponto_operacao(y, p, "especificidade", 0.5)
        self.assertEqual((r["tp"], r["fp"]), (2, 1))  # não (1, 1)
        self.assertAlmostEqual(r["limiar"], 0.4)

    def test_restricao_em_contagem_inteira(self):
        # 0,90 × 10 = 9 exatamente: espec 0,9 cumpre o piso (sem erro de float)
        r = po.ponto_operacao(Y, P, "sensibilidade", 0.90)
        self.assertEqual(r["tn"], 9)

    def test_interpolacao_em_trecho_plano_coincide(self):
        r = po.ponto_operacao(Y, P, "sensibilidade", 0.90)
        self.assertAlmostEqual(po.ponto_interpolado(Y, P, "sensibilidade", 0.90),
                               r["sensibilidade"])

    def test_interpolacao_em_trecho_inclinado(self):
        # Empate entre maligno e benigno em .5 cria segmento diagonal na ROC.
        y = np.array([1, 1, 0, 0])
        p = np.array([.9, .5, .5, .1])
        # Vértices: (fpr 0, sens .5) -> (fpr .5, sens 1). Em fpr .25: sens .75
        self.assertAlmostEqual(po.ponto_interpolado(y, p, "sensibilidade", 0.75), 0.75)
        self.assertAlmostEqual(po.ponto_interpolado(y, p, "especificidade", 0.75), 0.75)

    def test_bootstrap_reprodutivel_e_ic_contem_estimativa(self):
        dados = pd.DataFrame({"nodule_id": [f"N{i}" for i in range(len(Y))],
                              "patient_id": [f"P{i // 2}" for i in range(len(Y))],
                              "split": "teste", "y_true": Y, "y_prob": P})
        a = po.bootstrap_ponto(dados, "sensibilidade", 0.90, n_repeticoes=200, seed=1)
        b = po.bootstrap_ponto(dados, "sensibilidade", 0.90, n_repeticoes=200, seed=1)
        self.assertEqual(a, b)
        lo, hi = a["ic95"]["sensibilidade"]
        self.assertLessEqual(lo, hi)
        self.assertLessEqual(a["replicas_validas"], 200)


class LimiarMaxF1(unittest.TestCase):
    def dados(self, split="validacao"):
        return pd.DataFrame({"nodule_id": [f"N{i}" for i in range(len(Y))],
                             "patient_id": [f"P{i}" for i in range(len(Y))],
                             "split": split, "y_true": Y, "y_prob": P})

    def test_maximiza_f1_na_validacao(self):
        r = po.limiar_max_f1_validacao(self.dados())
        melhor = max(2 * ((P >= t) & (Y == 1)).sum()
                     / (((P >= t) & (Y == 1)).sum() * 2 + ((P >= t) & (Y == 0)).sum()
                        + ((P < t) & (Y == 1)).sum()) for t in np.unique(P))
        self.assertAlmostEqual(r["f1"], melhor)

    def test_recusa_teste(self):
        with self.assertRaises(ValueError):
            po.limiar_max_f1_validacao(self.dados("teste"))


if __name__ == "__main__":
    unittest.main()
