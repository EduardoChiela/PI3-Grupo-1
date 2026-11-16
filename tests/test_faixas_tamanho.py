"""Testes sintéticos das faixas de tamanho e da integração com o documento."""
import unittest

from mineracao import faixas_tamanho as ft


class Faixas(unittest.TestCase):
    def test_cortes_do_documento(self):
        self.assertEqual(ft.faixa(5.99), "< 6 mm")
        self.assertEqual(ft.faixa(6.0), "6 a 20 mm")   # 6 entra no meio
        self.assertEqual(ft.faixa(20.0), "6 a 20 mm")  # 20 entra no meio
        self.assertEqual(ft.faixa(20.01), "> 20 mm")

    def test_plural(self):
        self.assertEqual(ft._n(1, "maligno"), "1 maligno")
        self.assertEqual(ft._n(0, "maligno"), "0 malignos")


class Documento(unittest.TestCase):
    def test_substitui_so_entre_marcadores(self):
        ini, fim = ft.MARCAS["delong"]
        texto = f"antes\n{ini}\nvelho\n{fim}\ndepois"
        novo = ft.substituir(texto, "delong", f"{ini}\nnovo\n{fim}")
        self.assertEqual(novo, f"antes\n{ini}\nnovo\n{fim}\ndepois")

    def test_recusa_marcador_ausente_ou_duplicado(self):
        ini, fim = ft.MARCAS["faixas"]
        for texto in ("sem marcadores", f"{ini}{fim}{ini}{fim}"):
            with self.assertRaises(ValueError):
                ft.substituir(texto, "faixas", "x")


if __name__ == "__main__":
    unittest.main()
