"""Testes sintéticos; não abrem dados reais nem resultados históricos."""
from contextlib import redirect_stdout
from io import StringIO
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from mineracao import dataset as d
from mineracao import experimentos_sprint5 as e


def sintetico(raiz, teste_invalido=True):
    rng = np.random.default_rng(7)
    # Dois nódulos de cada paciente: o teste detecta divisão por nódulo.
    ids = [f"N{i:03}" for i in range(44)]
    pacientes = [f"P{i//2:03}" for i in range(44)]
    labels = ["benigno" if (i // 2) % 2 == 0 else "maligno" for i in range(44)]
    splits = ["treino"] * 28 + ["validacao"] * 12 + ["teste"] * 4
    pixels = rng.uniform(0, 1, (44, 3, 64, 64)).astype(np.float16)
    if teste_invalido:
        pixels[-4:] = np.nan
        labels[-4:] = ["TESTE_NAO_DEVE_SER_LIDO"] * 4
    man = pd.DataFrame(dict(nodule_id=ids, patient_id=pacientes, split=splits, label=labels))
    # NPZ fora da ordem do manifesto: o alinhamento deve usar ID.
    ordem = rng.permutation(len(ids))
    npz, csv = raiz / "sintetico.npz", raiz / "manifesto.csv"
    np.savez_compressed(npz, patches=pixels[ordem], nodule_id=np.array(ids)[ordem],
                        canais=np.array(d.CANAIS_ESPERADOS))
    man.to_csv(csv, index=False)
    return npz, csv, pixels, man


class Sprint5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raiz = Path(self.temp.name)
        self.npz, self.csv, self.pixels, self.man = sintetico(self.raiz)

    def test_loader_nao_acessa_array_inteiro_nem_labels_teste(self):
        original = np.lib.npyio.NpzFile.__getitem__
        def guard(z, key):
            if key == "patches":
                raise AssertionError("Acesso proibido ao array inteiro")
            return original(z, key)
        with patch.object(np.lib.npyio.NpzFile, "__getitem__", guard):
            ds, man, sha = d.carregar_desenvolvimento(self.npz, self.csv)
        self.assertEqual(len(ds), 40)
        self.assertEqual(set(man.split), {"treino", "validacao"})
        np.testing.assert_array_equal(ds.patches, self.pixels[:40])
        self.assertEqual(sha, d.sha256_arquivo(self.npz))

    def test_vazamento_de_paciente_inclusive_com_teste_bloqueado(self):
        self.man.loc[40, "patient_id"] = self.man.loc[0, "patient_id"]
        self.man.to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, "Paciente em mais de um split"):
            d.carregar_desenvolvimento(self.npz, self.csv)

    def test_nao_finitos_do_desenvolvimento_sao_rejeitados(self):
        self.pixels[0] = np.nan
        np.savez_compressed(self.npz, patches=self.pixels,
                            nodule_id=self.man.nodule_id.to_numpy(dtype=str),
                            canais=np.array(d.CANAIS_ESPERADOS))
        with self.assertRaisesRegex(ValueError, "não finitos"):
            d.carregar_desenvolvimento(self.npz, self.csv)

    def test_dobras_preservam_pacientes_e_cobrem_todos(self):
        ds, man, _ = d.carregar_desenvolvimento(self.npz, self.csv)
        dobras, mapa = e.construir_dobras(ds, man)
        self.assertEqual(len(dobras), 5)
        self.assertTrue(mapa.groupby("patient_id").dobra_validacao.nunique().eq(1).all())
        self.assertEqual(sorted(np.concatenate([v for _, v in dobras])), list(range(len(ds))))
        for it, iv in dobras:
            self.assertFalse(set(ds.patient_ids[it]) & set(ds.patient_ids[iv]))
        novamente, mapa2 = e.construir_dobras(ds, man)
        pd.testing.assert_frame_equal(mapa, mapa2)

    def test_pipeline_completo_retoma_e_nao_toca_comparacao(self):
        saida = self.raiz / "resultados"
        protegido = saida / "comparacao" / "sentinela.txt"
        protegido.parent.mkdir(parents=True)
        protegido.write_text("não alterar", encoding="utf-8")
        original_prever = e.prever
        chamadas = []
        def prever(modelo, ds, dispositivo):
            chamadas.append(tuple(ds.nodule_ids))
            return original_prever(modelo, ds, dispositivo)
        with redirect_stdout(StringIO()), patch.object(e, "prever", side_effect=prever):
            e.executar(self.npz, self.csv, saida, "tudo", epocas=2, real=False)
        # 6 runs com checkpoint: 2 épocas + previsão final. 5 folds: só 1 previsão.
        self.assertEqual(len(chamadas), 6 * 3 + 5)
        self.assertFalse(set(self.man.nodule_id.iloc[-4:]) & set(sum(chamadas, ())))
        self.assertEqual(protegido.read_text(encoding="utf-8"), "não alterar")
        seeds = pd.read_csv(saida / "sprint5/tabela_sementes.csv")
        cv = pd.read_csv(saida / "sprint5/tabela_cv.csv")
        self.assertEqual(seeds.seed.tolist(), list(e.SEMENTES))
        self.assertEqual(cv.dobra.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(cv.n_validacao.sum(), 40)
        resumo = json.loads((saida / "sprint5/tabela_sementes_resumo.json").read_text())
        self.assertAlmostEqual(resumo["desvio_amostral"], seeds.auc.std(ddof=1))
        self.assertFalse((saida / "sprint5/secao_resultados.md").exists())
        self.assertTrue((saida / "sprint5/comparacao_pesos.txt").is_file())
        logs = list(saida.rglob("log_treino.txt"))
        self.assertEqual(len(logs), 11)
        for log in logs:
            self.assertIn(d.sha256_arquivo(self.npz), log.read_text(encoding="utf-8"))
        # Segunda chamada precisa reutilizar; nenhum treino adicional.
        with redirect_stdout(StringIO()), patch.object(e, "CNNRasa", side_effect=AssertionError("retreinou")):
            e.executar(self.npz, self.csv, saida, "tudo", epocas=2, real=False)
        # Impede reutilizar evidência alterada.
        logs[0].write_text("adulterado", encoding="utf-8")
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, "Resultado alterado"):
            e.executar(self.npz, self.csv, saida, "tudo", epocas=2, real=False)

    def test_cinco_sementes_preservam_algoritmo_do_baseline(self):
        # 15/14 produz exatamente os pesos de referência originais.
        rng = np.random.default_rng(19)
        n = 39
        labels = ["benigno"] * 15 + ["maligno"] * 14 + ["benigno", "maligno"] * 5
        man = pd.DataFrame({"nodule_id": [f"N{i}" for i in range(n)],
                            "patient_id": [f"P{i}" for i in range(n)],
                            "split": ["treino"] * 29 + ["validacao"] * 8 + ["teste"] * 2,
                            "label": labels})
        man.to_csv(self.csv, index=False)
        np.savez_compressed(self.npz, patches=rng.random((n, 3, 64, 64)).astype(np.float16),
                            nodule_id=man.nodule_id.to_numpy(dtype=str),
                            canais=np.array(d.CANAIS_ESPERADOS))
        original = self.raiz / "original"
        cmd = [sys.executable, "-m", "mineracao.treino_baseline", "--npz", str(self.npz),
               "--manifesto", str(self.csv), "--saida", str(original), "--epocas", "2"]
        subprocess.run(cmd, cwd=e.RAIZ, check=True, capture_output=True,
                       env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        ds, man, sha = d.carregar_desenvolvimento(self.npz, self.csv)
        cfg = {"seed": 42, "npz_sha256": sha, "manifesto_sha256": d.sha256_arquivo(self.csv),
               "epocas": 2, "lr": e.LR, "batch": e.BATCH, "ambiente": {"dispositivo": "cpu"}}
        nova = self.raiz / "nova"
        with redirect_stdout(StringIO()):
            e.treinar_uma(e.subset(ds, np.arange(29), 42), e.subset(ds, np.arange(29, 37)),
                          nova, cfg, selecionar_epoca=True, conferir_referencia=True)
        a = pd.read_csv(original / "predicoes_validacao.csv")
        b = pd.read_csv(nova / "predicoes_validacao.csv")
        pd.testing.assert_frame_equal(a, b, check_exact=True)


if __name__ == "__main__":
    unittest.main()
