"""Issue #31: cinco sementes, CV por paciente e comparação de pesos.

Uso: python -m mineracao.experimentos_sprint5 --etapa verificar
     python -m mineracao.experimentos_sprint5 --etapa tudo

Não há opção para avaliar teste. Resultados completos são reutilizados apenas
quando configuração, código, ambiente e hashes coincidem; execução incompleta
com a mesma configuração é reiniciada. Não salva checkpoints de modelos.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from mineracao.avaliacao import calcular_metricas, escolher_threshold_validacao
from mineracao.dataset import (AugmentacaoFlip3D, PatchesDataset,
                               carregar_desenvolvimento, sha256_arquivo,
                               validar_separacao_pacientes)
from mineracao.treino_baseline import (CNNRasa, bce_ponderada, fixar_seed,
                                      pesos_de_classe, prever, tabela)

# Declarados antes de observar resultados. Não procurar uma seed vencedora.
SEMENTES = (42, 43, 44, 45, 46)
SEED_CV = 42
N_DOBRAS = 5
EPOCAS = 30
LR = 1e-3
BATCH = 32
RAIZ = Path(__file__).resolve().parents[1]


def gravar_json(caminho, valor):
    caminho.write_text(json.dumps(valor, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")


def subset(ds, indices, seed=None):
    return PatchesDataset(ds.patches[indices], ds.y[indices],
                          ds.nodule_ids[indices], ds.patient_ids[indices],
                          None if seed is None else AugmentacaoFlip3D(seed))


def construir_dobras(ds, man):
    """Reaproveita a guarda do dataset; cada paciente valida uma única vez."""
    cv = StratifiedGroupKFold(n_splits=N_DOBRAS, shuffle=True, random_state=SEED_CV)
    atribuicao = np.full(len(ds), -1, dtype=int)
    dobras = []
    for numero, (it, iv) in enumerate(cv.split(np.zeros(len(ds)), ds.y,
                                              groups=ds.patient_ids), 1):
        atual = man.copy()
        atual["split"] = "treino"
        atual.loc[iv, "split"] = "validacao"
        validar_separacao_pacientes(atual)
        if set(np.unique(ds.y[it])) != {0, 1} or set(np.unique(ds.y[iv])) != {0, 1}:
            raise ValueError(f"Dobra {numero} sem as duas classes. Não troque seeds até passar.")
        if (atribuicao[iv] != -1).any():
            raise ValueError("Nódulo validado mais de uma vez.")
        atribuicao[iv] = numero
        dobras.append((it, iv))
    if (atribuicao == -1).any():
        raise ValueError("Nódulo sem dobra de validação.")
    mapa = man[["nodule_id", "patient_id"]].copy()
    mapa["dobra_validacao"] = atribuicao
    validar_separacao_pacientes(mapa.rename(columns={"dobra_validacao": "split"}))
    return dobras, mapa


def treinar_uma(treino, val, pasta, config, selecionar_epoca, sem_pesos=False,
               conferir_referencia=False):
    """Mesma CNN, Adam, BCE, augmentation e regra de checkpoint do baseline.

    Sementes: menor BCE de validação, como no original.
    CV: última época prefixada; a dobra externa é prevista só ao terminar.
    """
    if set(np.unique(treino.y)) != {0, 1} or set(np.unique(val.y)) != {0, 1}:
        raise ValueError("Treino e validação devem ter ambas as classes.")
    cfg = {**config, "selecionar_epoca": selecionar_epoca, "sem_pesos": sem_pesos,
           "ids_treino": treino.nodule_ids.tolist(), "ids_validacao": val.nodule_ids.tolist()}
    cfg_path = pasta / "configuracao.json"
    if pasta.exists() and any(pasta.iterdir()) and not cfg_path.exists():
        raise ValueError(f"{pasta} já tem arquivos de outra execução. Preserve-os e use uma cópia limpa.")
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding="utf-8")) != cfg:
        raise ValueError(f"Configuração/ambiente mudou em {pasta}. Não misture experimentos.")
    resumo_path = pasta / "resumo.json"
    if resumo_path.exists():
        resumo = json.loads(resumo_path.read_text(encoding="utf-8"))
        for nome, digest in resumo["arquivos_sha256"].items():
            if not (pasta / nome).is_file() or sha256_arquivo(pasta / nome) != digest:
                raise ValueError(f"Resultado alterado ou incompleto: {pasta / nome}")
        print(f"Reutilizando execução concluída: {pasta}", flush=True)
        return resumo
    pasta.mkdir(parents=True, exist_ok=True)
    gravar_json(cfg_path, cfg)
    logger = logging.getLogger("sprint5." + str(pasta))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    arquivo = logging.FileHandler(pasta / "log_treino.txt", "w", encoding="utf-8")
    console = logging.StreamHandler(sys.stdout)
    for h in (arquivo, console):
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(h)
    try:
        logger.info("npz_sha256=%s", cfg["npz_sha256"])
        logger.info("manifesto_sha256=%s", cfg["manifesto_sha256"])
        logger.info("configuracao=%s", json.dumps({k: v for k, v in cfg.items()
                                                   if not k.startswith("ids_")}, ensure_ascii=False))
        logger.info("TREINO=%d VALIDACAO=%d; teste não carregado/avaliado", len(treino), len(val))
        fixar_seed(cfg["seed"])
        dispositivo = torch.device(cfg["ambiente"]["dispositivo"])
        pesos_calc = pesos_de_classe(treino.y, conferir_referencia=conferir_referencia)
        pesos = {0: 1., 1: 1.} if sem_pesos else pesos_calc
        logger.info("pesos=%s calculados_somente_no_treino=%s", pesos, pesos_calc)
        modelo = CNNRasa().to(dispositivo)
        otimizador = torch.optim.Adam(modelo.parameters(), lr=cfg["lr"])
        loader = DataLoader(treino, batch_size=cfg["batch"], shuffle=True,
                            generator=torch.Generator().manual_seed(cfg["seed"]), num_workers=0)
        w = torch.tensor([pesos[0], pesos[1]], dtype=torch.float32, device=dispositivo)
        melhor = (float("inf"), None, -1)
        for ep in range(1, cfg["epocas"] + 1):
            modelo.train()
            soma, n = 0., 0
            for x, y in loader:
                x, y = x.to(dispositivo), y.to(dispositivo)
                logit = modelo(x).squeeze(1)
                perda = (F.binary_cross_entropy_with_logits(logit, y, reduction="none")
                         * w[y.long()]).mean()
                if not torch.isfinite(perda):
                    raise ValueError("Perda de treino não finita.")
                otimizador.zero_grad()
                perda.backward()
                otimizador.step()
                soma, n = soma + perda.item() * len(y), n + len(y)
            if selecionar_epoca:
                pv = prever(modelo, val, dispositivo)
                perda_val = bce_ponderada(val.y, pv, pesos)
                auc_val = calcular_metricas(val.y, pv)["auc"]
                logger.info("epoca %02d perda_treino=%.6f perda_val=%.6f auc_val=%.6f",
                            ep, soma / n, perda_val, auc_val)
                if perda_val < melhor[0]:
                    melhor = (perda_val, {k: v.clone() for k, v in modelo.state_dict().items()}, ep)
            else:
                logger.info("epoca %02d perda_treino=%.6f; dobra externa não consultada", ep, soma / n)
        if selecionar_epoca:
            if melhor[1] is None:
                raise ValueError("Não houve checkpoint válido.")
            modelo.load_state_dict(melhor[1])
        epoca = melhor[2] if selecionar_epoca else cfg["epocas"]
        prob = prever(modelo, val, dispositivo)
        pred = tabela(val, "validacao", prob)
        pred.to_csv(pasta / "predicoes_validacao.csv", index=False)
        auc = calcular_metricas(val.y, prob)["auc"]
        resultado = {"seed": cfg["seed"], "dobra": cfg.get("dobra"), "auc": auc,
                     "epoca": epoca, "n_treino": len(treino), "n_validacao": len(val),
                     "pacientes_treino": len(set(treino.patient_ids)),
                     "pacientes_validacao": len(set(val.patient_ids)),
                     "benignos_treino": int((treino.y == 0).sum()),
                     "malignos_treino": int((treino.y == 1).sum()),
                     "benignos_validacao": int((val.y == 0).sum()),
                     "malignos_validacao": int((val.y == 1).sum()),
                     "npz_sha256": cfg["npz_sha256"]}
        if selecionar_epoca:
            resultado["threshold_youden"] = escolher_threshold_validacao(pred)["threshold"]
        logger.info("RESULTADO=%s", json.dumps(resultado, ensure_ascii=False))
    finally:
        for h in (arquivo, console):
            h.flush()
            h.close()
            logger.removeHandler(h)
    resultado["arquivos_sha256"] = {n: sha256_arquivo(pasta / n) for n in
                                     ("configuracao.json", "log_treino.txt", "predicoes_validacao.csv")}
    gravar_json(resumo_path, resultado)
    return resultado


def resumir(linhas, coluna, destino):
    df = pd.DataFrame([{k: v for k, v in l.items() if k != "arquivos_sha256"} for l in linhas])
    df.to_csv(destino.with_suffix(".csv"), index=False)
    vals = df.auc.to_numpy()
    stats = {"media": float(vals.mean()), "desvio_amostral": float(vals.std(ddof=1)),
             "minimo": float(vals.min()), "maximo": float(vals.max()), "n": len(vals)}
    gravar_json(destino.with_name(destino.name + "_resumo.json"), stats)
    texto = f"| {coluna} | AUC validação | Época | N treino | N validação | Pacientes validação |\n"
    texto += "|---|---:|---:|---:|---:|---:|\n"
    for l in linhas:
        texto += (f"| {l[coluna]} | {l['auc']:.6f} | {l['epoca']} | {l['n_treino']} | "
                  f"{l['n_validacao']} | {l['pacientes_validacao']} |\n")
    texto += f"| **Média** | **{stats['media']:.6f}** | — | — | — | — |\n"
    texto += f"| **Desvio-padrão amostral** | **{stats['desvio_amostral']:.6f}** | — | — | — | — |\n"
    texto += f"\nMínimo–máximo: {stats['minimo']:.6f}–{stats['maximo']:.6f}. Desvio com ddof=1; não é IC95%.\n"
    destino.with_suffix(".md").write_text(texto, encoding="utf-8")
    return texto, stats


def executar(npz, manifesto, raiz_resultados, etapa, epocas=EPOCAS, real=True, dispositivo="cpu",
             gerar_secao=False):
    ds, man, digest = carregar_desenvolvimento(npz, manifesto)
    it = np.flatnonzero(man.split.to_numpy() == "treino")
    iv = np.flatnonzero(man.split.to_numpy() == "validacao")
    if real and (len(it), len(iv)) != (522, 114):
        raise ValueError(f"Esperado 522/114; recebido {len(it)}/{len(iv)}. Conferir dados antes de continuar.")
    dobras, mapa = construir_dobras(ds, man)
    print(f"Desenvolvimento: {len(it)} treino + {len(iv)} validação = {len(ds)}.", flush=True)
    print(f"SHA-256 do NPZ: {digest}\nSementes: {SEMENTES}; CV: {N_DOBRAS} dobras por paciente.", flush=True)
    print(mapa.groupby("dobra_validacao").agg(nodulos=("nodule_id", "count"),
                                             pacientes=("patient_id", "nunique")).to_string(), flush=True)
    if etapa == "verificar":
        return
    raiz_resultados = Path(raiz_resultados)
    saida = raiz_resultados / "sprint5"
    ambiente = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
                "so": platform.platform(), "dispositivo": dispositivo,
                "threads_torch": torch.get_num_threads(),
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "hardware": torch.cuda.get_device_name(0) if dispositivo == "cuda" else platform.processor()}
    codigo = {p: sha256_arquivo(Path(__file__).parent / p) for p in
              ("experimentos_sprint5.py", "dataset.py", "treino_baseline.py", "avaliacao.py", "comparar_pesos.py")}
    comum = {"npz_sha256": digest, "manifesto_sha256": sha256_arquivo(manifesto),
             "codigo_sha256": codigo, "epocas": epocas, "lr": LR, "batch": BATCH,
             "augmentation": "flip3d_so_treino", "ambiente": ambiente,
             "sementes": list(SEMENTES), "n_dobras": N_DOBRAS, "seed_cv": SEED_CV,
             "cv_regra": "ultima_epoca_fixa_sem_selecao_na_dobra_externa", "dados_reais": real}
    protocolo_path = saida / "protocolo.json"
    if protocolo_path.exists() and json.loads(protocolo_path.read_text(encoding="utf-8")) != comum:
        raise ValueError("Protocolo mudou: não misture dados, código ou ambientes em sprint5/.")
    if saida.exists() and any(saida.iterdir()) and not protocolo_path.exists():
        raise ValueError("sprint5/ já existe sem protocolo; preserve o conteúdo anterior.")
    saida.mkdir(parents=True, exist_ok=True)
    gravar_json(protocolo_path, comum)
    man.to_csv(saida / "manifesto_desenvolvimento.csv", index=False)
    mapa.to_csv(saida / "dobras.csv", index=False)
    if etapa in ("tudo", "sementes"):
        linhas = []
        for seed in SEMENTES:
            pasta = (raiz_resultados / "real_com_pesos" if seed == 42
                     else saida / "sementes" / f"seed_{seed}")
            linhas.append(treinar_uma(subset(ds, it, seed), subset(ds, iv), pasta,
                                      {**comum, "seed": seed}, selecionar_epoca=True,
                                      conferir_referencia=real))
        resumir(linhas, "seed", saida / "tabela_sementes")
    if etapa in ("tudo", "cv"):
        linhas = []
        for dobra, (itr, iva) in enumerate(dobras, 1):
            linhas.append(treinar_uma(subset(ds, itr, SEED_CV), subset(ds, iva),
                                      saida / "cv" / f"dobra_{dobra}",
                                      {**comum, "seed": SEED_CV, "dobra": dobra},
                                      selecionar_epoca=False))
        resumir(linhas, "dobra", saida / "tabela_cv")
    if etapa in ("tudo", "pesos"):
        # A seed 42 ponderada é exatamente uma das cinco execuções, reutilizada.
        treinar_uma(subset(ds, it, 42), subset(ds, iv), raiz_resultados / "real_com_pesos",
                    {**comum, "seed": 42}, selecionar_epoca=True, conferir_referencia=real)
        treinar_uma(subset(ds, it, 42), subset(ds, iv), raiz_resultados / "real_sem_pesos",
                    {**comum, "seed": 42}, selecionar_epoca=True, sem_pesos=True,
                    conferir_referencia=real)
        comando = [sys.executable, "-m", "mineracao.comparar_pesos",
                   "--com-pesos", str(raiz_resultados / "real_com_pesos/predicoes_validacao.csv"),
                   "--sem-pesos", str(raiz_resultados / "real_sem_pesos/predicoes_validacao.csv"),
                   "--manifesto", str(saida / "manifesto_desenvolvimento.csv")]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        r = subprocess.run(comando, cwd=RAIZ, env=env, capture_output=True,
                           text=True, encoding="utf-8", check=True)
        (saida / "comparacao_pesos.txt").write_text(r.stdout, encoding="utf-8")
        print(r.stdout, flush=True)
    if sha256_arquivo(npz) != digest or sha256_arquivo(manifesto) != comum["manifesto_sha256"]:
        raise ValueError("Dados alterados durante a execução; resultados não devem ser usados.")
    if gerar_secao and all((saida / n).is_file() for n in ("tabela_sementes.md", "tabela_cv.md", "comparacao_pesos.txt")):
        ss = json.loads((saida / "tabela_sementes_resumo.json").read_text(encoding="utf-8"))
        cs = json.loads((saida / "tabela_cv_resumo.json").read_text(encoding="utf-8"))
        secao = "\n## Sprint 5 — Variabilidade do baseline (Issue #31)\n\n"
        secao += ("Protocolo fixado antes das execuções: sementes 42, 43, 44, 45 e 46; CNN rasa, "
                   f"Adam, lr={LR}, batch={BATCH}, {epocas} épocas, pesos calculados só no treino "
                   "e flip 3D só no treino. Mesmo split original nas cinco sementes. "
                   "Checkpoint de menor perda de validação, como no baseline original.\n\n"
                   f"SHA-256 do NPZ: `{digest}`. Ambiente e hashes do código em "
                   "`mineracao/resultados/sprint5/protocolo.json`.\n\n### Cinco sementes\n\n")
        secao += (saida / "tabela_sementes.md").read_text(encoding="utf-8")
        secao += ("\n### Validação cruzada por paciente\n\n"
                   "Cinco dobras sobre treino + validação, sem pacientes compartilhados entre "
                   "treino e validação de uma rodada. Partição estratificada por classe quando "
                   "compatível com os grupos; seed do particionamento e do treino igual a 42. "
                   "Pesos recalculados no treino de cada dobra. Cada paciente é avaliado em "
                   "uma única dobra. A guarda de `dataset.py` é reutilizada.\n\n"
                   f"Na CV, avalia-se a época final ({epocas}), prefixada: a dobra externa não "
                   "seleciona checkpoint nem threshold. Esta regra difere da seleção de época "
                   "das cinco sementes e impede escolher o checkpoint pela própria dobra avaliada.\n\n")
        secao += (saida / "tabela_cv.md").read_text(encoding="utf-8")
        secao += "\n### Com e sem pesos de classe\n\nMesma seed 42; comparação apenas na validação.\n\n```text\n"
        secao += (saida / "comparacao_pesos.txt").read_text(encoding="utf-8") + "```\n"
        secao += ("\n### Leitura dos resultados\n\n"
                   f"Entre as sementes, AUC média {ss['media']:.6f}, desvio amostral {ss['desvio_amostral']:.6f} "
                   f"e amplitude {ss['maximo'] - ss['minimo']:.6f}. "
                   f"Na CV, média {cs['media']:.6f}, desvio {cs['desvio_amostral']:.6f} "
                   f"e amplitude {cs['maximo'] - cs['minimo']:.6f}. "
                   "Essas dispersões são descritivas, não intervalos de confiança; as dobras "
                   "compartilham parte dos dados de treino.\n\n"
                   "O 0,9610 permanece o resultado histórico de teste de uma execução. "
                   "Os números acima são de validação: não estimam diretamente a variância "
                   "da AUC de teste, nem demonstram sozinhos que 0,9610 foi típico ou sorte. "
                   "A redação deve apresentar as dispersões observadas, sem transformar "
                   "0,9610 em média de cinco execuções. O teste não foi avaliado nesta issue, "
                   "e a pasta de comparação histórica não foi lida nem modificada pelo script.\n\n"
                   "A interpretação final também depende dos testes de sanidade da Issue #30; "
                   "se indicarem vazamento, investigar antes de validar estes resultados.\n")
        (saida / "secao_resultados.md").write_text(secao, encoding="utf-8")
        print(f"Seção para revisar e acrescentar à documentação: {saida / 'secao_resultados.md'}", flush=True)
    if not gerar_secao:
        print("Resultados guardados. Documentação e PR aguardam os merges das issues #30 e #34.",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--etapa", choices=("verificar", "sementes", "cv", "pesos", "tudo"), default="verificar")
    ap.add_argument("--npz", type=Path, default=RAIZ / "patches/patches_2_5d.npz")
    ap.add_argument("--manifesto", type=Path, default=RAIZ / "transformacao/manifest_patches.csv")
    ap.add_argument("--dispositivo", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--gerar-secao", action="store_true",
                    help="gerar rascunho só depois dos merges das issues #30 e #34")
    args = ap.parse_args()
    if args.dispositivo == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA indisponível neste ambiente; use CPU ou instale PyTorch com CUDA.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    executar(args.npz.resolve(), args.manifesto.resolve(), RAIZ / "mineracao/resultados",
             args.etapa, dispositivo=args.dispositivo, gerar_secao=args.gerar_secao)


if __name__ == "__main__":
    main()
