"""Issue #45: validação cruzada da ResNet18 nas dobras congeladas da Sprint 5.

Não carrega nem avalia o conjunto de teste. Reutiliza exatamente
mineracao/resultados/sprint5/dobras.csv e treina a ResNet18 com layer3+layer4
liberadas por 30 épocas fixas, sem consultar a dobra externa durante o treino.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mineracao.avaliacao import calcular_metricas
from mineracao.dataset import (
    AugmentacaoFlip3D,
    PatchesDataset,
    carregar_desenvolvimento,
    sha256_arquivo,
    validar_separacao_pacientes,
)
from mineracao.treino_resnet import (
    ResNet2_5D,
    fixar_seed,
    manter_batchnorm_em_avaliacao,
    prever,
    tabela_predicoes,
)

SEED = 42
N_DOBRAS = 5
EPOCAS = 30
BATCH = 32
LR_CABECA = 1e-3
LR_BACKBONE = 1e-4
DESCONGELAR = "layer3_layer4"
THRESHOLD_RELATORIO = 0.5

RAIZ = Path(__file__).resolve().parents[1]


def sha256_texto_lf(caminho: Path) -> str:
    """Hash de artefato textual independente da conversão CRLF/LF pelo Git."""
    return hashlib.sha256(caminho.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def gravar_json(caminho: Path, valor) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(valor, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def subset(ds, indices, seed=None):
    return PatchesDataset(
        ds.patches[indices],
        ds.y[indices],
        ds.nodule_ids[indices],
        ds.patient_ids[indices],
        None if seed is None else AugmentacaoFlip3D(seed),
    )


def pesos_de_classe(y):
    """Pesos calculados somente no treino da dobra, sem referência ao split original."""
    n = len(y)
    pesos = {}
    for classe in (0, 1):
        quantidade = int((y == classe).sum())
        if quantidade == 0:
            raise ValueError("Treino da dobra sem uma das classes.")
        pesos[classe] = n / (2 * quantidade)
    return pesos


def carregar_dobras_fixas(ds, man, caminho: Path):
    mapa = pd.read_csv(caminho)
    obrig = {"nodule_id", "patient_id", "dobra_validacao"}
    if obrig - set(mapa.columns):
        raise ValueError(f"dobras.csv sem colunas: {sorted(obrig - set(mapa.columns))}")
    if mapa.nodule_id.duplicated().any():
        raise ValueError("dobras.csv contém nodule_id duplicado.")

    base = man[["nodule_id", "patient_id"]].copy()
    alinhado = base.merge(
        mapa[list(obrig)],
        on="nodule_id",
        how="left",
        suffixes=("", "_dobras"),
        validate="one_to_one",
    )
    if alinhado.dobra_validacao.isna().any():
        raise ValueError("Há nódulos de desenvolvimento sem dobra congelada.")
    if not alinhado.patient_id.eq(alinhado.patient_id_dobras).all():
        raise ValueError("patient_id de dobras.csv diverge do manifesto.")
    if set(alinhado.dobra_validacao.astype(int)) != set(range(1, N_DOBRAS + 1)):
        raise ValueError("Esperava exatamente as dobras 1..5.")
    if alinhado.groupby("patient_id").dobra_validacao.nunique().gt(1).any():
        raise ValueError("Paciente aparece em mais de uma dobra de validação.")

    atribuicao = alinhado.dobra_validacao.astype(int).to_numpy()
    dobras = []
    for d in range(1, N_DOBRAS + 1):
        iv = np.flatnonzero(atribuicao == d)
        it = np.flatnonzero(atribuicao != d)
        if set(np.unique(ds.y[it])) != {0, 1} or set(np.unique(ds.y[iv])) != {0, 1}:
            raise ValueError(f"Dobra {d} sem as duas classes.")
        atual = man.copy()
        atual["split"] = "treino"
        atual.loc[iv, "split"] = "validacao"
        validar_separacao_pacientes(atual)
        dobras.append((it, iv))
    return dobras


def _config_ambiente(dispositivo: str):
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "so": platform.platform(),
        "dispositivo": dispositivo,
        "threads_torch": torch.get_num_threads(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "hardware": (
            torch.cuda.get_device_name(0)
            if dispositivo == "cuda"
            else platform.processor()
        ),
    }


def treinar_dobra(treino, val, pasta: Path, config: dict):
    cfg_path = pasta / "configuracao.json"
    resumo_path = pasta / "resumo.json"

    if pasta.exists() and any(pasta.iterdir()) and not cfg_path.exists():
        raise ValueError(
            f"{pasta} já contém arquivos sem configuracao.json; preserve e use pasta limpa."
        )
    if cfg_path.exists():
        anterior = json.loads(cfg_path.read_text(encoding="utf-8"))
        if anterior != config:
            raise ValueError(f"Configuração mudou em {pasta}; não misture execuções.")
    if resumo_path.exists():
        resumo = json.loads(resumo_path.read_text(encoding="utf-8"))
        normalizacao = resumo.get("normalizacao_arquivos_sha256")
        if normalizacao not in (None, "CRLF_para_LF"):
            raise ValueError(f"Normalização de hashes desconhecida em {resumo_path}.")
        verificar_hash = sha256_texto_lf if normalizacao else sha256_arquivo
        for nome, digest in resumo["arquivos_sha256"].items():
            arq = pasta / nome
            if not arq.is_file() or verificar_hash(arq) != digest:
                raise ValueError(f"Resultado alterado ou incompleto: {arq}")
        print(f"Reutilizando execução concluída: {pasta}", flush=True)
        return resumo

    pasta.mkdir(parents=True, exist_ok=True)
    gravar_json(cfg_path, config)

    logger = logging.getLogger(f"sprint6.resnet.dobra{config['dobra']}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    arquivo = logging.FileHandler(pasta / "log_treino.txt", "w", encoding="utf-8")
    console = logging.StreamHandler(sys.stdout)
    for h in (arquivo, console):
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(h)

    try:
        logger.info("issue=45 modelo=ResNet18 configuracao=%s", DESCONGELAR)
        logger.info("npz_sha256=%s", config["npz_sha256"])
        logger.info("dobras_sha256=%s", config["dobras_sha256"])
        logger.info(
            "dobra=%d seed=%d epocas=%d batch=%d lr_cabeca=%g lr_backbone=%g",
            config["dobra"],
            config["seed"],
            config["epocas"],
            config["batch"],
            config["lr_cabeca"],
            config["lr_backbone"],
        )
        logger.info(
            "TREINO=%d VALIDACAO=%d; teste não carregado/avaliado",
            len(treino),
            len(val),
        )
        logger.info(
            "regra_cv=epoca_final_fixa; dobra externa não seleciona checkpoint nem threshold"
        )

        fixar_seed(config["seed"])
        dispositivo = torch.device(config["ambiente"]["dispositivo"])
        pesos = pesos_de_classe(treino.y)
        logger.info("pesos_calculados_somente_no_treino=%s", pesos)

        modelo = ResNet2_5D(descongelar=DESCONGELAR).to(dispositivo)

        parametros_cabeca = list(modelo.rede.fc.parameters())
        parametros_backbone = [
            p
            for nome, p in modelo.rede.named_parameters()
            if p.requires_grad and not nome.startswith("fc.")
        ]
        grupos = [{"params": parametros_cabeca, "lr": config["lr_cabeca"]}]
        if parametros_backbone:
            grupos.append(
                {"params": parametros_backbone, "lr": config["lr_backbone"]}
            )
        otimizador = torch.optim.Adam(grupos)

        loader = DataLoader(
            treino,
            batch_size=config["batch"],
            shuffle=True,
            generator=torch.Generator().manual_seed(config["seed"]),
            num_workers=0,
        )
        w = torch.tensor(
            [pesos[0], pesos[1]], dtype=torch.float32, device=dispositivo
        )

        for epoca in range(1, config["epocas"] + 1):
            modelo.train()
            manter_batchnorm_em_avaliacao(modelo)
            soma, n = 0.0, 0
            for x, y in loader:
                x, y = x.to(dispositivo), y.to(dispositivo)
                logits = modelo(x).squeeze(1)
                perdas = F.binary_cross_entropy_with_logits(
                    logits, y, reduction="none"
                )
                perda = (perdas * w[y.long()]).mean()
                if not torch.isfinite(perda):
                    raise ValueError("Perda de treino não finita.")
                otimizador.zero_grad()
                perda.backward()
                otimizador.step()
                soma += perda.item() * len(y)
                n += len(y)
            logger.info(
                "epoca %02d perda_treino=%.6f; dobra externa não consultada",
                epoca,
                soma / n,
            )

        # A dobra externa só é aberta depois da época 30 fixa.
        probabilidades = prever(modelo, val, dispositivo)
        pred = tabela_predicoes(val, "validacao", probabilidades)
        pred.to_csv(pasta / "predicoes_validacao.csv", index=False)

        metricas = calcular_metricas(
            pred.y_true, pred.y_prob, threshold=THRESHOLD_RELATORIO
        )
        resultado = {
            "seed": config["seed"],
            "dobra": config["dobra"],
            "epoca": config["epocas"],
            "n_treino": len(treino),
            "n_validacao": len(val),
            "pacientes_treino": len(set(treino.patient_ids)),
            "pacientes_validacao": len(set(val.patient_ids)),
            "benignos_treino": int((treino.y == 0).sum()),
            "malignos_treino": int((treino.y == 1).sum()),
            "benignos_validacao": int((val.y == 0).sum()),
            "malignos_validacao": int((val.y == 1).sum()),
            "threshold_relatorio": THRESHOLD_RELATORIO,
            "metricas": metricas,
            "npz_sha256": config["npz_sha256"],
        }
        logger.info("RESULTADO=%s", json.dumps(resultado, ensure_ascii=False))
    finally:
        for h in (arquivo, console):
            h.flush()
            h.close()
            logger.removeHandler(h)

    # Os hashes originais dos bytes ficam registrados para rastrear a execução.
    # A guarda usa LF canônico, pois o Git pode converter as quebras de linha.
    resultado["arquivos_sha256_bytes_execucao"] = {
        nome: sha256_arquivo(pasta / nome)
        for nome in ("configuracao.json", "log_treino.txt", "predicoes_validacao.csv")
    }
    resultado["normalizacao_arquivos_sha256"] = "CRLF_para_LF"
    resultado["arquivos_sha256"] = {
        nome: sha256_texto_lf(pasta / nome)
        for nome in resultado["arquivos_sha256_bytes_execucao"]
    }
    gravar_json(resumo_path, resultado)
    return resultado


def executar(npz: Path, manifesto: Path, dobras_csv: Path, saida: Path, etapa: str, dispositivo: str):
    protocolo_s5_path = RAIZ / "mineracao/resultados/sprint5/protocolo.json"
    if not protocolo_s5_path.is_file():
        raise FileNotFoundError("protocolo.json da Sprint 5 não encontrado.")
    protocolo_s5 = json.loads(protocolo_s5_path.read_text(encoding="utf-8"))

    ds, man, digest = carregar_desenvolvimento(npz, manifesto)
    if digest != protocolo_s5["npz_sha256"]:
        raise ValueError(
            "O NPZ não é o mesmo da Sprint 5. Não execute a CV com dados diferentes."
        )
    if protocolo_s5["n_dobras"] != N_DOBRAS or protocolo_s5["seed_cv"] != SEED:
        raise ValueError("Protocolo da Sprint 5 não corresponde a 5 dobras/seed 42.")
    if protocolo_s5["epocas"] != EPOCAS:
        raise ValueError("A regra de época da Sprint 5 mudou; revisar antes de continuar.")

    dobras = carregar_dobras_fixas(ds, man, dobras_csv)
    print(
        f"Desenvolvimento carregado: {len(ds)} nódulos. "
        f"NPZ={digest}. Dobras congeladas válidas.",
        flush=True,
    )

    if etapa == "verificar":
        for d, (it, iv) in enumerate(dobras, 1):
            print(
                f"Dobra {d}: treino={len(it)}, validação={len(iv)}, "
                f"pacientes_val={len(set(ds.patient_ids[iv]))}",
                flush=True,
            )
        print("Verificação concluída; conjunto de teste não foi carregado.", flush=True)
        return

    ambiente = _config_ambiente(dispositivo)
    codigo = {
        nome: sha256_arquivo(RAIZ / "mineracao" / nome)
        for nome in (
            "experimentos_sprint6.py",
            "dataset.py",
            "treino_resnet.py",
            "avaliacao.py",
        )
    }
    comum = {
        "issue": 45,
        "modelo": "ResNet18",
        "descongelar": DESCONGELAR,
        "seed": SEED,
        "n_dobras": N_DOBRAS,
        "epocas": EPOCAS,
        "batch": BATCH,
        "lr_cabeca": LR_CABECA,
        "lr_backbone": LR_BACKBONE,
        "augmentation": "flip3d_so_treino",
        "normalizacao": "ImageNet",
        "threshold_relatorio": THRESHOLD_RELATORIO,
        "regra_cv": "epoca_final_30_fixa_sem_consultar_dobra_externa",
        "npz_sha256": digest,
        "manifesto_sha256": sha256_arquivo(manifesto),
        "dobras_sha256": sha256_arquivo(dobras_csv),
        "codigo_sha256": codigo,
        "ambiente": ambiente,
    }

    saida.mkdir(parents=True, exist_ok=True)
    protocolo = saida.parent.parent / "protocolo.json"
    if protocolo.exists():
        anterior = json.loads(protocolo.read_text(encoding="utf-8"))
        if anterior != comum:
            raise ValueError("Protocolo da Sprint 6 mudou; não misture resultados.")
    else:
        gravar_json(protocolo, comum)

    for d, (it, iv) in enumerate(dobras, 1):
        treino = subset(ds, it, SEED)
        val = subset(ds, iv)
        cfg = {
            **comum,
            "dobra": d,
            "ids_treino": treino.nodule_ids.tolist(),
            "ids_validacao": val.nodule_ids.tolist(),
        }
        treinar_dobra(treino, val, saida / f"dobra_{d}", cfg)

    if sha256_arquivo(npz) != digest:
        raise ValueError("NPZ foi alterado durante a execução.")
    print(
        "Cinco dobras da ResNet18 concluídas. Nenhum dado de teste foi usado.",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--etapa", choices=("verificar", "resnet"), default="verificar"
    )
    ap.add_argument("--npz", type=Path, default=RAIZ / "patches/patches_2_5d.npz")
    ap.add_argument(
        "--manifesto",
        type=Path,
        default=RAIZ / "transformacao/manifest_patches.csv",
    )
    ap.add_argument(
        "--dobras",
        type=Path,
        default=RAIZ / "mineracao/resultados/sprint5/dobras.csv",
    )
    ap.add_argument(
        "--saida",
        type=Path,
        default=RAIZ / "mineracao/resultados/sprint6/resnet18/cv",
    )
    ap.add_argument("--dispositivo", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()

    if args.dispositivo == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA indisponível; use --dispositivo cpu.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    executar(
        args.npz.resolve(),
        args.manifesto.resolve(),
        args.dobras.resolve(),
        args.saida.resolve(),
        args.etapa,
        args.dispositivo,
    )


if __name__ == "__main__":
    main()
