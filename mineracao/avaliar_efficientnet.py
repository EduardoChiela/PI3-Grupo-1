"""Avaliação final única do EfficientNet-B0 escolhido na validação (Issue #44)."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision

from mineracao.avaliacao import (
    avaliar_com_threshold,
    bootstrap_por_paciente,
    escolher_threshold_validacao,
    validar_com_manifesto,
)
from mineracao.dataset import carregar_particoes
from mineracao.pontos_operacao import bootstrap_ponto, limiar_max_f1_validacao
from mineracao.treino_efficientnet import EfficientNetB0_2_5D, prever, tabela_predicoes


RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "mineracao/resultados/efficientnet_s6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=RAIZ / "patches/patches_2_5d.npz")
    parser.add_argument("--manifesto", type=Path, default=RAIZ / "transformacao/manifest_patches.csv")
    parser.add_argument("--escolha", type=Path, default=BASE / "escolha_validacao.json")
    parser.add_argument("--confirmar-teste-final", action="store_true",
                        help="Confirma que a seleção está congelada e autoriza a única avaliação do teste.")
    args = parser.parse_args()
    if not args.confirmar_teste_final:
        raise SystemExit("Revise escolha_validacao.json e rode novamente com --confirmar-teste-final.")

    escolha = json.loads(args.escolha.read_text(encoding="utf-8"))
    if escolha.get("teste_utilizado_na_escolha") is not False:
        raise ValueError("A escolha não confirma que o teste foi excluído da seleção.")
    selecionada = escolha["configuracao_escolhida"]
    configuracao = selecionada["nome"]
    pasta = BASE / configuracao
    checkpoint = pasta / "modelo_efficientnet.pt"
    predicoes_val_path = pasta / "predicoes_validacao.csv"
    marker = BASE / "avaliacao_teste_iniciada.json"
    if marker.exists():
        raise FileExistsError(f"Avaliação final já iniciada; não execute o teste de novo: {marker}")
    if not checkpoint.is_file() or not predicoes_val_path.is_file():
        raise FileNotFoundError("Faltam checkpoint ou predições de validação da configuração escolhida.")

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    particoes, sha = carregar_particoes(args.npz, args.manifesto)
    if sha != escolha["npz_sha256"]:
        raise ValueError("O hash do NPZ não corresponde ao usado na seleção da configuração.")
    manifesto = pd.read_csv(args.manifesto)
    validacao, teste = particoes["validacao"], particoes["teste"]
    modelo = EfficientNetB0_2_5D(configuracao).to(dispositivo)
    modelo.load_state_dict(torch.load(checkpoint, map_location=dispositivo, weights_only=True))

    # Reconfere que o CSV de validação pertence ao checkpoint escolhido.
    val_salva = validar_com_manifesto(pd.read_csv(predicoes_val_path), manifesto)
    val_atual = validar_com_manifesto(
        tabela_predicoes(validacao, "validacao", prever(modelo, validacao, dispositivo)), manifesto
    )
    colunas = ["nodule_id", "patient_id", "y_true", "y_prob"]
    salva_ord = val_salva[colunas].sort_values("nodule_id").reset_index(drop=True)
    atual_ord = val_atual[colunas].sort_values("nodule_id").reset_index(drop=True)
    if not salva_ord[["nodule_id", "patient_id", "y_true"]].equals(
        atual_ord[["nodule_id", "patient_id", "y_true"]]
    ) or not np.allclose(salva_ord.y_prob, atual_ord.y_prob, rtol=1e-6, atol=1e-7):
        raise ValueError("O CSV de validação não corresponde ao checkpoint salvo; teste não foi acessado.")

    threshold_youden = escolher_threshold_validacao(val_salva)["threshold"]
    escolha_f1 = limiar_max_f1_validacao(val_salva)
    threshold_f1 = escolha_f1["threshold"]

    # O marcador exclusivo é gravado antes da primeira leitura do split de teste.
    with marker.open("x", encoding="utf-8") as arquivo:
        json.dump({"configuracao": configuracao, "npz_sha256": sha,
                   "checkpoint": str(checkpoint.relative_to(RAIZ))}, arquivo, indent=2)

    df_teste = validar_com_manifesto(
        tabela_predicoes(teste, "teste", prever(modelo, teste, dispositivo)), manifesto
    )
    resultado = {
        "arquitetura": "EfficientNet-B0 ImageNet1K_V1",
        "ambiente_avaliacao": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "dispositivo": str(dispositivo),
        },
        "configuracao": configuracao,
        "npz_sha256": sha,
        "teste_utilizado_na_escolha": False,
        "threshold_youden_validacao": threshold_youden,
        "metricas_youden_teste": avaliar_com_threshold(df_teste, threshold_youden),
        "ic95_youden_bootstrap_paciente": bootstrap_por_paciente(df_teste, threshold_youden, 2000, 42),
        "threshold_max_f1_validacao": threshold_f1,
        "f1_maximo_validacao": escolha_f1["f1"],
        "metricas_max_f1_teste": avaliar_com_threshold(df_teste, threshold_f1),
        "ic95_max_f1_bootstrap_paciente": bootstrap_por_paciente(df_teste, threshold_f1, 2000, 42),
        "sensibilidade_especificidade_minima_090": bootstrap_ponto(
            df_teste, "sensibilidade", 0.90, 2000, 42
        ),
        "especificidade_sensibilidade_minima_085": bootstrap_ponto(
            df_teste, "especificidade", 0.85, 2000, 42
        ),
        "n_teste": int(len(df_teste)),
        "n_pacientes_teste": int(df_teste.patient_id.nunique()),
        "bootstrap": {"repeticoes": 2000, "seed": 42, "unidade": "patient_id"},
    }
    df_teste.to_csv(BASE / "predicoes_teste.csv", index=False)
    (BASE / "avaliacao_final.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(json.dumps(resultado, indent=2, ensure_ascii=False, default=float))
    print(f"Resultados gravados em {BASE}")


if __name__ == "__main__":
    main()
