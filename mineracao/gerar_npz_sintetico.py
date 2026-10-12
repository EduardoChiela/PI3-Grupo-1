"""Gera um .npz SINTÉTICO com o mesmo formato do patches_2_5d.npz real.

Serve só para escrever e testar loader, treino, métricas e ROC antes da #11
fechar. Imagens = ruído uniforme [0, 1] SEM relação com o rótulo; os
nodule_id são os reais, lidos do manifesto, na ordem de `idx`.

Consequência útil: um modelo treinado nele deve dar AUC ~ 0,5. AUC claramente
acima disso significa vazamento ou bug no pipeline.

Versões testadas: Python 3.12, numpy 2.4, pandas 3.0.
Uso (na raiz do repositório):
    python mineracao/gerar_npz_sintetico.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
FORMA_PATCH = (3, 64, 64)
CANAIS = ("axial", "coronal", "sagital")  # mesma ordem do extrai_patches.py
N_ESPERADO = 747


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifesto", type=Path,
                    default=Path("transformacao/manifest_patches.csv"))
    ap.add_argument("--saida", type=Path,
                    default=Path("sintetico/patches_sintetico.npz"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    manifesto = pd.read_csv(args.manifesto).sort_values("idx")
    if manifesto.nodule_id.duplicated().any():
        raise SystemExit("Manifesto com nodule_id duplicado.")
    if len(manifesto) != N_ESPERADO:
        raise SystemExit(f"Esperava {N_ESPERADO} nódulos, o manifesto tem {len(manifesto)}.")

    rng = np.random.default_rng(args.seed)
    patches = rng.random((len(manifesto), *FORMA_PATCH)).astype(np.float16)

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.saida,
        patches=patches,
        nodule_id=manifesto.nodule_id.to_numpy().astype(str),
        canais=np.array(CANAIS),
    )
    print(f"[OK] {args.saida}  patches={patches.shape} {patches.dtype}  "
          f"min={patches.min():.3f} max={patches.max():.3f}")


if __name__ == "__main__":
    main()
