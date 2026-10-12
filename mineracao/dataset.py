"""Loader do baseline: .npz de patches + manifesto -> treino/validação/teste.

O split vem SEMPRE da coluna `split` do manifesto. Este módulo nunca sorteia
nem recalcula divisão. Augmentation só é aceita para o treino.

Só depende de numpy e pandas; PatchesDataset funciona como Dataset do PyTorch
(__len__/__getitem__) sem importar torch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("treino", "validacao", "teste")
ROTULOS = {"benigno": 0, "maligno": 1}
CANAIS_ESPERADOS = ("axial", "coronal", "sagital")
FORMA_PATCH = (3, 64, 64)


def sha256_arquivo(caminho: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while parte := f.read(bloco):
            h.update(parte)
    return h.hexdigest()


class PatchesDataset:
    """Um split. `transform` recebe e devolve array float32 (3, 64, 64)."""

    def __init__(self, patches, y, nodule_ids, patient_ids, transform=None):
        self.patches, self.y = patches, y
        self.nodule_ids, self.patient_ids = nodule_ids, patient_ids
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.patches[i].astype(np.float32)  # float16 em disco -> float32
        if self.transform is not None:
            x = self.transform(x)
        return x, np.float32(self.y[i])


def carregar_particoes(caminho_npz, caminho_manifesto, transform_treino=None):
    """Devolve ({'treino','validacao','teste'} -> PatchesDataset, sha256_do_npz).

    Alinha por nodule_id (não pela posição) e valida tudo antes de devolver.
    transform_treino é aplicada só ao treino; validação e teste nunca recebem.
    """
    caminho_npz = Path(caminho_npz)
    man = pd.read_csv(caminho_manifesto)
    for col in ("nodule_id", "patient_id", "split", "label"):
        if man[col].isna().any():
            raise ValueError(f"Manifesto com {col} ausente.")
    if man.nodule_id.duplicated().any():
        raise ValueError("Manifesto com nodule_id duplicado.")
    if not man.split.isin(SPLITS).all():
        raise ValueError(f"split fora de {SPLITS}.")
    if not man.label.isin(ROTULOS).all():
        raise ValueError("label fora de benigno/maligno.")
    if man.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("Paciente em mais de um split no manifesto.")

    with np.load(caminho_npz, allow_pickle=False) as z:
        patches, ids = z["patches"], z["nodule_id"].astype(str)
        canais = tuple(z["canais"].astype(str)) if "canais" in z.files else None
    if canais is not None and canais != CANAIS_ESPERADOS:
        raise ValueError(f"Ordem de canais {canais} != {CANAIS_ESPERADOS}.")
    if patches.shape[1:] != FORMA_PATCH:
        raise ValueError(f"Forma {patches.shape[1:]} != {FORMA_PATCH}.")
    if len(set(ids)) != len(ids) or set(ids) != set(man.nodule_id):
        raise ValueError("nodule_id do .npz e do manifesto não coincidem.")
    if not np.isfinite(patches.astype(np.float32)).all():
        raise ValueError("Patches com NaN/inf.")

    linha = {n: i for i, n in enumerate(ids)}
    particoes = {}
    for nome in SPLITS:
        parte = man[man.split == nome]
        sel = [linha[n] for n in parte.nodule_id]
        particoes[nome] = PatchesDataset(
            patches[sel],
            parte.label.map(ROTULOS).to_numpy(),
            parte.nodule_id.to_numpy(),
            parte.patient_id.to_numpy(),
            transform=transform_treino if nome == "treino" else None,
        )
    return particoes, sha256_arquivo(caminho_npz)


class AugmentacaoFlip3D:
    """Espelhamentos coerentes com o cubo 3D. Só para o TREINO.

    Convenção do extrai_patches.py: cubo com eixos (i, j, k);
      canal 0 axial   = cubo[:, :, c]  -> eixos (i, j)
      canal 1 coronal = cubo[c, :, :]  -> eixos (j, k)
      canal 2 sagital = cubo[:, c, :]  -> eixos (i, k)
    Espelhar o cubo num eixo = espelhar, em cada canal, o eixo correspondente
    (o canal que não contém o eixo fica igual). Cada eixo é sorteado com p=0,5.
    Não há rotação livre, translação nem interpolação: o resultado é sempre o
    patch que se obteria extraindo de um cubo espelhado. Detalhe: o plano
    central é o índice 32 de 64, então o espelho cai a meio voxel do centro.
    """

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        if self.rng.random() < 0.5:  # espelha i
            x[0], x[2] = x[0][::-1, :], x[2][::-1, :]
        if self.rng.random() < 0.5:  # espelha j
            x[0], x[1] = x[0][:, ::-1], x[1][::-1, :]
        if self.rng.random() < 0.5:  # espelha k
            x[1], x[2] = x[1][:, ::-1], x[2][:, ::-1]
        return np.ascontiguousarray(x)
