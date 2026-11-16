"""Loader do baseline: .npz de patches + manifesto -> treino/validação/teste.

O split vem SEMPRE da coluna `split` do manifesto. Este módulo nunca sorteia
nem recalcula divisão. Augmentation só é aceita para o treino.

Só depende de numpy e pandas; PatchesDataset funciona como Dataset do PyTorch
(__len__/__getitem__) sem importar torch.
"""
from __future__ import annotations

import hashlib
import csv
from pathlib import Path
from zipfile import ZipFile

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


def validar_separacao_pacientes(man):
    """Guarda comum ao split original e a cada dobra da Sprint 5."""
    if man.groupby("patient_id").split.nunique().gt(1).any():
        raise ValueError("Paciente em mais de um split no manifesto.")


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
    validar_separacao_pacientes(man)

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


def carregar_desenvolvimento(caminho_npz, caminho_manifesto):
    """Só treino/validação; nunca constrói Dataset nem lê rótulos do teste.

    Identificadores e splits completos servem exclusivamente para conferir o
    alinhamento e a separação de pacientes. As imagens selecionadas são lidas
    diretamente de patches.npy, dentro do ZIP, sem z['patches'] (que carregaria
    tudo). seek pode descompactar bytes ignorados; SHA-256 lê o arquivo inteiro
    como bytes. Nenhum pixel/label de teste é interpretado ou usado pelo modelo.
    """
    with open(caminho_manifesto, newline="", encoding="utf-8-sig") as f:
        leitor = csv.DictReader(f)
        colunas = {"nodule_id", "patient_id", "split", "label"}
        if not colunas.issubset(leitor.fieldnames or []):
            raise ValueError("Manifesto sem colunas obrigatórias.")
        metadados, desenvolvimento = [], []
        for linha in leitor:
            meta = {k: linha[k] for k in ("nodule_id", "patient_id", "split")}
            metadados.append(meta)
            if meta["split"] in ("treino", "validacao"):
                desenvolvimento.append({**meta, "label": linha["label"]})
    meta = pd.DataFrame(metadados)
    if meta.empty or meta.isna().any().any() or meta.eq("").any().any():
        raise ValueError("Identificação ausente no manifesto.")
    if meta.nodule_id.duplicated().any() or not meta.split.isin(SPLITS).all():
        raise ValueError("IDs duplicados ou split inválido.")
    validar_separacao_pacientes(meta)
    man = pd.DataFrame(desenvolvimento)
    if man.empty or not man.label.isin(ROTULOS).all():
        raise ValueError("Desenvolvimento vazio ou com rótulo inválido.")

    with np.load(caminho_npz, allow_pickle=False) as z:
        ids = z["nodule_id"].astype(str)
        canais = tuple(z["canais"].astype(str))
    if ids.ndim != 1 or len(set(ids)) != len(ids) or set(ids) != set(meta.nodule_id):
        raise ValueError("IDs do NPZ e manifesto não coincidem.")
    if canais != CANAIS_ESPERADOS:
        raise ValueError(f"Ordem de canais inválida: {canais}.")
    posicoes = {n: i for i, n in enumerate(ids)}
    selecao = sorted((posicoes[n], j) for j, n in enumerate(man.nodule_id))
    with ZipFile(caminho_npz) as z, z.open("patches.npy") as f:
        versao = np.lib.format.read_magic(f)
        leitores = {(1, 0): np.lib.format.read_array_header_1_0,
                    (2, 0): np.lib.format.read_array_header_2_0}
        if versao not in leitores:
            raise ValueError(f"Cabeçalho NPY não suportado: {versao}.")
        forma, fortran, dtype = leitores[versao](f)
        if forma != (len(ids), *FORMA_PATCH) or fortran or dtype.kind != "f":
            raise ValueError("Esperado patches float em ordem C, (N,3,64,64).")
        inicio, tamanho = f.tell(), int(np.prod(FORMA_PATCH)) * dtype.itemsize
        patches = np.empty((len(man), *FORMA_PATCH), dtype=dtype)
        for origem, destino in selecao:
            f.seek(inicio + origem * tamanho)
            dados = f.read(tamanho)
            if len(dados) != tamanho:
                raise ValueError("NPZ truncado.")
            patches[destino] = np.frombuffer(dados, dtype=dtype).reshape(FORMA_PATCH)
    if not np.isfinite(patches).all() or (patches < 0).any() or (patches > 1).any():
        raise ValueError("Patches de desenvolvimento fora de [0,1] ou não finitos.")
    ds = PatchesDataset(patches, man.label.map(ROTULOS).to_numpy(),
                        man.nodule_id.to_numpy(), man.patient_id.to_numpy())
    return ds, man, sha256_arquivo(Path(caminho_npz))


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
