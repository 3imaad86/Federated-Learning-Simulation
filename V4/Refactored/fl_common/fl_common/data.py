"""Modele (Net) + donnees CIFAR-10 (IID ou NON-IID Dirichlet) + partitionnement.

Le mode de partitionnement est controle par les parametres `partitioning`
et `alpha` passes a `load_data(...)` / `partition_sizes(...)`.

Valeurs possibles :
  * partitioning = "iid"    -> indices melanges puis coupes en N parts egales
  * partitioning = "noniid" -> Dirichlet(alpha) (alpha petit = plus non-IID)
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

DATA_ROOT = Path(
    os.environ.get("FL_DATA_ROOT", str(Path.home() / ".flwr_data"))
)
VAL_RATIO = 0.2
SEED = int(os.environ.get("FL_SEED", "42"))


class Net(nn.Module):
    """Petit CNN (identique a la base quickstart)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
_trainset = None
_testset = None
_parts_cache = {}


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_trainset():
    global _trainset
    if _trainset is None:
        _trainset = CIFAR10(root=str(DATA_ROOT), train=True, download=True, transform=_transforms)
    return _trainset


def get_testset():
    """Test set CIFAR-10 officiel (10k images equilibrees, jamais vu en train)."""
    global _testset
    if _testset is None:
        _testset = CIFAR10(root=str(DATA_ROOT), train=False, download=True, transform=_transforms)
    return _testset


def _build_iid(num_partitions, seed):
    ds = get_trainset()
    idx = np.arange(len(ds))
    np.random.default_rng(seed).shuffle(idx)
    return [p.tolist() for p in np.array_split(idx, num_partitions)]


def _build_dirichlet(num_partitions, alpha, seed):
    ds = get_trainset()
    targets = np.asarray(ds.targets)
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(num_partitions)]
    for label in np.unique(targets):
        label_idx = np.where(targets == label)[0]
        rng.shuffle(label_idx)
        proportions = rng.dirichlet([alpha] * num_partitions)
        counts = rng.multinomial(len(label_idx), proportions)
        start = 0
        for pid, c in enumerate(counts):
            parts[pid].extend(label_idx[start:start + c].tolist())
            start += c
    for pid in range(num_partitions):
        if not parts[pid]:
            donor = max(range(num_partitions), key=lambda i: len(parts[i]))
            if len(parts[donor]) > 1:
                parts[pid].append(parts[donor].pop())
    return parts


def build_partitions(num_partitions, partitioning="noniid", alpha=0.3, seed=SEED):
    mode = str(partitioning).lower()
    if mode == "iid":
        return _build_iid(num_partitions, seed)
    if mode != "noniid":
        raise ValueError(
            f"partitioning={partitioning!r} invalide; valeurs attendues: 'iid' ou 'noniid'"
        )
    if float(alpha) <= 0.0:
        raise ValueError("dirichlet-alpha doit etre > 0 pour un partitionnement noniid")
    return _build_dirichlet(num_partitions, float(alpha), seed)


def get_partitions(num_partitions, partitioning="noniid", alpha=0.3, seed=SEED):
    key = (int(num_partitions), str(partitioning).lower(), float(alpha), int(seed))
    if key not in _parts_cache:
        _parts_cache[key] = build_partitions(num_partitions, partitioning, alpha, seed)
    return _parts_cache[key]


def partition_sizes(num_partitions, partitioning="noniid", alpha=0.3):
    return [len(p) for p in get_partitions(num_partitions, partitioning, alpha)]


def load_data(pid, num_partitions, batch_size, data_hetero=0,
              partitioning="noniid", alpha=0.3):
    """Charge (trainloader, valloader) pour le client `pid`.

    Si data_hetero = 1, tronque le TRAIN selon une fraction qui depend du pid
    (keep = 0.2..1.0) pour simuler des tailles differentes entre clients.
    Le VAL reste complet pour que l'evaluation soit comparable.
    """
    ds = get_trainset()
    idx = np.array(get_partitions(num_partitions, partitioning, alpha)[pid])
    np.random.default_rng(SEED + pid).shuffle(idx)
    if len(idx) == 0:
        tr, va = [], []
    elif len(idx) == 1:
        tr = idx.tolist()
        va = idx.tolist()
    else:
        val_size = min(max(1, int(len(idx) * VAL_RATIO)), len(idx) - 1)
        tr, va = idx[:-val_size].tolist(), idx[-val_size:].tolist()
    tr_full = list(tr)
    if int(data_hetero):
        keep = 0.2 + 0.8 * (pid / max(1, num_partitions - 1))
        n_keep = max(1, int(len(tr) * keep))
        tr = tr[:n_keep]
    if not tr:
        tr = tr_full[:1] or va[:1]
    return (
        DataLoader(Subset(ds, tr), batch_size=batch_size, shuffle=bool(tr)),
        DataLoader(Subset(ds, va), batch_size=batch_size, shuffle=False),
    )


def model_size_bytes():
    """Taille (octets) des parametres -> estimation du volume echange par client."""
    return sum(p.numel() * p.element_size() for p in Net().parameters())
