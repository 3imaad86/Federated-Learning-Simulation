"""Modele (Net) + donnees CIFAR-10 (IID ou NON-IID Dirichlet) + partitionnement.

Le mode est controle par les params `partitioning` et `alpha` :
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


# ============================================================================
# 1) Constantes + transforms + chemin disque
# ============================================================================

DATA_ROOT = Path(os.environ.get("FL_DATA_ROOT", str(Path.home() / ".flwr_data")))
SEED = int(os.environ.get("FL_SEED", "42"))
VAL_RATIO = 0.2

_TRANSFORMS = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def set_seed(seed):
    """Force la reproducibilite : meme init modele + meme ordre batches.

    A appeler au DEBUT de chaque process (server, chaque client) pour eviter
    les soucis de propagation d'env vars dans Ray.
    """
    if seed is None or int(seed) < 0:
        return
    import random
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# 2) Modele : petit CNN (3 conv + 1 linear, ~23k params)
# ============================================================================

class Net(nn.Module):
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


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def model_size_bytes():
    """Taille (octets) des parametres -> volume echange par client."""
    return sum(p.numel() * p.element_size() for p in Net().parameters())


# ============================================================================
# 3) Datasets CIFAR-10 (cache global)
# ============================================================================

_trainset = None
_testset = None


def get_trainset():
    global _trainset
    if _trainset is None:
        _trainset = CIFAR10(root=str(DATA_ROOT), train=True, download=True,
                            transform=_TRANSFORMS)
    return _trainset


def get_testset():
    """Test set CIFAR-10 officiel (10k images equilibrees, jamais vu en train)."""
    global _testset
    if _testset is None:
        _testset = CIFAR10(root=str(DATA_ROOT), train=False, download=True,
                           transform=_TRANSFORMS)
    return _testset


# ============================================================================
# 4) Partitionnement IID / Dirichlet (avec cache)
# ============================================================================

_parts_cache = {}


def _build_iid(num_partitions, seed):
    idx = np.arange(len(get_trainset()))
    np.random.default_rng(seed).shuffle(idx)
    return [p.tolist() for p in np.array_split(idx, num_partitions)]


def _build_dirichlet(num_partitions, alpha, seed):
    """Dirichlet par classe : pour chaque label, repartit ses indices entre clients
    selon proportions ~ Dirichlet(alpha)."""
    targets = np.asarray(get_trainset().targets)
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

    # Garantit que chaque client a au moins 1 exemple (donne par le plus gros)
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
        raise ValueError(f"partitioning={partitioning!r} invalide (attendu: 'iid'|'noniid')")
    if float(alpha) <= 0.0:
        raise ValueError("dirichlet-alpha doit etre > 0 pour partitionnement noniid")
    return _build_dirichlet(num_partitions, float(alpha), seed)


def get_partitions(num_partitions, partitioning="noniid", alpha=0.3, seed=SEED):
    """Memoize build_partitions (les partitions sont stables pour un (n, mode, alpha, seed) donne)."""
    key = (int(num_partitions), str(partitioning).lower(), float(alpha), int(seed))
    if key not in _parts_cache:
        _parts_cache[key] = build_partitions(num_partitions, partitioning, alpha, seed)
    return _parts_cache[key]


def partition_sizes(num_partitions, partitioning="noniid", alpha=0.3):
    return [len(p) for p in get_partitions(num_partitions, partitioning, alpha)]


# ============================================================================
# 5) Loader par client (train + val)
# ============================================================================

def _split_train_val(idx_list):
    """Coupe une liste d'indices en (train, val) selon VAL_RATIO."""
    n = len(idx_list)
    if n == 0:
        return [], []
    if n == 1:
        return idx_list, idx_list
    val_size = min(max(1, int(n * VAL_RATIO)), n - 1)
    return idx_list[:-val_size], idx_list[-val_size:]


def _apply_data_hetero(tr_idx, pid, num_partitions):
    """Tronque le train du client a une fraction tiree au hasard dans [0.2, 1.0].

    L'ancienne formule `keep = 0.2 + 0.8 * pid/(N-1)` etait MONOTONE en pid :
    le client 0 avait toujours 20% des donnees, le client N-1 toujours 100%.
    Combine avec `tier = pid % 3` (heterogeneite compute) cela creait des
    correlations parasites entre tier-compute, tier-reseau et taille des
    donnees, empechant d'isoler l'effet de chaque facteur.

    Ici on tire `keep` au hasard via un RNG seede par pid (reproductible mais
    non-monotone). L'offset 7919 (un nombre premier) decorrele ce RNG des
    autres RNG seeds par pid (loaders, network profile, etc.).
    """
    rng = np.random.default_rng(SEED + pid + 7919)
    keep = float(rng.uniform(0.2, 1.0))
    n_keep = max(1, int(len(tr_idx) * keep))
    return tr_idx[:n_keep]


def _make_loader_generator(pid, seed=-1):
    """Generator torch seede pour rendre le shuffle DataLoader deterministe.

    seed < 0 -> None (comportement non-deterministe par defaut).
    """
    if seed is None or int(seed) < 0:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed) + int(pid))
    return g


def load_data(pid, num_partitions, batch_size, data_hetero=0,
              partitioning="noniid", alpha=0.3, seed=-1):
    """Retourne (trainloader, valloader) pour le client `pid`.

    Si data_hetero=1, train est tronque selon `keep(pid)` pour simuler des
    tailles differentes. Le val reste complet pour comparabilite.
    Si seed >= 0, le DataLoader shuffle devient deterministe.
    """
    ds = get_trainset()
    idx = np.array(get_partitions(num_partitions, partitioning, alpha)[pid])
    np.random.default_rng(SEED + pid).shuffle(idx)
    tr, va = _split_train_val(idx.tolist())

    tr_full = list(tr)
    if int(data_hetero):
        tr = _apply_data_hetero(tr, pid, num_partitions)
    if not tr:
        tr = tr_full[:1] or va[:1]

    gen = _make_loader_generator(pid, seed)
    return (
        DataLoader(Subset(ds, tr), batch_size=batch_size,
                   shuffle=bool(tr), generator=gen),
        DataLoader(Subset(ds, va), batch_size=batch_size, shuffle=False),
    )
