"""Inspecte les partitions Dirichlet : distribution des classes par client.

Usage :
    conda activate FL
    cd Refactored
    python inspect_partitions.py 0.3        # alpha=0.3 (default 10 clients)
    python inspect_partitions.py 0.1 20     # alpha=0.1, 20 clients
    python inspect_partitions.py iid        # mode IID
"""

import sys

import numpy as np

from fl_common.data import get_partitions, get_trainset


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "0.3"
    num_clients = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    if arg == "iid":
        mode, alpha = "iid", 0.0
    else:
        mode, alpha = "noniid", float(arg)

    parts = get_partitions(num_clients, partitioning=mode, alpha=alpha)
    targets = np.asarray(get_trainset().targets)

    # Matrice [client, classe]
    counts = np.zeros((num_clients, 10), dtype=int)
    for pid, idx in enumerate(parts):
        if idx:
            for c in targets[idx]:
                counts[pid, c] += 1

    sizes = counts.sum(axis=1)
    fracs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

    label = f"Dirichlet(alpha={alpha})" if mode == "noniid" else "IID"
    print(f"\n=== Partitionnement {label}, {num_clients} clients ===\n")
    print(f"{'pid':>4}  {'n':>6}  | " + "  ".join(f"c{c:>1}" for c in range(10)))
    print("-" * 70)
    for pid in range(num_clients):
        row = "  ".join(f"{fracs[pid, c] * 100:>3.0f}" for c in range(10))
        print(f"{pid:>4}  {sizes[pid]:>6}  | {row}")

    # Stats globales
    nz = np.count_nonzero(counts, axis=1)  # nb classes presentes par client
    dom = fracs.max(axis=1) * 100          # % de la classe dominante
    print()
    print(f"Tailles partitions : min={sizes.min()} max={sizes.max()} "
          f"mean={sizes.mean():.0f} std={sizes.std():.0f}")
    print(f"Nb classes/client  : min={nz.min()} max={nz.max()} mean={nz.mean():.1f}")
    print(f"Classe dominante   : min={dom.min():.0f}% max={dom.max():.0f}% "
          f"mean={dom.mean():.0f}%")
    print()


if __name__ == "__main__":
    main()
