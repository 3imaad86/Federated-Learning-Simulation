"""Metriques FL (efficacite + fairness). Produit 4 CSV dans results/.

Eval 100% cote SERVEUR sur le test set CIFAR-10 (10k IID).
Toutes les metriques par-classe sont donc non biaisees par le non-IID.
"""

import csv
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score


# ============================================================================
# 1) Constantes : chemins, headers, num classes
# ============================================================================

NUM_CLASSES = 10

RESULTS_DIR = os.environ.get(
    "FL_RESULTS_DIR",
    str(Path(__file__).resolve().parent.parent / "results"),
)
GLOBAL_CSV = os.path.join(RESULTS_DIR, "metrics_global.csv")
PER_CLASS_CSV = os.path.join(RESULTS_DIR, "metrics_per_class.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")
PARTICIPATION_CSV = os.path.join(RESULTS_DIR, "metrics_participation.csv")

GLOBAL_HEADER = [
    "round", "accuracy", "loss", "macro_recall", "macro_f1",
    "jfi_classes", "worst_class_acc", "acc_variance_classes", "min_max_class_gap",
    "comm_cost_mb",
    "round_time_s", "mean_client_time_s", "max_client_time_s",
    "mean_epochs_used", "mean_resource_tier",
    "energy_j_round", "energy_j_cumulative",
    "local_loss", "local_acc",
]
PER_CLASS_HEADER = ["round"] + [f"class_{i}" for i in range(NUM_CLASSES)]
SUMMARY_HEADER = [
    "total_time_s", "rtc90", "rounds_to_50", "rounds_to_70", "rounds_to_90",
    "participation_jfi", "worst_participation", "best_participation",
]
PARTICIPATION_HEADER = ["client_id", "times_selected"]


# ============================================================================
# 2) Gestion des fichiers (creer / reset / append / overwrite)
# ============================================================================

def ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def reset_files():
    ensure_dir()
    for p in (GLOBAL_CSV, PER_CLASS_CSV, SUMMARY_CSV, PARTICIPATION_CSV):
        if os.path.exists(p):
            os.remove(p)


def _append(path, header, row):
    """Append une ligne (cree le fichier + header si absent)."""
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def _overwrite(path, header, rows):
    """Reecrit le fichier complet (header + rows)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def resolve_dst_results_dir(project_dir_name):
    """Cherche le dossier `<project>/results` dans cwd ou parents pour copier les CSV.

    Pourquoi : Flower simulation execute le code depuis un cache temporaire,
    donc on copie les CSV dans le vrai dossier projet pour qu'ils soient persistents.
    """
    if "FL_RESULTS_DIR" in os.environ:
        return os.environ["FL_RESULTS_DIR"]

    def is_project(p):
        return p.is_dir() and p.name == project_dir_name and (p / "pyproject.toml").exists()

    cwd = Path(os.getcwd()).resolve()
    candidates = [cwd] + list(cwd.parents)
    for parent in candidates:
        if is_project(parent):
            return str(parent / "results")
        try:
            for child in parent.iterdir():
                if is_project(child):
                    return str(child / "results")
        except OSError:
            continue
    return str(cwd / "results")


# ============================================================================
# 3) Calculs de metriques (fairness + per-class + convergence)
# ============================================================================

def jains_fairness_index(values):
    """JFI = (sum xi)^2 / (n * sum xi^2). 1 = parfaitement equitable."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or (arr * arr).sum() == 0.0:
        return 0.0
    return float(arr.sum() ** 2 / (arr.size * (arr * arr).sum()))


def class_accuracies_from_preds(y_true, y_pred, num_classes=NUM_CLASSES):
    """Accuracy par classe a partir d'une matrice de confusion."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        return [0.0] * num_classes
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return [float(cm[c, c] / cm[c].sum()) if cm[c].sum() > 0 else 0.0
            for c in range(num_classes)]


def macro_recall_f1_from_preds(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0, 0.0
    return (
        float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def rounds_to_convergence(accuracies, ratio=0.9):
    """Nb de rounds pour atteindre `ratio` × max(accuracies)."""
    accs = list(accuracies)
    if not accs:
        return None
    threshold = ratio * max(accs)
    for i, a in enumerate(accs, start=1):
        if a >= threshold:
            return i
    return None


def rounds_to_target(accuracies, target):
    """Nb de rounds pour atteindre `target` (ex: 0.5)."""
    for i, a in enumerate(accuracies, start=1):
        if a >= target:
            return i
    return None


# ============================================================================
# 4) Logging par round (metrics_global.csv + metrics_per_class.csv)
# ============================================================================

def _normalize_class_accs(class_accuracies):
    """Pad/tronque la liste pour avoir exactement NUM_CLASSES."""
    accs = [float(a) for a in (class_accuracies or [])]
    if len(accs) < NUM_CLASSES:
        return accs + [0.0] * (NUM_CLASSES - len(accs))
    return accs[:NUM_CLASSES]


def log_round(server_round, accuracy, loss, macro_recall, macro_f1,
              class_accuracies,
              comm_cost_mb=0.0,
              round_time_s=0.0, mean_client_time_s=0.0, max_client_time_s=0.0,
              mean_epochs_used=0.0, mean_resource_tier=0.0,
              energy_j_round=0.0, energy_j_cumulative=0.0,
              local_loss=0.0, local_acc=0.0):
    """Ecrit une ligne dans metrics_global.csv + metrics_per_class.csv."""
    ensure_dir()
    class_accs = _normalize_class_accs(class_accuracies)

    _append(GLOBAL_CSV, GLOBAL_HEADER, [
        server_round, float(accuracy), float(loss),
        float(macro_recall), float(macro_f1),
        jains_fairness_index(class_accs),
        float(min(class_accs)),
        float(np.var(class_accs)),
        float(max(class_accs) - min(class_accs)),
        float(comm_cost_mb),
        float(round_time_s), float(mean_client_time_s), float(max_client_time_s),
        float(mean_epochs_used), float(mean_resource_tier),
        float(energy_j_round), float(energy_j_cumulative),
        float(local_loss), float(local_acc),
    ])
    _append(PER_CLASS_CSV, PER_CLASS_HEADER, [server_round] + class_accs)


# ============================================================================
# 5) Logging final (summary + participation)
# ============================================================================

def _participation_counts_list(counts, num_clients=None):
    """Convertit le dict {pid: count} -> liste alignee sur range(num_clients)."""
    if isinstance(counts, dict):
        if num_clients is not None:
            return [int(counts.get(cid, 0)) for cid in range(int(num_clients))]
        return list(counts.values())
    return list(counts) if counts else []


def log_summary(total_time_s, accs_history, participation_counts,
                targets=(0.5, 0.7, 0.9), num_clients=None):
    """Ecrit metrics_summary.csv (rtc90, rounds-to-targets, participation fairness)."""
    ensure_dir()
    rtc90 = rounds_to_convergence(accs_history, ratio=0.9)
    rt = [rounds_to_target(accs_history, t) for t in targets]
    counts = _participation_counts_list(participation_counts, num_clients)

    _overwrite(SUMMARY_CSV, SUMMARY_HEADER, [[
        float(total_time_s),
        rtc90 if rtc90 is not None else "",
        rt[0] if rt[0] is not None else "",
        rt[1] if rt[1] is not None else "",
        rt[2] if rt[2] is not None else "",
        jains_fairness_index(counts) if counts else 0.0,
        int(min(counts)) if counts else 0,
        int(max(counts)) if counts else 0,
    ]])


def log_participation(participation_counts, num_clients=None):
    """Ecrit metrics_participation.csv (combien de fois chaque client a participe)."""
    ensure_dir()
    if num_clients is not None:
        rows = [[cid, int(participation_counts.get(cid, 0))]
                for cid in range(int(num_clients))]
    else:
        rows = [[int(cid), int(n)]
                for cid, n in sorted(participation_counts.items())]
    _overwrite(PARTICIPATION_CSV, PARTICIPATION_HEADER, rows)


# ============================================================================
# 6) Helper divers (utilitaire pour les strategies)
# ============================================================================

def extract_server_round(msg):
    """Recupere le numero du round depuis un Message Flower (config OU group_id)."""
    cfg = msg.content.get("config")
    if cfg is not None and cfg.get("server-round") is not None:
        return int(cfg.get("server-round"))
    md = getattr(msg, "metadata", None)
    if md is not None:
        gid = getattr(md, "group_id", None)
        if gid:
            try:
                return int(gid)
            except (TypeError, ValueError):
                pass
    return -1
