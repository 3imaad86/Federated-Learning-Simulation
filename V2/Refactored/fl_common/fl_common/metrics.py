"""Metriques FL (efficacite + fairness). Produit 4 CSV dans results/.

Evaluation 100% cote SERVEUR sur le test set CIFAR-10 officiel (10k IID).
Les metriques par-classe sont calculees sur ce meme test set, donc non
biaisees par le non-IID (contrairement a une eval federee classique).
"""

import csv
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score

RESULTS_DIR = os.environ.get(
    "FL_RESULTS_DIR",
    str(Path(__file__).resolve().parent.parent / "results"),
)
NUM_CLASSES = 10

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
]
PER_CLASS_HEADER = ["round"] + [f"class_{i}" for i in range(NUM_CLASSES)]
SUMMARY_HEADER = [
    "total_time_s", "rtc90", "rounds_to_50", "rounds_to_70", "rounds_to_90",
    "participation_jfi", "worst_participation", "best_participation",
]
PARTICIPATION_HEADER = ["client_id", "times_selected"]


def ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def resolve_dst_results_dir(project_dir_name):
    """Retourne le dossier ou copier les CSV hors du cache Flower."""
    if "FL_RESULTS_DIR" in os.environ:
        return os.environ["FL_RESULTS_DIR"]
    cwd = Path(os.getcwd()).resolve()

    def _matches(p):
        return p.is_dir() and p.name == project_dir_name and (p / "pyproject.toml").exists()

    if _matches(cwd):
        return str(cwd / "results")
    try:
        for child in cwd.iterdir():
            if _matches(child):
                return str(child / "results")
    except OSError:
        pass
    for parent in cwd.parents:
        if _matches(parent):
            return str(parent / "results")
        try:
            for sibling in parent.iterdir():
                if _matches(sibling):
                    return str(sibling / "results")
        except OSError:
            continue
    return str(cwd / "results")


def reset_files():
    ensure_dir()
    for p in (GLOBAL_CSV, PER_CLASS_CSV, SUMMARY_CSV, PARTICIPATION_CSV):
        if os.path.exists(p):
            os.remove(p)


def _append(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def _overwrite(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def jains_fairness_index(values):
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    s = arr.sum()
    sq = float((arr * arr).sum())
    if sq == 0.0:
        return 0.0
    return float((s * s) / (arr.size * sq))


def class_accuracies_from_preds(y_true, y_pred, num_classes=NUM_CLASSES):
    labels = list(range(num_classes))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return [0.0] * num_classes
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accs = []
    for c in labels:
        tot = int(cm[c].sum())
        accs.append(float(cm[c, c] / tot) if tot > 0 else 0.0)
    return accs


def macro_recall_f1_from_preds(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0, 0.0
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float(macro_recall), float(macro_f1)


def rounds_to_convergence(accuracies, ratio=0.9):
    accs = list(accuracies)
    if not accs:
        return None
    threshold = ratio * max(accs)
    for i, a in enumerate(accs, start=1):
        if a >= threshold:
            return i
    return None


def rounds_to_target(accuracies, target):
    for i, a in enumerate(accuracies, start=1):
        if a >= target:
            return i
    return None


def log_round(server_round, accuracy, loss, macro_recall, macro_f1,
              class_accuracies,
              comm_cost_mb=0.0,
              round_time_s=0.0, mean_client_time_s=0.0, max_client_time_s=0.0,
              mean_epochs_used=0.0, mean_resource_tier=0.0):
    """Ecrit une ligne dans metrics_global.csv + metrics_per_class.csv.

    Toutes les metriques de qualite viennent de l'eval centralisee serveur
    (non biaisee par le non-IID). Fairness par CLASSE uniquement.
    """
    ensure_dir()

    class_accs = [float(a) for a in (class_accuracies or [])]
    if len(class_accs) < NUM_CLASSES:
        class_accs = class_accs + [0.0] * (NUM_CLASSES - len(class_accs))
    else:
        class_accs = class_accs[:NUM_CLASSES]

    jfi_k = jains_fairness_index(class_accs)
    worst_k = float(min(class_accs))
    var_k = float(np.var(class_accs))
    gap_k = float(max(class_accs) - min(class_accs))

    _append(
        GLOBAL_CSV, GLOBAL_HEADER,
        [server_round, float(accuracy), float(loss),
         float(macro_recall), float(macro_f1),
         jfi_k, worst_k, var_k, gap_k,
         float(comm_cost_mb),
         float(round_time_s), float(mean_client_time_s), float(max_client_time_s),
         float(mean_epochs_used), float(mean_resource_tier)],
    )
    _append(
        PER_CLASS_CSV, PER_CLASS_HEADER,
        [server_round] + class_accs,
    )


def log_summary(total_time_s, accs_history, participation_counts,
                targets=(0.5, 0.7, 0.9), num_clients=None):
    ensure_dir()
    rtc90 = rounds_to_convergence(accs_history, ratio=0.9)
    rt = [rounds_to_target(accs_history, t) for t in targets]
    if isinstance(participation_counts, dict):
        if num_clients is not None:
            counts = [int(participation_counts.get(cid, 0)) for cid in range(int(num_clients))]
        else:
            counts = list(participation_counts.values())
    else:
        counts = list(participation_counts) if participation_counts else []
    p_jfi = jains_fairness_index(counts) if counts else 0.0
    worst_p = int(min(counts)) if counts else 0
    best_p = int(max(counts)) if counts else 0
    _overwrite(
        SUMMARY_CSV, SUMMARY_HEADER,
        [[float(total_time_s),
          rtc90 if rtc90 is not None else "",
          rt[0] if rt[0] is not None else "",
          rt[1] if rt[1] is not None else "",
          rt[2] if rt[2] is not None else "",
          p_jfi, worst_p, best_p]],
    )


def log_participation(participation_counts, num_clients=None):
    ensure_dir()
    if num_clients is not None:
        rows = [[cid, int(participation_counts.get(cid, 0))] for cid in range(int(num_clients))]
    else:
        rows = [[int(cid), int(n)] for cid, n in sorted(participation_counts.items())]
    _overwrite(PARTICIPATION_CSV, PARTICIPATION_HEADER, rows)


def extract_server_round(msg):
    cfg = msg.content.get("config")
    if cfg is not None:
        sr = cfg.get("server-round")
        if sr is not None:
            return int(sr)
    md = getattr(msg, "metadata", None)
    if md is not None:
        gid = getattr(md, "group_id", None)
        if gid:
            try:
                return int(gid)
            except (TypeError, ValueError):
                pass
    return -1
