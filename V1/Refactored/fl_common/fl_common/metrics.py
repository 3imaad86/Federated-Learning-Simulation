"""Metriques FL (efficacite + fairness). Produit 4 CSV dans results/.

Ajout vs projets initiaux : 2 colonnes dans metrics_global.csv pour
la validation CENTRALISEE cote serveur (sur le test set CIFAR-10 officiel) :
  * central_accuracy
  * central_loss
Ainsi que central_macro_recall / central_macro_f1.

Les metriques client-side (global_accuracy, jfi_clients, ...) restent
en place : elles mesurent la fairness, pas la qualite du modele sous non-IID.
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
    "round", "global_accuracy", "global_loss", "comm_cost_mb",
    "macro_recall", "macro_f1",
    "jfi_clients", "worst_client_acc", "acc_variance_clients",
    "jfi_classes", "worst_class_acc", "acc_variance_classes", "min_max_class_gap",
    "round_time_s", "mean_client_time_s", "max_client_time_s",
    "mean_epochs_used", "mean_resource_tier",
    "central_accuracy", "central_loss", "central_macro_recall", "central_macro_f1",
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


def log_round(server_round, global_accuracy, global_loss, comm_cost_mb,
              macro_recall, macro_f1,
              client_accuracies, class_accuracies,
              round_time_s=0.0, mean_client_time_s=0.0, max_client_time_s=0.0,
              mean_epochs_used=0.0, mean_resource_tier=0.0,
              central_accuracy=0.0, central_loss=0.0,
              central_macro_recall=0.0, central_macro_f1=0.0):
    """Cote SERVEUR : calcule fairness + temps et ecrit les 2 CSV par round.

    Les parametres central_* viennent de l'evaluation centralisee sur le
    test set CIFAR-10 officiel (10k images IID).
    """
    ensure_dir()

    client_accs = [float(a) for a in (client_accuracies or [])]
    class_accs = [float(a) for a in (class_accuracies or [])]
    if len(class_accs) < NUM_CLASSES:
        class_accs = class_accs + [0.0] * (NUM_CLASSES - len(class_accs))
    else:
        class_accs = class_accs[:NUM_CLASSES]

    jfi_c = jains_fairness_index(client_accs)
    worst_c = float(min(client_accs)) if client_accs else 0.0
    var_c = float(np.var(client_accs)) if client_accs else 0.0

    jfi_k = jains_fairness_index(class_accs)
    worst_k = float(min(class_accs))
    var_k = float(np.var(class_accs))
    gap_k = float(max(class_accs) - min(class_accs))

    _append(
        GLOBAL_CSV, GLOBAL_HEADER,
        [server_round, float(global_accuracy), float(global_loss), float(comm_cost_mb),
         float(macro_recall), float(macro_f1),
         jfi_c, worst_c, var_c, jfi_k, worst_k, var_k, gap_k,
         float(round_time_s), float(mean_client_time_s), float(max_client_time_s),
         float(mean_epochs_used), float(mean_resource_tier),
         float(central_accuracy), float(central_loss),
         float(central_macro_recall), float(central_macro_f1)],
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
