"""Helpers partages par les 4 ClientApp.

Logique straggler en 2 temps :
  1. `decide_early_drop` : si panne reseau, le client renvoie immediatement
     un reply "drop" (pas d'entrainement, pas de sleep). Le serveur n'est
     donc pas bloque par les stragglers dropes.
  2. `finalize_comm` : apres entrainement, simule l'upload SAUF si on sait
     deja que la deadline sera depassee (short-circuit).
"""

import time

from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

from .data import model_size_bytes
from .energy import compute_energy_j
from .straggler import simulate_comm_delay


def read_common_config(context):
    """Params communs a tous les client_app lus depuis run_config."""
    cfg = context.run_config
    return {
        "pid": context.node_config["partition-id"],
        "num_parts": context.node_config["num-partitions"],
        "bs": cfg["batch-size"],
        "base_epochs": int(cfg.get("local-epochs", 1)),
        "data_hetero": int(cfg.get("data-heterogeneity", 0)),
        "epochs_hetero": int(cfg.get("epochs-heterogeneity", 0)),
        "partitioning": str(cfg.get("partitioning", "noniid")),
        "dir_alpha": float(cfg.get("dirichlet-alpha", 0.3)),
        "straggler_sim": int(cfg.get("straggler-sim", 0)),
        "round_deadline": float(cfg.get("round-deadline-s", 0.0)),
    }


def compute_tier_epochs(pid, base_epochs, epochs_hetero):
    """Tier de calcul + nb d'epochs (heterogeneite compute)."""
    if not epochs_hetero:
        return 1, base_epochs
    tier = pid % 3
    epochs = {0: 1, 1: 3, 2: 6}[tier]
    return tier, epochs


def decide_early_drop(straggler_sim, pid, round_idx):
    """Dropout PRE-training. Retourne (net_tier, delay, dropped_early).

    Si dropped_early=True : skipper l'entrainement, retour instantane.
    Sinon : `delay` secondes a attendre (simule upload) apres entrainement.
    """
    if not straggler_sim:
        return 1, 0.0, False
    model_mb = model_size_bytes() / (1024.0 * 1024.0)
    net_tier, delay = simulate_comm_delay(pid, model_mb, round_idx)
    if delay is None:
        return net_tier, None, True
    return net_tier, delay, False


def make_drop_reply(msg, global_sd, pid, tier, net_tier, extra_metrics=None):
    """Reply rapide (sans entrainement) : poids globaux inchanges + dropped=1.

    Energie ~0 : le client n'a ni entraine ni upload (un drop reseau coute
    quasiment rien en energie).
    """
    m = {
        "train_loss": 0.0, "num-examples": 0, "local_time_s": 0.0,
        "partition_id": float(pid), "epochs_used": 0.0,
        "resource_tier": float(tier), "net_tier": float(net_tier),
        "comm_time_s": 0.0, "dropped": 1.0, "energy_j": 0.0,
    }
    if extra_metrics:
        m.update(extra_metrics)
    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(global_sd),
            "metrics": MetricRecord(m),
        }),
        reply_to=msg,
    )


def finalize_comm(delay, local_time_s, round_deadline):
    """Simule l'upload, avec short-circuit si la deadline sera depassee.

    Retourne (comm_time_s, dropped_deadline).
    """
    if round_deadline > 0 and (local_time_s + delay) > round_deadline:
        return 0.0, True
    time.sleep(delay)
    return delay, False


def make_train_reply(msg, state_dict, train_loss, num_examples, local_time_s,
                     pid, tier, epochs, net_tier, comm_time_s, dropped,
                     extra_metrics=None):
    """Reply d'entrainement normal (non drope).

    L'energie est calculee automatiquement :
        E = P_compute(tier) * local_time_s + P_comm(net_tier) * comm_time_s
    """
    energy_j = compute_energy_j(tier, net_tier, local_time_s, comm_time_s)
    m = {
        "train_loss": float(train_loss),
        "num-examples": int(num_examples),
        "local_time_s": float(local_time_s),
        "partition_id": float(pid),
        "epochs_used": float(epochs),
        "resource_tier": float(tier),
        "net_tier": float(net_tier),
        "comm_time_s": float(comm_time_s),
        "dropped": float(dropped),
        "energy_j": float(energy_j),
    }
    if extra_metrics:
        m.update(extra_metrics)
    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(state_dict),
            "metrics": MetricRecord(m),
        }),
        reply_to=msg,
    )
