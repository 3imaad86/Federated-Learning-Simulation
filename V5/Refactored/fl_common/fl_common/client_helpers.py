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
from .straggler import network_profile, simulate_comm_delay
from .training import test as _test


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
        # Ratio de taille effective transmise apres compression/quantification.
        # 1.0 = no compression, 0.25 = quantif 8-bit, 0.1 = sparsite 90%, etc.
        "comm_size_ratio": float(cfg.get("comm-size-ratio", 1.0)),
    }


def compute_tier_epochs(pid, base_epochs, epochs_hetero):
    """Tier de calcul + nb d'epochs (heterogeneite compute)."""
    if not epochs_hetero:
        return 1, base_epochs
    tier = pid % 3
    epochs = {0: 1, 1: 3, 2: 6}[tier]
    return tier, epochs


def decide_early_drop(straggler_sim, pid, round_idx, comm_size_ratio=1.0):
    """Simulation reseau. Retourne (net_tier, delay, dropped_early).

    - `comm_size_ratio` : taille effective apres compression (1.0 = full fp32,
      0.25 = quant 8-bit, etc). Reduit `delay` -> reduit l'energie comm.
    - `delay` est TOUJOURS calcule (meme sans straggler-sim) pour que l'energie
      reflete la compression. Mais on ne sleep() que si straggler-sim=1.
    - `dropped_early=True` : panne reseau, skipper l'entrainement.
    """
    model_mb = model_size_bytes() / (1024.0 * 1024.0) * float(comm_size_ratio)
    if straggler_sim:
        net_tier, delay = simulate_comm_delay(pid, model_mb, round_idx)
        if delay is None:
            return net_tier, None, True
        return net_tier, delay, False
    # Mode simulation rapide : delay simule pour l'energie, pas de drop, pas de sleep.
    net_tier, bw, rtt, _, _ = network_profile(pid)
    delay = 2.0 * (model_mb / bw) + rtt
    return net_tier, delay, False


def local_eval_metrics(net, loader, device):
    """Eval du modele local sur la partition du client (pour voir le gap non-IID).

    En non-IID, le modele local overfitte sa propre distribution : l'accuracy
    locale est sur-estimee par rapport a l'eval serveur sur un test set IID.
    Cette difference (visible sur les plots server vs local) est la signature
    du biais non-IID.
    """
    loss, acc = _test(net, loader, device)
    return {"local_eval_loss": float(loss), "local_eval_acc": float(acc)}


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
        "local_eval_loss": 0.0, "local_eval_acc": 0.0,
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
        "local_eval_loss": 0.0, "local_eval_acc": 0.0,
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
