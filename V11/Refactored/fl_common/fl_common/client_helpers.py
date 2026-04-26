"""Helpers partages par les 4 ClientApp.

Logique straggler en 2 temps :
  1. `decide_early_drop` : si panne reseau, le client renvoie immediatement
     un reply "drop" (pas d'entrainement, pas de sleep). Le serveur n'est
     donc pas bloque par les stragglers dropes.
  2. `finalize_comm` : apres entrainement, simule l'upload SAUF si la
     deadline sera depassee (short-circuit).
"""

import time

from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

from .data import model_size_bytes
from .energy import compute_energy_j
from .straggler import network_profile, simulate_comm_delay
from .training import test as _test


# ============================================================================
# 1) Lecture de la config client (run_config + node_config)
# ============================================================================

def read_common_config(context):
    """Params communs a tous les client_app, lus depuis context."""
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
        # Compression : 1.0 = full fp32, 0.25 = quant 8-bit, 0.1 = sparse 90%, etc.
        "comm_size_ratio": float(cfg.get("comm-size-ratio", 1.0)),
        # Override taille modele POUR LA SIMULATION (en MB). 0 = vraie taille.
        "sim_model_mb": float(cfg.get("sim-model-mb", 0.0)),
        # Momentum SGD. Mettre 0.0 pour FedNova (rescaling tau_i suppose vanilla SGD).
        "momentum": float(cfg.get("momentum", 0.9)),
        # Seed reproducibilite. -1 = pas de seeding (comportement par defaut).
        # >= 0 = init torch + DataLoader deterministe pour ce pid.
        "seed": int(cfg.get("seed", -1)),
    }


def compute_tier_epochs(pid, base_epochs, epochs_hetero):
    """Tier compute (0/1/2) + nb d'epochs (heterogeneite hardware).

    Le ratio max/min des epochs determine l'amplitude du biais d'agregation
    de FedAvg : plus c'est extreme, plus FedNova devrait apporter un gain
    visible. Le papier original utilise des ratios ~10x-15x.
    """
    if not epochs_hetero:
        return 1, base_epochs
    tier = pid % 3
    epochs = {0: 1, 1: 3, 2: 6}[tier]   # ratio 15x (avant: 6x)
    return tier, epochs


# ============================================================================
# 2) Simulation reseau (delay + dropouts)
# ============================================================================

def _effective_model_mb(comm_size_ratio, sim_model_mb):
    """Taille effective transmise apres compression (en MB)."""
    base = float(sim_model_mb) if sim_model_mb > 0 else (
        model_size_bytes() / (1024.0 * 1024.0))
    return base * float(comm_size_ratio)


def decide_early_drop(straggler_sim, pid, round_idx, comm_size_ratio=1.0,
                      sim_model_mb=0.0):
    """Retourne (net_tier, delay, dropped_early).

    - `delay` est TOUJOURS calcule (meme sans straggler-sim) pour que
      l'energie reflete la compression. Mais on ne `sleep()` que si
      straggler-sim=1 (via finalize_comm).
    - `dropped_early=True` : panne reseau, skipper l'entrainement.
    """
    model_mb = _effective_model_mb(comm_size_ratio, sim_model_mb)
    if straggler_sim:
        net_tier, delay = simulate_comm_delay(pid, model_mb, round_idx)
        if delay is None:
            return net_tier, None, True
        return net_tier, delay, False
    # Mode rapide : pas de drop, pas de sleep, juste le delay pour l'energie.
    net_tier, bw, rtt, _, _ = network_profile(pid)
    return net_tier, 2.0 * (model_mb / bw) + rtt, False


def finalize_comm(delay, local_time_s, round_deadline):
    """Simule l'upload (sleep), avec short-circuit si la deadline depasse.

    Retourne (comm_time_s, dropped_deadline).
    """
    if round_deadline > 0 and (local_time_s + delay) > round_deadline:
        return 0.0, True
    time.sleep(delay)
    return delay, False


# ============================================================================
# 3) Eval locale (utilisee par les 4 client_app pour visualiser le gap non-IID)
# ============================================================================

def local_eval_metrics(net, loader, device):
    """Eval du modele entraine sur la partition LOCALE du client.

    En non-IID, ce score surestime fortement l'accuracy reelle (chaque client
    overfitte sa propre distribution). C'est le point de comparaison avec
    l'eval serveur sur CIFAR-10 IID -> visualise le biais non-IID.
    """
    loss, acc = _test(net, loader, device)
    return {"local_eval_loss": float(loss), "local_eval_acc": float(acc)}


# ============================================================================
# 4) Construction des replies (drop ou train)
# ============================================================================

def _build_reply(msg, state_dict, metrics):
    """Construit un Message reply standard (poids + metric record)."""
    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(state_dict),
            "metrics": MetricRecord(metrics),
        }),
        reply_to=msg,
    )


def _base_metrics(pid, tier, net_tier):
    """Champs metric communs aux replies drop et train."""
    return {
        "partition_id": float(pid),
        "resource_tier": float(tier),
        "net_tier": float(net_tier),
        "local_eval_loss": 0.0,
        "local_eval_acc": 0.0,
    }


def make_drop_reply(msg, global_sd, pid, tier, net_tier, extra_metrics=None):
    """Reply rapide (sans entrainement) : poids globaux inchanges, dropped=1.

    Energie ~0 : un drop reseau coute quasiment rien (pas de compute, pas d'upload).
    """
    m = _base_metrics(pid, tier, net_tier) | {
        "train_loss": 0.0, "num-examples": 0,
        "local_time_s": 0.0, "comm_time_s": 0.0,
        "epochs_used": 0.0, "dropped": 1.0, "energy_j": 0.0,
    }
    if extra_metrics:
        m.update(extra_metrics)
    return _build_reply(msg, global_sd, m)


def make_train_reply(msg, state_dict, train_loss, num_examples, local_time_s,
                     pid, tier, epochs, net_tier, comm_time_s, dropped,
                     extra_metrics=None):
    """Reply d'entrainement normal.

    Energie calculee automatiquement :
        E = P_compute(tier) * local_time_s + P_comm(net_tier) * comm_time_s
    """
    energy_j = compute_energy_j(tier, net_tier, local_time_s, comm_time_s)
    m = _base_metrics(pid, tier, net_tier) | {
        "train_loss": float(train_loss),
        "num-examples": int(num_examples),
        "local_time_s": float(local_time_s),
        "comm_time_s": float(comm_time_s),
        "epochs_used": float(epochs),
        "dropped": float(dropped),
        "energy_j": float(energy_j),
    }
    if extra_metrics:
        m.update(extra_metrics)
    return _build_reply(msg, state_dict, m)
