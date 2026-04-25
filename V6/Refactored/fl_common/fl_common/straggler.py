"""Simulation stragglers : reseau variable + dropouts aleatoires (edge IoT)."""

import os
import random as _rnd

SEED = int(os.environ.get("FL_SEED", "42"))

# tier : (bw_mbps, rtt_s, jitter_s, p_drop_par_round)
NET_TIERS = {
    0: (0.5,  0.8,  0.3,  0.15),   # faible (LoRa / 2G)        -> 15% dropout
    1: (5.0,  0.2,  0.05, 0.05),   # moyen  (LTE smartphone)   ->  5% dropout
    2: (50.0, 0.03, 0.01, 0.01),   # fort   (WiFi edge gateway)->  1% dropout
}
NET_TIER_WEIGHTS = [0.4, 0.4, 0.2]


def network_profile(pid, seed=SEED):
    """Tier reseau STABLE du client (meme pid -> meme tier a tous les rounds)."""
    rng = _rnd.Random(seed + pid)
    tier = rng.choices([0, 1, 2], weights=NET_TIER_WEIGHTS)[0]
    bw, rtt, jitter, pdrop = NET_TIERS[tier]
    return tier, bw, rtt, jitter, pdrop


def simulate_comm_delay(pid, model_mb, round_idx, seed=SEED):
    """Simule la communication d'UN round pour le client `pid`.

    Renvoie (tier, delay_s) si transfert reussi, (tier, None) si dropout reseau.
    delay = 2*(model_mb/bw) + rtt + |jitter|  (download + upload du modele)
    """
    tier, bw, rtt, jitter, pdrop = network_profile(pid, seed)
    rng = _rnd.Random(seed + pid * 1000 + round_idx)
    if rng.random() < pdrop:
        return tier, None
    delay = 2.0 * (model_mb / bw) + rtt + abs(rng.gauss(0.0, jitter))
    return tier, delay
