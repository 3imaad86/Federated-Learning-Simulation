"""Modele d'energie simple pour edge IoT.

On estime la consommation d'un client sur un round par :

    E = P_compute * local_time_s  +  P_comm * comm_time_s

Les puissances (Watts) dependent du device (resource_tier) pour le compute
et du lien reseau (net_tier) pour la communication. Les valeurs sont des
ordres de grandeur typiques d'edge IoT pour simulation uniquement.
"""

# Puissance CPU/GPU pendant l'entrainement (Watts)
#   tier 0 = device faible (ex: Raspberry Pi Zero, MCU)
#   tier 1 = device moyen  (ex: Raspberry Pi 4)
#   tier 2 = device fort   (ex: Jetson Nano)
POWER_COMPUTE_W = {
    0: 1.5,
    1: 4.0,
    2: 7.0,
}

# Puissance radio pendant l'upload (Watts)
#   net_tier 0 = LoRa  (faible puissance, debit tres lent)
#   net_tier 1 = LTE   (moderee)
#   net_tier 2 = WiFi  (moderee)
POWER_COMM_W = {
    0: 0.2,
    1: 2.0,
    2: 0.8,
}


def compute_energy_j(tier, net_tier, local_time_s, comm_time_s):
    """Retourne l'energie en Joules pour un round d'un client."""
    p_comp = POWER_COMPUTE_W.get(int(tier), POWER_COMPUTE_W[1])
    p_comm = POWER_COMM_W.get(int(net_tier), POWER_COMM_W[1])
    return p_comp * float(local_time_s) + p_comm * float(comm_time_s)
