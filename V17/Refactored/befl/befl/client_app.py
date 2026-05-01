"""ClientApp BEFL : Balancing Energy Consumption (Liu et al. 2022).

Idee : chaque client adapte son nombre d'epochs locales en fonction du
niveau de batterie restant.
  - Batterie pleine -> fait toutes les epochs (ex: 2)
  - Demi batterie   -> fait moitie des epochs (ex: 1)
  - Batterie vide   -> drop (ne participe plus)

Resultat attendu : equilibrage de l'energie consommee entre clients,
durant la duree de vie totale de la federation.

Le compteur d'energie est sauve dans context.state (persiste entre rounds).
"""

import time

from flwr.app import Context, Message, MetricRecord
from flwr.clientapp import ClientApp

from fl_common.client_helpers import (
    compute_tier_epochs, decide_early_drop, finalize_comm,
    local_eval_metrics, make_drop_reply, make_train_reply, read_common_config,
)
from fl_common.data import Net, get_device, load_data, set_seed
from fl_common.energy import battery_for_tier, compute_energy_j
from fl_common.training import train as train_fn

app = ClientApp()


# ============================================================================
# Helpers : lecture / sauvegarde du cumul d'energie via context.state
# ============================================================================

def get_energy_used(context):
    """Lit l'energie cumulee depuis context.state. 0 au 1er round."""
    mr = context.state.metric_records.get("befl_state")
    if mr is None:
        return 0.0
    return float(mr.get("energy_used", 0.0))


def save_energy_used(context, energy_used):
    """Sauve l'energie cumulee pour les rounds suivants."""
    context.state.metric_records["befl_state"] = MetricRecord(
        {"energy_used": float(energy_used)}
    )


# ============================================================================
# Logique BEFL : calcul du ratio batterie + adaptation des epochs
# ============================================================================

def battery_ratio(energy_used, battery_j):
    """Retourne (1 - energy_used / battery) clipe a [0, 1].

    battery_j <= 0 -> mode unlimited (toujours 1.0).
    """
    if battery_j <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - energy_used / battery_j))


def battery_metrics(energy_used, battery_j):
    """Metriques batterie envoyees au serveur pour affichage/logging."""
    ratio = battery_ratio(energy_used, battery_j)
    remaining_j = 0.0 if battery_j <= 0 else max(0.0, battery_j - energy_used)
    return {
        "battery_remaining": float(ratio),
        "battery_remaining_j": float(remaining_j),
        "battery_capacity_j": float(battery_j),
    }


def adapted_epochs(base_epochs, ratio):
    """Nombre d'epochs adapte au ratio batterie.

    ratio=1.0 -> base_epochs
    ratio=0.5 -> moitie de base_epochs (arrondi)
    ratio=0.0 -> 0  (le client n'a plus assez de batterie pour 1 epoch utile)

    L'ancienne version forcait `max(1, ...)` ce qui faisait toujours au moins
    1 epoch tant que le client n'etait pas dead -- contredisait le but du
    papier BEFL (faire MOINS d'epochs quand la batterie est faible). Le
    cas epochs=0 est traite cote appelant comme un drop economique.
    """
    return max(0, int(round(base_epochs * ratio)))


# ============================================================================
# Train principal
# ============================================================================

@app.train()
def train(msg: Message, context: Context):
    c = read_common_config(context)
    pid = c["pid"]
    if c["seed"] >= 0:
        set_seed(c["seed"] + pid)

    # BEFL params (lus directement depuis run_config)
    cfg = context.run_config
    base_battery_j = float(cfg.get("befl-battery-j", 0.0))
    death_threshold = float(cfg.get("befl-death-threshold", 0.05))

    base_tier, base_epochs = compute_tier_epochs(
        pid, c["base_epochs"], c["epochs_hetero"])
    battery_j = battery_for_tier(base_battery_j, base_tier)

    # 1) Recupere energie cumulee + calcule ratio batterie
    energy_used = get_energy_used(context)
    ratio = battery_ratio(energy_used, battery_j)

    lr = msg.content["config"]["lr"]
    round_idx = int(msg.content["config"].get("round", 0))
    global_sd = msg.content["arrays"].to_torch_state_dict()

    # 2) Si batterie morte : drop direct (pas d'entrainement, energie ~0)
    if battery_j > 0 and ratio < death_threshold:
        return make_drop_reply(
            msg, global_sd, pid, base_tier, 1,
            extra_metrics=battery_metrics(energy_used, battery_j))

    # 3) Adaptation des epochs en fonction de la batterie
    epochs = adapted_epochs(base_epochs, ratio)
    # Si la batterie ne permet meme pas 1 epoch utile (apres arrondi), drop
    # economique : pas d'entrainement ni d'upload, energie ~0.
    if epochs <= 0:
        return make_drop_reply(
            msg, global_sd, pid, base_tier, 1,
            extra_metrics=battery_metrics(energy_used, battery_j))

    # 4) Drop reseau classique (panne reseau, independant de la batterie)
    net_tier, delay, dropped_early = decide_early_drop(
        c["straggler_sim"], pid, round_idx,
        c["comm_size_ratio"], c["sim_model_mb"])
    if dropped_early:
        return make_drop_reply(
            msg, global_sd, pid, base_tier, net_tier,
            extra_metrics=battery_metrics(energy_used, battery_j))

    # 5) Setup model + load data
    model = Net()
    model.load_state_dict(global_sd)
    device = get_device()
    model.to(device)

    trainloader, _ = load_data(pid, c["num_parts"], c["bs"],
                               data_hetero=c["data_hetero"],
                               partitioning=c["partitioning"],
                               alpha=c["dir_alpha"],
                               seed=c["seed"])

    # 6) Training avec epochs adaptes
    t0 = time.perf_counter()
    train_loss, _ = train_fn(model, trainloader, epochs, lr, device,
                             momentum=c["momentum"])
    local_time_s = time.perf_counter() - t0

    # 7) Upload simule + check deadline
    comm_time_s, dropped = delay, 0
    if c["straggler_sim"]:
        comm_time_s, dropped_deadline = finalize_comm(
            delay, local_time_s, c["round_deadline"])
        if dropped_deadline:
            dropped = 1
            model.load_state_dict(global_sd)

    # 8) Calcule l'energie consommee CE round + maj cumul
    this_round_j = compute_energy_j(base_tier, net_tier, local_time_s, comm_time_s)
    energy_used_after = energy_used + this_round_j
    save_energy_used(context, energy_used_after)
    new_ratio = battery_ratio(energy_used_after, battery_j)

    # 9) Eval locale + battery dans les metrics envoyees au serveur
    extra = local_eval_metrics(model, trainloader, device)
    extra.update(battery_metrics(energy_used_after, battery_j))

    return make_train_reply(
        msg, model.state_dict(), train_loss, len(trainloader.dataset),
        local_time_s, pid, base_tier, epochs, net_tier, comm_time_s, dropped,
        extra_metrics=extra,
    )
