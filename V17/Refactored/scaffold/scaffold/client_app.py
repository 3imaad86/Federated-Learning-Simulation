"""ClientApp SCAFFOLD : training avec correction par control variates.

Le client recoit (w_global, c_global) packes dans le meme ArrayRecord avec
prefix __cg__, et maintient son c_local persistant via context.state (cle
versionnee par le partitionnement courant pour eviter les bugs cross-config).

Etapes par round :
  1. Decode (w_global, c_global) du message
  2. Recupere c_local du context.state (zeros au 1er round)
  3. Train SCAFFOLD : y = y - lr * (grad + c_global - c_local)
  4. Calcule c_local_new = c_local - c_global + (1/(K*lr)) * (w_global - y)
  5. Calcule delta_c = c_local_new - c_local
  6. Verifie le drop deadline ; si on participe vraiment, sauve c_local_new
     dans context.state (sinon on garde l'ancien c_local)
  7. Renvoie (y, delta_c) packes (prefix __dc__)
"""

import time

import torch
from flwr.app import ArrayRecord, Context, Message
from flwr.clientapp import ClientApp

from fl_common.client_helpers import (
    compute_tier_epochs, decide_early_drop, finalize_comm,
    local_eval_metrics, make_drop_reply, make_train_reply, read_common_config,
)
from fl_common.data import Net, get_device, load_data, set_seed
from fl_common.strategy import CG_PREFIX, DC_PREFIX
from fl_common.training import train_scaffold

app = ClientApp()


def _split_arrays(combined_sd):
    """Decode (w_global, c_global) depuis l'ArrayRecord combine."""
    w_sd, c_global_sd = {}, {}
    for k, v in combined_sd.items():
        if k.startswith(CG_PREFIX):
            c_global_sd[k[len(CG_PREFIX):]] = v
        else:
            w_sd[k] = v
    return w_sd, c_global_sd


def _c_local_key(c):
    """Cle de stockage versionnee par configuration de partitionnement.

    Si l'utilisateur change `num-clients`, `partitioning` ou `dirichlet-alpha`
    entre deux runs, la partition locale du client devient differente -- son
    ancien c_local n'a plus de sens et peut meme avoir une shape incompatible.
    En hashant la config dans la cle, chaque partitionnement obtient son
    propre c_local, ce qui evite les bugs silencieux.
    """
    return (
        f"c_local_n{c['num_parts']}"
        f"_p{c['partitioning']}"
        f"_a{c['dir_alpha']:.4f}"
    )


def _get_or_init_c_local(context, c_global_sd, key):
    """Lit c_local depuis context.state. Si absent, init a zeros."""
    if key in context.state.array_records:
        return context.state.array_records[key].to_torch_state_dict()
    return {name: torch.zeros_like(t) for name, t in c_global_sd.items()}


def _save_c_local(context, c_local_sd, key):
    """Sauve c_local pour le round suivant."""
    context.state.array_records[key] = ArrayRecord(c_local_sd)


def _pack_y_and_delta_c(model, delta_c_sd):
    """Pack y (model state) + delta_c dans un seul state_dict (prefix __dc__)."""
    combined = dict(model.state_dict())
    for name, t in delta_c_sd.items():
        combined[f"{DC_PREFIX}{name}"] = t
    return combined


@app.train()
def train(msg: Message, context: Context):
    c = read_common_config(context)
    pid = c["pid"]
    if c["seed"] >= 0:
        set_seed(c["seed"] + pid)
    tier, epochs = compute_tier_epochs(pid, c["base_epochs"], c["epochs_hetero"])
    lr = msg.content["config"]["lr"]
    round_idx = int(msg.content["config"].get("round", 0))

    # Garde-fou SCAFFOLD : la correction (c_global - c_local) injectee dans le
    # gradient s'accumule dans le buffer momentum, ce qui peut faire diverger
    # le training. Theorie : vanilla SGD. On previent UNE fois (pid=0, r=1).
    if pid == 0 and round_idx == 1 and c["momentum"] > 0:
        print(f"[scaffold] WARN: momentum={c['momentum']} > 0, peut faire "
              f"diverger SCAFFOLD (theorie suppose vanilla SGD).")

    # 1) Decode (w_global, c_global) du message
    combined_sd = msg.content["arrays"].to_torch_state_dict()
    w_global_sd, c_global_sd = _split_arrays(combined_sd)

    # 2) Dropout PRE-training (panne reseau)
    net_tier, delay, dropped_early = decide_early_drop(
        c["straggler_sim"], pid, round_idx,
        c["comm_size_ratio"], c["sim_model_mb"])
    if dropped_early:
        # delta_c = 0 pour un drop (pas de contribution)
        zero_dc = {name: torch.zeros_like(t) for name, t in c_global_sd.items()}
        drop_combined = _pack_drop_reply(w_global_sd, zero_dc)
        return make_drop_reply(msg, drop_combined, pid, tier, net_tier)

    # 3) Setup model + recupere c_local persistant (cle versionnee par config)
    model = Net()
    model.load_state_dict(w_global_sd)
    device = get_device()
    model.to(device)
    state_key = _c_local_key(c)
    c_local_sd = _get_or_init_c_local(context, c_global_sd, state_key)

    # 4) Sauve w_global pour calculer (w - y) apres training
    w_global_copy = {name: t.detach().clone() for name, t in w_global_sd.items()}

    # 5) Train SCAFFOLD avec correction
    trainloader, _ = load_data(pid, c["num_parts"], c["bs"],
                               data_hetero=c["data_hetero"],
                               partitioning=c["partitioning"],
                               alpha=c["dir_alpha"],
                               seed=c["seed"])
    t0 = time.perf_counter()
    train_loss, num_steps = train_scaffold(
        model, trainloader, epochs, lr, device, c_global_sd, c_local_sd,
        momentum=c["momentum"],
    )
    local_time_s = time.perf_counter() - t0

    # 6) Calcule c_local_new (Option II du papier) :
    #    c_new = c_old - c_global + (1/(K*lr)) * (w_global - y)
    y_sd = {name: p.detach().cpu() for name, p in model.named_parameters()
            if p.is_floating_point()}
    coef = 1.0 / max(num_steps * lr, 1e-8)
    new_c_local_sd = {}
    for name in c_local_sd:
        if name in y_sd:
            new_c_local_sd[name] = (
                c_local_sd[name] - c_global_sd[name]
                + coef * (w_global_copy[name] - y_sd[name])
            )
        else:
            new_c_local_sd[name] = c_local_sd[name]

    # 7) delta_c = c_new - c_old (pour aggregation serveur)
    delta_c_sd = {name: new_c_local_sd[name] - c_local_sd[name]
                  for name in c_local_sd}

    # 8) Upload simule (voir fedavg)
    comm_time_s, dropped = delay, 0
    if c["straggler_sim"]:
        comm_time_s, dropped_deadline = finalize_comm(
            delay, local_time_s, c["round_deadline"])
        if dropped_deadline:
            dropped = 1
            model.load_state_dict(w_global_sd)

    # 9) Sauve c_local SEULEMENT si on a vraiment participe (post-deadline).
    #    Si on a drop apres training (deadline), le serveur n'utilisera pas
    #    notre y -> garder l'ancien c_local evite que le client diverge de
    #    l'etat coherent que le serveur "voit" pour ce client.
    if not dropped:
        _save_c_local(context, new_c_local_sd, state_key)

    # 10) Eval locale (gap non-IID)
    extra = local_eval_metrics(model, trainloader, device)

    # 11) Pack y + delta_c dans un seul state_dict (prefix __dc__)
    full_state = _pack_y_and_delta_c(model, delta_c_sd)

    return make_train_reply(
        msg, full_state, train_loss, len(trainloader.dataset),
        local_time_s, pid, tier, epochs, net_tier, comm_time_s, dropped,
        extra_metrics=extra,
    )


def _pack_drop_reply(w_global_sd, zero_dc_sd):
    """Pour un drop : renvoie w inchange + delta_c=0 (pas de contribution)."""
    combined = dict(w_global_sd)
    for name, t in zero_dc_sd.items():
        combined[f"{DC_PREFIX}{name}"] = t
    return combined
