"""Fonctions communes aux ClientApp Flower.

Chaque algorithme garde un petit fichier `client_app.py`, et ce module
contient le flux commun : lecture de config, entrainement, evaluation et
simulation simple des stragglers.
"""

import time

from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict

from .data import Net, get_device, load_data, model_size_bytes
from .straggler import simulate_comm_delay
from .training import fedsgd_update, test_with_class_accuracies
from .training import train as train_fn


def read_common_config(context):
    """Extrait les params communs a tous les client_app du run_config."""
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
    if epochs_hetero:
        tier = pid % 3
        if tier == 0:
            epochs = 1
        elif tier == 1:
            epochs = 3
        else:
            epochs = 6
    else:
        tier = 1
        epochs = base_epochs
    return tier, epochs


def decide_early_drop(straggler_sim, pid, round_idx):
    """Decide AVANT entrainement si le client est drope (panne reseau).

    Returns:
        (net_tier, delay_or_None, dropped_early)
          - dropped_early=True  : skipper l'entrainement, retour instantane
          - dropped_early=False : `delay` secondes a attendre pour l'upload
    """
    if not straggler_sim:
        return 1, 0.0, False
    model_mb = model_size_bytes() / (1024.0 * 1024.0)
    net_tier, delay = simulate_comm_delay(pid, model_mb, round_idx)
    if delay is None:
        return net_tier, None, True
    return net_tier, delay, False


def make_drop_reply(msg, global_sd, pid, tier, net_tier, extra_metrics=None):
    """Construit un reply rapide (sans entrainement) pour un client drope.

    Le reply contient :
      - arrays  = poids globaux inchanges (filtres par la strategy)
      - metrics : dropped=1, num-examples=0, local_time_s=0

    Comme ce retour arrive immediatement (pas de train, pas de sleep),
    le serveur n'est PAS bloque par ce client : round_time reflete
    uniquement les clients qui ont reellement travaille.
    """
    m = {
        "train_loss": 0.0,
        "num-examples": 0,
        "local_time_s": 0.0,
        "partition_id": float(pid),
        "epochs_used": 0.0,
        "resource_tier": float(tier),
        "net_tier": float(net_tier),
        "comm_time_s": 0.0,
        "dropped": 1.0,
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
    """Simule l'upload post-entrainement avec short-circuit deadline.

    Si `local_time_s + delay > round_deadline` on sait deja que la
    reponse arrivera hors deadline : inutile de simuler l'upload,
    on retourne immediatement dropped_deadline=True.

    Returns:
        (comm_time_s, dropped_deadline)
    """
    if round_deadline > 0 and (local_time_s + delay) > round_deadline:
        return 0.0, True
    time.sleep(delay)
    return delay, False


def train_client(msg, context, algo):
    """Execute un round local pour FedAvg, FedProx, FedNova ou FedSGD."""
    c = read_common_config(context)
    pid = c["pid"]
    lr = msg.content["config"]["lr"]
    round_idx = int(msg.content["config"].get("round", 0))
    global_sd = msg.content["arrays"].to_torch_state_dict()

    if algo == "fedsgd":
        tier, epochs = 1, 1
    else:
        tier, epochs = compute_tier_epochs(pid, c["base_epochs"], c["epochs_hetero"])

    net_tier, delay, dropped_early = decide_early_drop(
        c["straggler_sim"], pid, round_idx
    )
    if dropped_early:
        extra = {"tau_i": 0.0} if algo == "fednova" else None
        return make_drop_reply(msg, global_sd, pid, tier, net_tier, extra)

    model = Net()
    model.load_state_dict(global_sd)
    device = get_device()
    model.to(device)

    trainloader, _ = load_data(
        pid, c["num_parts"], c["bs"],
        data_hetero=c["data_hetero"],
        partitioning=c["partitioning"],
        alpha=c["dir_alpha"],
    )

    t0 = time.perf_counter()
    tau_i = 0.0
    if algo == "fedsgd":
        train_loss, _ = fedsgd_update(model, trainloader, lr, device)
    else:
        mu = float(msg.content["config"].get("mu", 0.0)) if algo == "fedprox" else 0.0
        global_params = None
        if algo == "fedprox":
            global_params = [p.detach().clone() for p in model.parameters()]
        train_loss, tau_i = train_fn(
            model, trainloader, epochs, lr, device, mu=mu, global_params=global_params
        )
    local_time_s = time.perf_counter() - t0

    comm_time_s, dropped = 0.0, 0
    if c["straggler_sim"]:
        comm_time_s, dropped_deadline = finalize_comm(
            delay, local_time_s, c["round_deadline"]
        )
        if dropped_deadline:
            dropped = 1
            model.load_state_dict(global_sd)
            tau_i = 0.0

    metrics = {
        "train_loss": float(train_loss),
        "num-examples": len(trainloader.dataset),
        "local_time_s": float(local_time_s),
        "partition_id": float(pid),
        "epochs_used": float(epochs),
        "resource_tier": float(tier),
        "net_tier": float(net_tier),
        "comm_time_s": float(comm_time_s),
        "dropped": float(dropped),
    }
    if algo == "fednova":
        metrics["tau_i"] = float(tau_i)

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(metrics),
        }),
        reply_to=msg,
    )


def evaluate_client(msg, context):
    """Evaluation locale commune a tous les algorithmes."""
    pid = context.node_config["partition-id"]
    num_parts = context.node_config["num-partitions"]
    cfg = context.run_config

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    _, valloader = load_data(
        pid, num_parts, cfg["batch-size"],
        partitioning=str(cfg.get("partitioning", "noniid")),
        alpha=float(cfg.get("dirichlet-alpha", 0.3)),
    )
    loss, accuracy, class_accs, macro_recall, macro_f1 = test_with_class_accuracies(
        model, valloader, device
    )

    metrics = MetricRecord({
        "loss": float(loss),
        "accuracy": float(accuracy),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "class_accuracies": [float(a) for a in class_accs],
        "num-examples": len(valloader.dataset),
        "partition_id": float(pid),
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
