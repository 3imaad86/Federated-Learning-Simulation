"""ClientApp FedNova (Option C : rescaling SERVEUR, fidele au papier).

Le client fait un SGD standard (pas de rescaling cote client) et renvoie
ses poids locaux bruts + son tau_i. Le serveur applique la formule FedNova
a l'agregation.
"""

import time

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from fl_common.client_helpers import (
    compute_tier_epochs, decide_early_drop, finalize_comm,
    local_eval_metrics, make_drop_reply, make_train_reply, read_common_config,
)
from fl_common.data import Net, get_device, load_data
from fl_common.training import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    c = read_common_config(context)
    pid = c["pid"]
    tier, epochs = compute_tier_epochs(pid, c["base_epochs"], c["epochs_hetero"])
    lr = msg.content["config"]["lr"]
    round_idx = int(msg.content["config"].get("round", 0))
    global_sd = msg.content["arrays"].to_torch_state_dict()

    # 1) Dropout PRE-training : tau_i=0 pour que la strategy ignore ce reply
    net_tier, delay, dropped_early = decide_early_drop(
        c["straggler_sim"], pid, round_idx, c["comm_size_ratio"])
    if dropped_early:
        return make_drop_reply(msg, global_sd, pid, tier, net_tier,
                               extra_metrics={"tau_i": 0.0})

    # 2) Entrainement SGD standard (rescaling fait cote serveur)
    model = Net()
    model.load_state_dict(global_sd)
    device = get_device()
    model.to(device)

    trainloader, _ = load_data(pid, c["num_parts"], c["bs"],
                               data_hetero=c["data_hetero"],
                               partitioning=c["partitioning"],
                               alpha=c["dir_alpha"])
    t0 = time.perf_counter()
    train_loss, tau_i = train_fn(model, trainloader, epochs, lr, device)
    local_time_s = time.perf_counter() - t0

    # 3) Upload simule (voir fedavg pour details)
    comm_time_s, dropped = delay, 0
    if c["straggler_sim"]:
        comm_time_s, dropped_deadline = finalize_comm(
            delay, local_time_s, c["round_deadline"])
        if dropped_deadline:
            dropped = 1
            model.load_state_dict(global_sd)
            tau_i = 0.0

    # 4) Eval locale sur la partition du client (pour voir le gap non-IID)
    extra = local_eval_metrics(model, trainloader, device)
    extra["tau_i"] = float(tau_i)

    return make_train_reply(
        msg, model.state_dict(), train_loss, len(trainloader.dataset),
        local_time_s, pid, tier, epochs, net_tier, comm_time_s, dropped,
        extra_metrics=extra,
    )
