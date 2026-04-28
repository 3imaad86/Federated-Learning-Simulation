"""ClientApp q-FedAvg.

Specifite : avant l'entrainement local, le client calcule F_k(w) =
loss du modele GLOBAL sur la partition LOCALE. Cette valeur est envoyee
au serveur via le metric "f_k" pour pondererr l'agregation.
"""

import time

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from fl_common.client_helpers import (
    compute_tier_epochs, decide_early_drop, finalize_comm,
    local_eval_metrics, make_drop_reply, make_train_reply, read_common_config,
)
from fl_common.data import Net, get_device, load_data, set_seed
from fl_common.training import test as test_fn, train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    c = read_common_config(context)
    pid = c["pid"]
    if c["seed"] >= 0:
        set_seed(c["seed"] + pid)
    tier, epochs = compute_tier_epochs(pid, c["base_epochs"], c["epochs_hetero"])
    lr = msg.content["config"]["lr"]
    round_idx = int(msg.content["config"].get("round", 0))
    global_sd = msg.content["arrays"].to_torch_state_dict()

    # 1) Dropout PRE-training (panne reseau)
    net_tier, delay, dropped_early = decide_early_drop(
        c["straggler_sim"], pid, round_idx,
        c["comm_size_ratio"], c["sim_model_mb"])
    if dropped_early:
        return make_drop_reply(msg, global_sd, pid, tier, net_tier,
                               extra_metrics={"f_k": 0.0})

    # 2) Setup
    model = Net()
    model.load_state_dict(global_sd)
    device = get_device()
    model.to(device)

    trainloader, _ = load_data(pid, c["num_parts"], c["bs"],
                               data_hetero=c["data_hetero"],
                               partitioning=c["partitioning"],
                               alpha=c["dir_alpha"],
                               seed=c["seed"])

    # 3) IMPORTANT q-FedAvg : F_k(w) = loss du modele GLOBAL sur partition locale
    #    AVANT l'entrainement (forward pass uniquement, sans grad).
    f_k, _ = test_fn(model, trainloader, device)

    # 4) Train standard
    t0 = time.perf_counter()
    train_loss, _ = train_fn(model, trainloader, epochs, lr, device,
                             momentum=c["momentum"])
    local_time_s = time.perf_counter() - t0

    # 5) Upload simule + check deadline
    comm_time_s, dropped = delay, 0
    if c["straggler_sim"]:
        comm_time_s, dropped_deadline = finalize_comm(
            delay, local_time_s, c["round_deadline"])
        if dropped_deadline:
            dropped = 1
            model.load_state_dict(global_sd)

    # 6) Eval locale (visualiser le gap non-IID) + envoi de f_k au serveur
    extra = local_eval_metrics(model, trainloader, device)
    extra["f_k"] = float(f_k)

    return make_train_reply(
        msg, model.state_dict(), train_loss, len(trainloader.dataset),
        local_time_s, pid, tier, epochs, net_tier, comm_time_s, dropped,
        extra_metrics=extra,
    )
