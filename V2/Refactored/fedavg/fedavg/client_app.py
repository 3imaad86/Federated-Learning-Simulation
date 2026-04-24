"""ClientApp FedAvg (partitionnement IID ou non-IID configurable)."""

import time

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from fl_common.client_helpers import (
    compute_tier_epochs, decide_early_drop, finalize_comm,
    make_drop_reply, make_train_reply, read_common_config,
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

    # 1) Dropout PRE-training (panne reseau) : retour immediat
    net_tier, delay, dropped_early = decide_early_drop(c["straggler_sim"], pid, round_idx)
    if dropped_early:
        return make_drop_reply(msg, global_sd, pid, tier, net_tier)

    # 2) Entrainement local
    model = Net()
    model.load_state_dict(global_sd)
    device = get_device()
    model.to(device)

    trainloader, _ = load_data(pid, c["num_parts"], c["bs"],
                               data_hetero=c["data_hetero"],
                               partitioning=c["partitioning"],
                               alpha=c["dir_alpha"])
    t0 = time.perf_counter()
    train_loss, _ = train_fn(model, trainloader, epochs, lr, device)
    local_time_s = time.perf_counter() - t0

    # 3) Upload simule + check deadline (short-circuit)
    comm_time_s, dropped = 0.0, 0
    if c["straggler_sim"]:
        comm_time_s, dropped_deadline = finalize_comm(
            delay, local_time_s, c["round_deadline"])
        if dropped_deadline:
            dropped = 1
            model.load_state_dict(global_sd)

    return make_train_reply(
        msg, model.state_dict(), train_loss, len(trainloader.dataset),
        local_time_s, pid, tier, epochs, net_tier, comm_time_s, dropped,
    )
