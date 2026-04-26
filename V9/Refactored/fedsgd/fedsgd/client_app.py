"""ClientApp FedSGD : UN seul pas SGD sur UN mini-batch (fedsgd_update)."""

import time

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from fl_common.client_helpers import (
    decide_early_drop, finalize_comm,
    local_eval_metrics, make_drop_reply, make_train_reply, read_common_config,
)
from fl_common.data import Net, get_device, load_data
from fl_common.training import fedsgd_update

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    c = read_common_config(context)
    pid = c["pid"]
    # FedSGD : pas d'heterogeneite d'epochs (1 seul pas SGD fixe).
    tier, epochs = 1, 1
    lr = msg.content["config"]["lr"]
    round_idx = int(msg.content["config"].get("round", 0))
    global_sd = msg.content["arrays"].to_torch_state_dict()

    # 1) Dropout PRE-training (panne reseau) : retour immediat
    net_tier, delay, dropped_early = decide_early_drop(
        c["straggler_sim"], pid, round_idx,
        c["comm_size_ratio"], c["sim_model_mb"])
    if dropped_early:
        return make_drop_reply(msg, global_sd, pid, tier, net_tier)

    # 2) Un seul pas SGD
    model = Net()
    model.load_state_dict(global_sd)
    device = get_device()
    model.to(device)

    trainloader, valloader = load_data(pid, c["num_parts"], c["bs"],
                                       data_hetero=c["data_hetero"],
                                       partitioning=c["partitioning"],
                                       alpha=c["dir_alpha"],
                                       seed=c["partition_seed"])
    t0 = time.perf_counter()
    train_loss, batch_examples = fedsgd_update(model, trainloader, lr, device)
    local_time_s = time.perf_counter() - t0

    # 3) Upload simule (voir fedavg pour details)
    comm_time_s, dropped = delay, 0
    if c["straggler_sim"]:
        comm_time_s, dropped_deadline = finalize_comm(
            delay, local_time_s, c["round_deadline"])
        if dropped_deadline:
            dropped = 1
            model.load_state_dict(global_sd)

    # 4) Eval locale sur la partition du client (pour voir le gap non-IID)
    extra = local_eval_metrics(model, valloader, device)

    return make_train_reply(
        msg, model.state_dict(), train_loss, batch_examples,
        local_time_s, pid, tier, epochs, net_tier, comm_time_s, dropped,
        extra_metrics=extra,
    )
