"""ServerApp q-FedAvg (Li, Sanjabi, Smith 2019).

Update :
    Delta_i        = L * (w - w_local_i)
    F_i            = loss du modele global sur la partition LOCALE du client i
    weighted_delta = F_i^q * Delta_i
    h_i            = q * F_i^(q-1) * ||Delta_i||^2 + L * F_i^q
    w_new          = w - sum(weighted_delta) / sum(h_i)

q grand -> les clients avec haute loss F_i contribuent plus -> fairness.
q = 0   -> equivalent a FedAvg.
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training
from fl_common.strategy import QFedAvgStrategy

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    q = float(cfg.get("qfedavg-q", 1.0))
    L = float(cfg.get("qfedavg-L", 1.0))

    run_federated_training(
        grid=grid,
        cfg=cfg,
        algo_name="q-FedAvg",
        strategy_class=QFedAvgStrategy,
        strategy_kwargs={"q": q, "L": L},
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="qfedavg",
        banner_extra=f" q={q} L={L}",
    )
