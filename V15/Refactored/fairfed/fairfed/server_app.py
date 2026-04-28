"""ServerApp FairFed (Ezzeldin et al. 2023, adapte CIFAR-10).

Algorithme :
    p_k         = n_k / sum(n)              (poids FedAvg de base)
    F_global    = sum(p_k * F_k)            (consensus de loss)
    Delta_k     = |F_k - F_global|          (deviation par rapport au consensus)
    mean_delta  = mean(Delta_k)
    w_k         = max(0, p_k - beta * (Delta_k - mean_delta))
                  (clients pres du consensus -> upweight, outliers -> downweight)
    Normalise w_k puis aggrege.

Difference avec q-FedAvg :
    q-FedAvg : upweight les clients haute loss (aide les strugglers)
    FairFed  : upweight les clients pres du consensus (filtre les outliers)
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training
from fl_common.strategy import FairFedStrategy

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    beta = float(cfg.get("fairfed-beta", 0.5))

    run_federated_training(
        grid=grid,
        cfg=cfg,
        algo_name="FairFed",
        strategy_class=FairFedStrategy,
        strategy_kwargs={"beta": beta},
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="fairfed",
        banner_extra=f" beta={beta}",
    )
