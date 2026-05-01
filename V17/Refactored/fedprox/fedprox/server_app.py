"""ServerApp FedProx (partitionnement IID ou non-IID configurable).

Identique a FedAvg cote serveur ; on broadcast 'mu' aux clients via train_config.
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training
from fl_common.strategy import FedAvgDropFilter

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    mu = float(context.run_config["mu"])
    run_federated_training(
        grid=grid,
        cfg=context.run_config,
        algo_name="FedProx",
        strategy_class=FedAvgDropFilter,
        strategy_kwargs={},
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "mu": mu, "round": r},
        project_dir_name="fedprox",
        banner_extra=f" mu={mu}",
    )
