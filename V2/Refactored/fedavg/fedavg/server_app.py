"""ServerApp FedAvg (partitionnement IID ou non-IID configurable)."""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training
from fl_common.strategy import FedAvgDropFilter

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    run_federated_training(
        grid=grid,
        cfg=context.run_config,
        algo_name="FedAvg",
        strategy_class=FedAvgDropFilter,
        strategy_kwargs={},
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="fedavg",
    )
