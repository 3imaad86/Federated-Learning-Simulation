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

    def strategy_factory(agg_train, agg_eval, frac_eval):
        return FedAvgDropFilter(
            fraction_evaluate=frac_eval,
            train_metrics_aggr_fn=agg_train,
            evaluate_metrics_aggr_fn=agg_eval,
        )

    def train_config_fn(round_idx, lr, cfg):
        return {"lr": lr, "mu": mu, "round": round_idx}

    run_federated_training(
        grid=grid,
        cfg=context.run_config,
        algo_name="FedProx",
        strategy_factory=strategy_factory,
        train_config_fn=train_config_fn,
        project_dir_name="fedprox",
        banner_extra=f" mu={mu}",
    )
