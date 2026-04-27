"""ServerApp SCAFFOLD (Karimireddy et al. 2020).

Le serveur maintient :
  - w_global   : modele (gere par parent FedAvgDropFilter)
  - c_global   : control variate global, init a zeros, recalcule chaque round

A chaque round :
  - Pack (w_global, c_global) dans un seul ArrayRecord (prefixes __cg__).
  - Aggrege les replies : nouveau w via FedAvg, c_new = c_global + mean(delta_c).
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training
from fl_common.strategy import ScaffoldStrategy

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    run_federated_training(
        grid=grid,
        cfg=context.run_config,
        algo_name="SCAFFOLD",
        strategy_class=ScaffoldStrategy,
        strategy_kwargs={},
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="scaffold",
    )
