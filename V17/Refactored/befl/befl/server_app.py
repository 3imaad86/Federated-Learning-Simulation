"""ServerApp BEFL.

Le serveur reste FedAvg standard. Toute la logique BEFL est cote client :
  - Chaque client adapte ses epochs selon son niveau de batterie
  - Si batterie morte, il renvoie un drop reply (filtre par FedAvgDropFilter)

Donc rien de specifique a faire ici, juste reutiliser FedAvgDropFilter.
"""

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
        algo_name="BEFL",
        strategy_class=FedAvgDropFilter,
        strategy_kwargs={},
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="befl",
    )
