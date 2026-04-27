"""ServerApp FedNova (Option C : rescaling 100% cote SERVEUR, fidele au papier).

Le client fait du SGD vanilla et envoie ses poids bruts + tau_i.
Le serveur calcule tau_eff a CHAQUE round (pas d'estimation a priori) :

    Delta_i = w_local_i - w_global
    tau_eff = sum (n_i/N) * tau_i              <- recalcule chaque round
    w_new   = w_global + tau_eff * sum (n_i/N) * (Delta_i / tau_i)

Le `tau_eff` reel est affiche en fin de chaque round via `extra_tail_fn`.
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training
from fl_common.strategy import FedNovaStrategy

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    run_federated_training(
        grid=grid,
        cfg=context.run_config,
        algo_name="FedNova",
        strategy_class=FedNovaStrategy,
        strategy_kwargs={},
        # Option C : aucune info FedNova-specifique a envoyer aux clients.
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="fednova",
        # Affiche tau_eff REEL (calcule au round) dans les logs
        extra_tail_fn=lambda r, strategy: f" tau_eff={strategy.last_tau_eff:.1f}",
    )
