"""ServerApp FedNova (Option C : rescaling cote SERVEUR, fidele au papier).

Formule appliquee dans FedNovaStrategy.aggregate_train :
    Delta_i = w_local_i - w_global
    tau_eff = sum (n_i/N) * tau_i
    w_new   = w_global + tau_eff * sum (n_i/N) * (Delta_i / tau_i)
"""

from flwr.app import Context
from flwr.serverapp import Grid, ServerApp

from fl_common.server_runner import run_federated_training
from fl_common.strategy import FedNovaStrategy, compute_tau_eff_estimate

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    tau_eff_init = compute_tau_eff_estimate(
        int(cfg.get("num-clients", 10)),
        int(cfg["local-epochs"]),
        int(cfg["batch-size"]),
        str(cfg.get("partitioning", "noniid")),
        float(cfg.get("dirichlet-alpha", 0.3)),
    )

    run_federated_training(
        grid=grid,
        cfg=cfg,
        algo_name="FedNova",
        strategy_class=FedNovaStrategy,
        strategy_kwargs={},
        # Option C : pas besoin d'envoyer tau_eff au client (calcule serveur).
        train_config_fn=lambda r, lr, cfg: {"lr": lr, "round": r},
        project_dir_name="fednova",
        extra_tail_fn=lambda r, strategy: f" tau_eff={strategy.last_tau_eff:.1f}",
        banner_extra=f" tau_eff_init={tau_eff_init:.2f}",
    )
