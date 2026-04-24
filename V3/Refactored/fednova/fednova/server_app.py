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
    num_clients = int(cfg.get("num-clients", 10))
    epochs = int(cfg["local-epochs"])
    batch = int(cfg["batch-size"])
    partitioning = str(cfg.get("partitioning", "noniid"))
    dir_alpha = float(cfg.get("dirichlet-alpha", 0.3))
    tau_eff_init = compute_tau_eff_estimate(num_clients, epochs, batch, partitioning, dir_alpha)

    # On capture la strategie creee pour pouvoir lire strategy.last_tau_eff
    # dans le tail du print [round N] et du [done].
    holder = {}

    def strategy_factory(agg_train, agg_eval, frac_eval):
        s = FedNovaStrategy(
            fraction_evaluate=frac_eval,
            train_metrics_aggr_fn=agg_train,
            evaluate_metrics_aggr_fn=agg_eval,
        )
        holder["strategy"] = s
        return s

    def train_config_fn(round_idx, lr, cfg):
        # Option C : pas besoin d'envoyer tau_eff au client (calcule serveur).
        return {"lr": lr, "round": round_idx}

    def extra_tail_fn(r):
        s = holder.get("strategy")
        if s is None:
            return ""
        return f" tau_eff={s.last_tau_eff:.1f}"

    run_federated_training(
        grid=grid,
        cfg=cfg,
        algo_name="FedNova",
        strategy_factory=strategy_factory,
        train_config_fn=train_config_fn,
        project_dir_name="fednova",
        extra_tail_fn=extra_tail_fn,
        banner_extra=f" tau_eff_init={tau_eff_init:.2f}",
    )
