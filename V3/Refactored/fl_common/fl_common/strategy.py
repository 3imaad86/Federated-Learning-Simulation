"""Strategies partagees :
  - FedAvgDropFilter : FedAvg + filtrage des clients dropes (straggler-sim)
  - FedNovaStrategy  : FedNova Option C (rescaling cote serveur, fidele papier)
"""

from flwr.app import ArrayRecord
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords, aggregate_metricrecords,
)


class FedAvgDropFilter(FedAvg):
    """FedAvg simple: agregation seulement avec les clients non dropes.

    Override aggregate_train/aggregate_evaluate pour appeler les callbacks
    MEME sur liste vide (tous clients en timeout). Sinon on perdrait le
    log par-round quand le deadline serveur coupe tout le monde.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_global_arrays = None

    def configure_train(self, server_round, arrays, config, grid):
        self._last_global_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_evaluate(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        reply_contents = [msg.content for msg in valid_replies]
        # Appel UNCONDITIONNEL : meme si aucun eval reply recu, on veut
        # que le callback agg_eval tourne (pour le print par-round et
        # l'ecriture CSV). Le callback gere lui-meme le cas vide.
        return self.evaluate_metrics_aggr_fn(reply_contents, self.weighted_by_key)

    def aggregate_train(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        reply_contents = [msg.content for msg in valid_replies]

        # IMPORTANT : on appelle train_metrics_aggr_fn MEME si reply_contents
        # est vide (tous clients en timeout serveur). Sinon le callback
        # agg_train ne tournerait pas et n_timeout_round ne serait pas mis
        # a jour pour ce round.
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        if not reply_contents:
            return self._last_global_arrays, metrics

        accepted_contents = []
        for content in reply_contents:
            mr = next(iter(content.metric_records.values()))
            is_dropped = float(mr.get("dropped", 0.0)) >= 0.5
            if not is_dropped:
                accepted_contents.append(content)

        arrays = self._last_global_arrays
        if accepted_contents:
            arrays = aggregate_arrayrecords(accepted_contents, self.weighted_by_key)
        return arrays, metrics


class FedNovaStrategy(FedAvgDropFilter):
    """FedNova avec agregation cote serveur (Option C).

    Formule :
        Delta_i = w_local_i - w_global
        tau_eff = sum (n_i/N) * tau_i
        w_new   = w_global + tau_eff * sum (n_i/N) * (Delta_i / tau_i)

    Herite de FedAvgDropFilter pour recuperer l'override
    aggregate_evaluate (appel unconditionnel du callback, utile quand
    tous les clients sont en timeout). Override uniquement
    aggregate_train avec la formule FedNova.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_tau_eff = 0.0

    def aggregate_train(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        reply_contents = [msg.content for msg in valid_replies]

        # Appel UNCONDITIONNEL du callback (meme sur liste vide) pour que
        # agg_train mette a jour n_timeout_round cote server_runner.
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        if not reply_contents:
            # Aucun reply recu (tous timeout / dropes) : on garde le
            # modele global precedent.
            return self._last_global_arrays, metrics
        if self._last_global_arrays is None:
            return None, metrics

        global_sd = self._last_global_arrays.to_torch_state_dict()

        deltas, tau_is, n_is = [], [], []
        for rec in reply_contents:
            mr = next(iter(rec.metric_records.values()))
            is_dropped = float(mr.get("dropped", 0.0)) >= 0.5
            if is_dropped:
                continue
            tau_i = float(mr.get("tau_i", 0.0))
            n_i = float(mr.get("num-examples", 0.0))
            if tau_i <= 0 or n_i <= 0:
                continue
            ar = next(iter(rec.array_records.values()))
            local_sd = ar.to_torch_state_dict()
            delta = {
                k: v - global_sd[k].to(v.device)
                for k, v in local_sd.items() if v.is_floating_point()
            }
            deltas.append(delta)
            tau_is.append(tau_i)
            n_is.append(n_i)

        total_n = sum(n_is)
        if total_n == 0 or not deltas:
            # Aucun client valide pour FedNova: on conserve le modele global.
            # Cela evite d'agreger des replies "DROP" contenant les poids
            # globaux inchanges, ce qui biaiserait artificiellement le round.
            return self._last_global_arrays, metrics

        tau_eff = sum((n / total_n) * t for n, t in zip(n_is, tau_is))
        self.last_tau_eff = tau_eff

        new_sd = {}
        for k, v_global in global_sd.items():
            if v_global.is_floating_point():
                acc = None
                for i, delta in enumerate(deltas):
                    coef = (n_is[i] / total_n) / tau_is[i]
                    term = delta[k] * coef
                    acc = term if acc is None else acc + term
                new_sd[k] = v_global + tau_eff * acc
            else:
                new_sd[k] = v_global

        arrays = ArrayRecord(new_sd)
        return arrays, metrics


def compute_tau_eff_estimate(num_clients, epochs, batch, partitioning, alpha):
    """Estimation initiale de tau_eff (affichage demarrage FedNova).

    Le vrai tau_eff est calcule a chaque round par FedNovaStrategy.aggregate_train.
    """
    from .data import partition_sizes
    sizes = partition_sizes(num_clients, partitioning=partitioning, alpha=alpha)
    taus = [epochs * max(1, -(-n // batch)) for n in sizes]
    total = sum(sizes)
    if total == 0:
        return float(sum(taus) / max(1, len(taus)))
    return float(sum((n / total) * t for n, t in zip(sizes, taus)))
