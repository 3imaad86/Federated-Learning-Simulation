"""Strategies partagees.

  * FedAvgDropFilter : FedAvg + filtre des clients `dropped=1` hors agregation.
    Appelle aussi agg_train MEME sur liste vide (log par-round pour un round
    dont tous les clients ont timeout).

  * FedNovaStrategy  : FedNova Option C (rescaling cote serveur).
        Delta_i = w_local_i - w_global
        tau_eff = sum (n_i/N) * tau_i
        w_new   = w_global + tau_eff * sum (n_i/N) * (Delta_i / tau_i)
"""

from flwr.app import ArrayRecord
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords


class FedAvgDropFilter(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_global_arrays = None

    def configure_train(self, server_round, arrays, config, grid):
        self._last_global_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        contents = [msg.content for msg in valid]
        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)

        if not contents:
            return self._last_global_arrays, metrics

        # Filtre les clients dropes (ils renvoient les poids globaux
        # inchanges mais ne doivent pas participer a l'agregation).
        accepted = [
            c for c in contents
            if float(next(iter(c.metric_records.values())).get("dropped", 0.0)) < 0.5
        ]
        arrays = self._last_global_arrays
        if accepted:
            arrays = aggregate_arrayrecords(accepted, self.weighted_by_key)
        return arrays, metrics


class FedNovaStrategy(FedAvgDropFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_tau_eff = 0.0

    def aggregate_train(self, server_round, replies):
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        contents = [msg.content for msg in valid]
        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)

        if not contents:
            return self._last_global_arrays, metrics
        if self._last_global_arrays is None:
            return None, metrics

        global_sd = self._last_global_arrays.to_torch_state_dict()

        deltas, taus, ns = [], [], []
        for rec in contents:
            ar = next(iter(rec.array_records.values()))
            mr = next(iter(rec.metric_records.values()))
            tau_i = float(mr.get("tau_i", 0.0))
            n_i = float(mr.get("num-examples", 0.0))
            if tau_i <= 0 or n_i <= 0:
                continue
            local_sd = ar.to_torch_state_dict()
            delta = {
                k: v - global_sd[k].to(v.device)
                for k, v in local_sd.items() if v.is_floating_point()
            }
            deltas.append(delta)
            taus.append(tau_i)
            ns.append(n_i)

        total_n = sum(ns)
        if total_n == 0 or not deltas:
            # Fallback FedAvg si aucun tau_i/n_i valide
            arrays = aggregate_arrayrecords(contents, self.weighted_by_key)
            return arrays, metrics

        tau_eff = sum((n / total_n) * t for n, t in zip(ns, taus))
        self.last_tau_eff = tau_eff

        new_sd = {}
        for k, v_global in global_sd.items():
            if not v_global.is_floating_point():
                new_sd[k] = v_global
                continue
            acc = None
            for i, delta in enumerate(deltas):
                coef = (ns[i] / total_n) / taus[i]
                term = delta[k] * coef
                acc = term if acc is None else acc + term
            new_sd[k] = v_global + tau_eff * acc

        return ArrayRecord(new_sd), metrics


def compute_tau_eff_estimate(num_clients, epochs, batch, partitioning, alpha):
    """Estimation initiale de tau_eff (affichage demarrage FedNova).

    Le vrai tau_eff est calcule a chaque round par FedNovaStrategy.
    """
    from .data import partition_sizes
    sizes = partition_sizes(num_clients, partitioning=partitioning, alpha=alpha)
    taus = [epochs * max(1, -(-n // batch)) for n in sizes]
    total = sum(sizes)
    if total == 0:
        return float(sum(taus) / max(1, len(taus)))
    return float(sum((n / total) * t for n, t in zip(sizes, taus)))
