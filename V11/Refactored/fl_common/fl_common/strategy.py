"""Strategies partagees.

  * FedAvgDropFilter : FedAvg + filtre des clients dropped=1 hors agregation.
  * FedNovaStrategy  : FedNova Option C (rescaling cote serveur).

Formule FedNova :
    Delta_i = w_local_i - w_global       (mise-a-jour locale)
    tau_eff = sum (n_i / N) * tau_i      (norm. effective)
    w_new   = w_global + tau_eff * sum (n_i / N) * (Delta_i / tau_i)
"""

from flwr.app import ArrayRecord
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords


# ============================================================================
# 1) FedAvg + filtre des clients dropes
# ============================================================================

def _is_dropped(content):
    """True si le client a renvoye un reply 'drop' (panne/deadline)."""
    mr = next(iter(content.metric_records.values()))
    return float(mr.get("dropped", 0.0)) >= 0.5


class FedAvgDropFilter(FedAvg):
    """FedAvg qui ignore les contents avec dropped=1 dans l'agregation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_global_arrays = None

    def configure_train(self, server_round, arrays, config, grid):
        # On garde une ref aux poids globaux pour les fallback (drop / no replies)
        self._last_global_arrays = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        contents = [msg.content for msg in valid]
        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)

        if not contents:
            return self._last_global_arrays, metrics

        accepted = [c for c in contents if not _is_dropped(c)]
        arrays = (aggregate_arrayrecords(accepted, self.weighted_by_key)
                  if accepted else self._last_global_arrays)
        return arrays, metrics


# ============================================================================
# 2) FedNova : extraction des deltas + rescaling tau_eff
# ============================================================================

def _extract_delta_tau_n(content, global_sd):
    """Pour un client : retourne (delta_dict, tau_i, n_i) ou None si invalide."""
    ar = next(iter(content.array_records.values()))
    mr = next(iter(content.metric_records.values()))
    tau_i = float(mr.get("tau_i", 0.0))
    n_i = float(mr.get("num-examples", 0.0))
    if tau_i <= 0 or n_i <= 0:
        return None
    local_sd = ar.to_torch_state_dict()
    delta = {
        k: v - global_sd[k].to(v.device)
        for k, v in local_sd.items() if v.is_floating_point()
    }
    return delta, tau_i, n_i


def _apply_fednova_update(global_sd, deltas, taus, ns):
    """Calcule w_new = w_global + tau_eff * sum (n_i/N) * (Delta_i / tau_i).

    Retourne (new_state_dict, tau_eff).
    """
    total_n = sum(ns)
    tau_eff = sum((n / total_n) * t for n, t in zip(ns, taus))

    new_sd = {}
    for k, w_global in global_sd.items():
        if not w_global.is_floating_point():
            new_sd[k] = w_global
            continue
        # acc = sum_i (n_i/N) * (Delta_i[k] / tau_i)
        acc = None
        for delta, tau_i, n_i in zip(deltas, taus, ns):
            term = delta[k] * ((n_i / total_n) / tau_i)
            acc = term if acc is None else acc + term
        new_sd[k] = w_global + tau_eff * acc
    return new_sd, tau_eff


class FedNovaStrategy(FedAvgDropFilter):
    """FedNova Option C : tau_eff calcule + applique cote serveur."""

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

        # Extrait deltas/tau/n des clients valides (ignore les drops + tau<=0)
        triples = [_extract_delta_tau_n(c, global_sd) for c in contents]
        triples = [t for t in triples if t is not None]

        if not triples:
            # Fallback FedAvg si aucun reply utilisable
            return aggregate_arrayrecords(contents, self.weighted_by_key), metrics

        deltas, taus, ns = zip(*triples)
        new_sd, tau_eff = _apply_fednova_update(global_sd, deltas, taus, ns)
        self.last_tau_eff = tau_eff
        return ArrayRecord(new_sd), metrics
