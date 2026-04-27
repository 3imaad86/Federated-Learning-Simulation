"""Strategies partagees.

  * FedAvgDropFilter : FedAvg + filtre des clients dropped=1 hors agregation.
  * FedNovaStrategy  : FedNova Option C (rescaling cote serveur).
  * ScaffoldStrategy : SCAFFOLD avec control variates (Karimireddy et al. 2020).
"""

import torch
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


def _array_record_bytes(arrays):
    """Taille (octets) d'un ArrayRecord = somme numel * element_size des tensors."""
    sd = arrays.to_torch_state_dict()
    return sum(t.numel() * t.element_size() for t in sd.values())


def _content_bytes(content):
    """Taille totale des ArrayRecords dans un content (somme sur tous les arrays)."""
    return sum(_array_record_bytes(ar) for ar in content.array_records.values())


class FedAvgDropFilter(FedAvg):
    """FedAvg qui ignore les contents avec dropped=1 dans l'agregation.

    Mesure aussi les bytes reels transmis chaque round (downlink + uplink).
    Permet aux strategies enfant (SCAFFOLD) d'avoir leur cout reflete
    automatiquement, sans facteur arbitraire.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_global_arrays = None
        # Bytes reels transmis au dernier round (server <-> tous les clients).
        self.last_downlink_bytes = 0
        self.last_uplink_bytes = 0

    def configure_train(self, server_round, arrays, config, grid):
        # Garde une ref aux poids globaux (fallback drop / no replies)
        self._last_global_arrays = arrays
        msgs = super().configure_train(server_round, arrays, config, grid)
        # Mesure downlink : taille du payload * nombre de messages envoyes
        self.last_downlink_bytes = _array_record_bytes(arrays) * len(msgs)
        return msgs

    def aggregate_train(self, server_round, replies):
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        contents = [msg.content for msg in valid]
        # Mesure uplink : somme des bytes recus (incluant les drops, qui transmettent
        # quand meme leur reply au serveur)
        self.last_uplink_bytes = sum(_content_bytes(c) for c in contents)

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
        # Mesure uplink (bytes reels)
        self.last_uplink_bytes = sum(_content_bytes(c) for c in contents)

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


# ============================================================================
# 3) SCAFFOLD : control variates pour reduction de variance non-IID
# ============================================================================

# Prefixes pour packer plusieurs arrays dans un seul ArrayRecord :
#   __cg__ = control global (server -> client)
#   __dc__ = delta control  (client -> server)
CG_PREFIX = "__cg__"
DC_PREFIX = "__dc__"


class ScaffoldStrategy(FedAvgDropFilter):
    """SCAFFOLD : FedAvg + control variates (Karimireddy et al. 2020).

    Le serveur maintient w_global (gere par parent) et c_global.
    Chaque client maintient son c_local persistant via context.state.

    Communication packee dans un seul ArrayRecord (pour eviter de modifier
    la signature de FedAvg) :
      - Server -> Client : {w_keys}    + {__cg__ + w_keys} pour c_global
      - Client -> Server : {y_keys}    + {__dc__ + w_keys} pour delta_c

    Update serveur a chaque round :
      w_new = FedAvg(y_i)              (moyenne ponderee par num-examples)
      c_new = c_global + mean(delta_c) (moyenne ponderee aussi)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._c_global_sd = None     # init au 1er configure_train
        self._w_global_arrays = None  # garde la version SANS c_global (pour fallback)

    def _init_c_global(self, w_state_dict):
        """Initialise c_global a zeros (meme shape et type que les params)."""
        self._c_global_sd = {
            name: torch.zeros_like(t)
            for name, t in w_state_dict.items()
            if t.is_floating_point()
        }

    def configure_train(self, server_round, arrays, config, grid):
        """Pack w_global et c_global dans un seul ArrayRecord."""
        w_sd = arrays.to_torch_state_dict()
        if self._c_global_sd is None:
            self._init_c_global(w_sd)
        self._w_global_arrays = arrays  # version "pure" pour fallback

        combined = dict(w_sd)
        for name, t in self._c_global_sd.items():
            combined[f"{CG_PREFIX}{name}"] = t
        return super().configure_train(server_round, ArrayRecord(combined), config, grid)

    def aggregate_train(self, server_round, replies):
        """Aggregation: FedAvg sur w + maj de c_global avec mean(delta_c)."""
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        contents = [msg.content for msg in valid]
        # Mesure uplink (bytes reels). Pour SCAFFOLD chaque content fait 2x
        # la taille du modele (y + delta_c packes).
        self.last_uplink_bytes = sum(_content_bytes(c) for c in contents)

        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)

        if not contents:
            return self._w_global_arrays, metrics

        accepted = [c for c in contents if not _is_dropped(c)]
        if not accepted:
            return self._w_global_arrays, metrics

        # Aggregation des arrays packes (FedAvg-style sur tout)
        agg = aggregate_arrayrecords(accepted, self.weighted_by_key)
        agg_sd = agg.to_torch_state_dict()

        # Split: w (sans prefix) et delta_c (prefixe __dc__)
        new_w_sd = {k: v for k, v in agg_sd.items() if not k.startswith(DC_PREFIX)}
        delta_c_avg = {k[len(DC_PREFIX):]: v
                       for k, v in agg_sd.items() if k.startswith(DC_PREFIX)}

        # Update c_global : c_new = c_old + mean(delta_c_i)
        for name in self._c_global_sd:
            if name in delta_c_avg:
                self._c_global_sd[name] = (
                    self._c_global_sd[name]
                    + delta_c_avg[name].to(self._c_global_sd[name].device)
                )

        return ArrayRecord(new_w_sd), metrics
