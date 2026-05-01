"""Strategies partagees.

  * FedAvgDropFilter : FedAvg + filtre des clients dropped=1 hors agregation.
  * FedNovaStrategy  : FedNova Option C (rescaling cote serveur).
  * ScaffoldStrategy : SCAFFOLD avec control variates (Karimireddy et al. 2020).
  * QFedAvgStrategy  : q-FedAvg = fair FL pondere par loss^q (Li et al. 2019).
  * FairFedStrategy  : FairFed = upweight les clients pres du consensus
                       (Ezzeldin et al. 2023, adapte CIFAR-10).
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

    Update serveur a chaque round (papier Algo 1) :
      w_new = FedAvg(y_i)                     (ponderee par num-examples)
      c_new = c_global + (|S|/N) * mean(Δc_i) (moyenne UNIFORME, facteur |S|/N)

    Differences avec une implementation naive :
      1. Δc agrege en moyenne UNIFORME (pas pondere par num-examples).
      2. Facteur |S|/N applique a la maj de c_global (Karimireddy et al.
         2020, Algo 1 ligne 17). Sans ce facteur c_global derive trop vite
         en participation partielle.
    """

    def __init__(self, *args, num_clients_total=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._c_global_sd = None     # init au 1er configure_train
        self._w_global_arrays = None  # garde la version SANS c_global (pour fallback)
        # Total clients dans la federation (pour le facteur |S|/N). Si None,
        # on utilise |S| -> scale=1 (comme si tous participent toujours).
        self.num_clients_total = num_clients_total

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

    @staticmethod
    def _uniform_mean_delta_c(contents):
        """Moyenne UNIFORME (1/|S|) des Δc_i extraits des contents packes.

        Conforme au papier SCAFFOLD : la maj de c_global utilise une moyenne
        arithmetique simple, PAS une moyenne ponderee par num-examples.
        """
        dc_sds = []
        for c in contents:
            ar = next(iter(c.array_records.values()))
            full_sd = ar.to_torch_state_dict()
            dc_sd = {k[len(DC_PREFIX):]: v for k, v in full_sd.items()
                     if k.startswith(DC_PREFIX)}
            if dc_sd:
                dc_sds.append(dc_sd)
        if not dc_sds:
            return None
        n = len(dc_sds)
        keys = dc_sds[0].keys()
        return {name: sum(dc[name] for dc in dc_sds) / n for name in keys}

    def aggregate_train(self, server_round, replies):
        """Aggregation: FedAvg sur w + maj de c_global avec (|S|/N)*mean_uniform(Δc)."""
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

        # 1) Aggregation des y : FedAvg pondere par num-examples (sur les keys
        #    sans prefix). Les keys __dc__ sont aussi agregees ici mais on ne
        #    les utilise PAS (Δc se moyenne uniformement, pas par num-examples).
        agg = aggregate_arrayrecords(accepted, self.weighted_by_key)
        agg_sd = agg.to_torch_state_dict()
        new_w_sd = {k: v for k, v in agg_sd.items() if not k.startswith(DC_PREFIX)}

        # 2) Moyenne UNIFORME des Δc (papier SCAFFOLD).
        delta_c_uniform = self._uniform_mean_delta_c(accepted)

        # 3) Update c_global : c_new = c_old + (|S|/N) * mean_uniform(Δc_i)
        #    Le facteur |S|/N (Karimireddy et al. 2020, Algo 1 ligne 17) est
        #    crucial pour la participation partielle : sans lui c_global
        #    derive trop vite quand seule une fraction des clients repond.
        if delta_c_uniform is not None:
            n_active = len(accepted)
            n_total = max(self.num_clients_total or n_active, 1)
            scale = n_active / n_total
            for name in self._c_global_sd:
                if name in delta_c_uniform:
                    self._c_global_sd[name] = (
                        self._c_global_sd[name]
                        + scale * delta_c_uniform[name].to(
                            self._c_global_sd[name].device)
                    )

        return ArrayRecord(new_w_sd), metrics


# ============================================================================
# 4) q-FedAvg : Fair FL pondere par loss^q (Li, Sanjabi, Smith 2019)
# ============================================================================

def _qfedavg_terms(content, global_sd, q, L):
    """Pour un client : compute (weighted_delta, h_k) pour q-FedAvg.

    Soit F_i la loss du modele global sur la partition locale du client i :
        Delta_i        = L * (w_global - w_local_i)
        weighted_delta = F_i^q * Delta_i
        h_i            = q * F_i^(q-1) * ||Delta_i||^2 + L * F_i^q

    Retourne (weighted_delta_dict, h_i_scalar) ou None si invalide.

    Robustesse :
      - On rejette uniquement les valeurs invalides (NaN, negatives).
      - On applique un PLANCHER 1e-8 sur F_i au lieu d'exclure les clients
        bien fittes (F_i ~ 0). Sinon ces clients sortiraient silencieusement
        de l'agregation, perdant leur contribution au modele.
    """
    ar = next(iter(content.array_records.values()))
    mr = next(iter(content.metric_records.values()))
    f_i = float(mr.get("f_k", 0.0))
    # Rejet des valeurs invalides (NaN ou negatives -- impossible pour une CE)
    if f_i != f_i or f_i < 0.0:
        return None
    # Plancher numerique : evite l'exclusion des clients bien fittes et la
    # division par 0 dans F_i^(q-1) quand q < 1.
    f_i = max(f_i, 1e-8)

    local_sd = ar.to_torch_state_dict()
    # Delta_i = L * (w - w_local) sur params float
    delta = {}
    norm_sq = 0.0
    for name, w_g in global_sd.items():
        if not w_g.is_floating_point():
            continue
        d = L * (w_g - local_sd[name].to(w_g.device))
        delta[name] = d
        norm_sq += float((d * d).sum().item())

    # weighted_delta = F_i^q * Delta_i
    f_q = f_i ** q
    weighted_delta = {name: f_q * d for name, d in delta.items()}

    # h_i = q * F_i^(q-1) * ||Delta||^2 + L * F_i^q
    if q > 0:
        h_i = q * (f_i ** (q - 1.0)) * norm_sq + L * f_q
    else:
        h_i = L  # cas q=0 (= FedAvg)
    return weighted_delta, h_i


def _apply_qfedavg_update(global_sd, weighted_deltas, h_total):
    """w_new = w - (1 / sum(h_i)) * sum_i (weighted_delta_i)."""
    new_sd = {}
    for name, w_g in global_sd.items():
        if not w_g.is_floating_point():
            new_sd[name] = w_g
            continue
        # acc = sum_i weighted_delta_i[name]
        acc = None
        for wd in weighted_deltas:
            acc = wd[name] if acc is None else acc + wd[name]
        new_sd[name] = w_g - acc / h_total
    return new_sd


class QFedAvgStrategy(FedAvgDropFilter):
    """q-FedAvg : Fair Federated Learning (Li, Sanjabi, Smith 2019).

    Au lieu de FedAvg qui pondere par n_i (taille des donnees), q-FedAvg
    pondere par F_i^q ou F_i = loss du client AVANT training local.

    Plus q est grand, plus on favorise les clients qui souffrent (haute loss)
    -> reduction de la variance d'accuracy entre clients.

      q = 0   : equivalent a FedAvg (pas de fairness)
      q = 1   : standard q-FedAvg
      q = 5   : forte fairness (parfois lent a converger)
    """

    def __init__(self, q=1.0, L=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = float(q)
        self.L = float(L)

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

        accepted = [c for c in contents if not _is_dropped(c)]
        if not accepted:
            return self._last_global_arrays, metrics

        global_sd = self._last_global_arrays.to_torch_state_dict()

        # Compute (weighted_delta, h_i) pour chaque client
        pairs = [_qfedavg_terms(c, global_sd, self.q, self.L) for c in accepted]
        pairs = [p for p in pairs if p is not None]

        if not pairs:
            # Fallback FedAvg si rien d'utilisable
            return aggregate_arrayrecords(accepted, self.weighted_by_key), metrics

        weighted_deltas, hs = zip(*pairs)
        h_total = sum(hs)
        if h_total <= 0:
            return aggregate_arrayrecords(accepted, self.weighted_by_key), metrics

        new_sd = _apply_qfedavg_update(global_sd, weighted_deltas, h_total)
        return ArrayRecord(new_sd), metrics


# ============================================================================
# 5) FairFed : pondere par proximite au consensus (Ezzeldin et al. 2023)
# ============================================================================

def _fairfed_weights(infos, beta):
    """Calcule les poids FairFed a partir des (f_k, n_k) des clients.

      p_k         = n_k / sum(n)            (poids FedAvg de base)
      F_global    = sum(p_k * F_k)          (consensus)
      Delta_k     = |F_k - F_global|        (deviation)
      mean_delta  = mean(Delta_k)
      w_k_brut    = p_k - beta * (Delta_k - mean_delta)
      w_k_final   = max(0, w_k_brut) puis normalise

    `infos` : liste de tuples (f_k, n_k).
    Retourne la liste de poids normalises (somme = 1) ou None si invalide.
    """
    total_n = sum(n for _, n in infos)
    if total_n <= 0:
        return None
    f_global = sum(f * n for f, n in infos) / total_n
    deltas = [abs(f - f_global) for f, _ in infos]
    mean_delta = sum(deltas) / len(deltas)

    raw = [max(0.0, (n / total_n) - beta * (d - mean_delta))
           for (_, n), d in zip(infos, deltas)]
    s = sum(raw)
    if s <= 0:
        return None
    return [w / s for w in raw]


def _aggregate_with_weights(global_sd, local_sds, weights):
    """Aggregation manuelle : new_w = sum_k weights[k] * local_sd[k]."""
    new_sd = {}
    for name, w_g in global_sd.items():
        if not w_g.is_floating_point():
            new_sd[name] = w_g
            continue
        acc = None
        for sd, weight in zip(local_sds, weights):
            term = weight * sd[name].to(w_g.device)
            acc = term if acc is None else acc + term
        new_sd[name] = acc
    return new_sd


class FairFedStrategy(FedAvgDropFilter):
    """FairFed : pondere par proximite au consensus.

    Chaque client envoie F_k (loss du modele global sur ses donnees AVANT
    training). Plus F_k est proche de la moyenne globale, plus son poids
    augmente. Les outliers (clients tres differents) sont downweightes.

      beta = 0   : FedAvg (pas d'ajustement)
      beta > 0   : reponderation vers le consensus, plus c'est grand plus
                   l'effet est marque.
    """

    def __init__(self, beta=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = float(beta)

    def aggregate_train(self, server_round, replies):
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        contents = [msg.content for msg in valid]
        # Mesure uplink
        self.last_uplink_bytes = sum(_content_bytes(c) for c in contents)
        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)

        if not contents:
            return self._last_global_arrays, metrics
        accepted = [c for c in contents if not _is_dropped(c)]
        if not accepted:
            return self._last_global_arrays, metrics

        # Extrait (arrays, f_k, n_k) pour chaque client valide
        rows = []
        for c in accepted:
            ar = next(iter(c.array_records.values()))
            mr = next(iter(c.metric_records.values()))
            f_k = float(mr.get("f_k", 0.0))
            n_k = float(mr.get("num-examples", 0.0))
            if n_k > 0:
                rows.append((ar, f_k, n_k))

        if not rows:
            return aggregate_arrayrecords(accepted, self.weighted_by_key), metrics

        # Calcule les poids FairFed
        weights = _fairfed_weights([(f, n) for _, f, n in rows], self.beta)
        if weights is None:
            return aggregate_arrayrecords(accepted, self.weighted_by_key), metrics

        # Aggregation manuelle avec les poids ajustes
        global_sd = self._last_global_arrays.to_torch_state_dict()
        local_sds = [ar.to_torch_state_dict() for ar, _, _ in rows]
        new_sd = _aggregate_with_weights(global_sd, local_sds, weights)
        return ArrayRecord(new_sd), metrics
