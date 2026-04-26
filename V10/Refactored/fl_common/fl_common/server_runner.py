"""Boucle serveur partagee par les 4 algos (FedAvg/FedProx/FedNova/FedSGD).

Pour chaque round :
  1. strategy.start(num_rounds=1) -> train cote clients
  2. server_evaluate()             -> eval sur CIFAR-10 IID (10k images)
  3. log_round() + print_round()
  4. early stopping (optionnel)

Pourquoi eval cote serveur ?
  En non-IID, les metriques agregees cote client surestiment l'accuracy
  (chaque client overfitte sa partition). On utilise donc le test set
  CIFAR-10 officiel comme verite terrain.
"""

import logging
import os
import shutil
import time

import torch
from flwr.app import ArrayRecord, ConfigRecord
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords
from torch.utils.data import DataLoader

from .data import Net, get_device, get_testset, model_size_bytes, set_seed
from .metrics import (
    RESULTS_DIR, ensure_dir, log_participation, log_round, log_summary,
    reset_files, resolve_dst_results_dir, rounds_to_convergence, rounds_to_target,
)
from .training import test_with_class_accuracies


# ============================================================================
# 1) Lecture de la config (run_config -> dict)
# ============================================================================

def parse_config(cfg):
    """Lit run_config et calcule la taille modele effective (compression)."""
    sim_mb = float(cfg.get("sim-model-mb", 0.0))
    ratio = float(cfg.get("comm-size-ratio", 1.0))
    base_mb = sim_mb if sim_mb > 0 else model_size_bytes() / (1024.0 * 1024.0)
    return {
        "num_rounds": int(cfg["num-server-rounds"]),
        "lr": float(cfg["learning-rate"]),
        "num_clients": int(cfg.get("num-clients", 10)),
        "partitioning": str(cfg.get("partitioning", "noniid")),
        "dir_alpha": float(cfg.get("dirichlet-alpha", 0.3)),
        "es_patience": int(cfg.get("early-stopping-patience", 0)),
        "es_min_delta": float(cfg.get("early-stopping-min-delta", 0.001)),
        "straggler_sim": int(cfg.get("straggler-sim", 0)),
        "round_deadline_s": float(cfg.get("round-deadline-s", 0.0)),
        "comm_size_ratio": ratio,
        "sim_model_mb": sim_mb,
        "model_mb": base_mb * ratio,  # taille effective transmise par round
        "seed": int(cfg.get("seed", -1)),
    }


def print_banner(algo_name, cfg, banner_extra=""):
    """Affiche '[config] algo=... partitioning=... ...' au debut du run."""
    extra = f" alpha={cfg['dir_alpha']}" if cfg["partitioning"].lower() != "iid" else ""
    strag = (f" straggler-sim=1 deadline={cfg['round_deadline_s']}s"
             if cfg["straggler_sim"] else "")
    compr = (f" comm-size-ratio={cfg['comm_size_ratio']}"
             if cfg["comm_size_ratio"] != 1.0 else "")
    sim = f" sim-model-mb={cfg['sim_model_mb']}" if cfg["sim_model_mb"] > 0 else ""
    sd = f" seed={cfg['seed']}" if cfg["seed"] >= 0 else ""
    print(f"[config] algo={algo_name}{banner_extra} partitioning={cfg['partitioning']}"
          f"{extra} num_rounds={cfg['num_rounds']} lr={cfg['lr']}{strag}{compr}{sim}{sd}")


# ============================================================================
# 2) Evaluation cote serveur sur CIFAR-10 IID
# ============================================================================

def server_evaluate(arrays, device, _cache={}):
    """Eval du modele global sur CIFAR-10 testset (10k images IID)."""
    if "loader" not in _cache:
        _cache["loader"] = DataLoader(get_testset(), batch_size=256, shuffle=False)
    net = Net()
    net.load_state_dict(arrays.to_torch_state_dict())
    net.to(device)
    loss, acc, class_accs, mr, mf = test_with_class_accuracies(net, _cache["loader"], device)
    return {
        "accuracy": float(acc), "loss": float(loss),
        "macro_recall": float(mr), "macro_f1": float(mf),
        "class_accs": [float(a) for a in class_accs],
    }


# ============================================================================
# 3) Aggregation des metriques clients (callback agg_train)
# ============================================================================

def _client_metrics(rec):
    """Extrait un dict propre depuis un MetricRecord d'un client."""
    mr = next(iter(rec.metric_records.values()))
    return {
        "pid": int(mr.get("partition_id", -1)),
        "n": int(mr.get("num-examples", 0)),
        "epochs": float(mr.get("epochs_used", 0.0)),
        "tier": float(mr.get("resource_tier", 1.0)),
        "net_tier": int(mr.get("net_tier", 1)),
        "time": float(mr.get("local_time_s", 0.0)),
        "comm_time": float(mr.get("comm_time_s", 0.0)),
        "dropped": int(float(mr.get("dropped", 0.0)) >= 0.5),
        "energy": float(mr.get("energy_j", 0.0)),
        "local_loss": float(mr.get("local_eval_loss", 0.0)),
        "local_acc": float(mr.get("local_eval_acc", 0.0)),
    }


def make_agg_train(round_info):
    """Cree le callback agg_train. round_info[0] sera rempli a chaque round."""
    def agg_train(records, wk):
        recs = list(records)
        m = aggregate_metricrecords(recs, wk) if recs else {}
        details = [_client_metrics(rec) for rec in recs]

        # Moyenne ponderee par num-examples (ignore les drops) pour local_loss/acc.
        active = [d for d in details if not d["dropped"] and d["n"] > 0]
        w = sum(d["n"] for d in active) or 1  # garde anti-/0
        local_loss = sum(d["local_loss"] * d["n"] for d in active) / w
        local_acc = sum(d["local_acc"] * d["n"] for d in active) / w

        # Compteur de participation (combien de fois chaque pid a participe).
        part = {}
        for d in details:
            if d["pid"] >= 0:
                part[d["pid"]] = part.get(d["pid"], 0) + 1

        round_info[0] = {
            "n_clients": len(details),
            "details": details,
            "n_dropped": sum(d["dropped"] for d in details),
            "participation_add": part,
            "energy_j_round": sum(d["energy"] for d in details),
            "local_loss": local_loss,
            "local_acc": local_acc,
            "mean_epochs": (sum(d["epochs"] for d in details) / len(details)
                            if details else 0.0),
            "mean_tier": (sum(d["tier"] for d in details) / len(details)
                          if details else 0.0),
            "times": [d["time"] for d in details],
        }
        return m
    return agg_train


def empty_round_info():
    """Valeurs par defaut quand aucun client n'a repondu."""
    return {
        "n_clients": 0, "details": [], "n_dropped": 0,
        "participation_add": {}, "energy_j_round": 0.0,
        "local_loss": 0.0, "local_acc": 0.0,
        "mean_epochs": 0.0, "mean_tier": 0.0, "times": [],
    }


# ============================================================================
# 4) Affichage console par round
# ============================================================================

TIER_NAMES = {0: "weak", 1: "medium", 2: "strong"}
NET_NAMES = {0: "lora", 1: "lte", 2: "wifi"}


def print_round(r, ev, tr, comm_mb, round_time_s, energy_round, energy_cumul,
                num_clients, straggler_sim, tail=""):
    """Print detaille d'un round (par-client + resume + gap non-IID)."""
    print(f"[round {r}] clients participants ({tr['n_clients']}):")
    for d in sorted(tr["details"], key=lambda x: x["pid"]):
        tname = TIER_NAMES.get(int(d["tier"]), "?")
        nname = NET_NAMES.get(d["net_tier"], "?")
        flag = " DROP" if d["dropped"] else ""
        print(f"  pid={d['pid']:>2}  n={d['n']:>5}  E={d['epochs']:.0f}  "
              f"tier={tname:<6}  net={nname:<4}  t={d['time']:.2f}s  "
              f"comm={d['comm_time']:.2f}s  energy={d['energy']:.1f}J{flag}")

    if straggler_sim:
        n_timeout = max(0, num_clients - tr["n_clients"])
        print(f"[round {r}] stragglers dropped={tr['n_dropped']}/{tr['n_clients']} "
              f"timeout={n_timeout}/{num_clients}")

    gap = tr["local_acc"] - ev["accuracy"]
    mean_ct = sum(tr["times"]) / max(len(tr["times"]), 1)
    print(f"[round {r}] server: acc={ev['accuracy']:.3f} loss={ev['loss']:.3f} "
          f"recall={ev['macro_recall']:.3f} f1={ev['macro_f1']:.3f}")
    print(f"[round {r}] local : acc={tr['local_acc']:.3f} loss={tr['local_loss']:.3f} "
          f"(gap_acc={gap:+.3f} = biais non-IID)")
    print(f"[round {r}] comm={comm_mb:.2f}MB n={tr['n_clients']} "
          f"round={round_time_s:.1f}s mean_ct={mean_ct:.2f}s "
          f"E={tr['mean_epochs']:.2f} tier={tr['mean_tier']:.2f} "
          f"energy={energy_round:.1f}J (cumul={energy_cumul:.1f}J){tail}")


# ============================================================================
# 5) Fin de run : summary CSV + copie + print final
# ============================================================================

def finalize_run(arrays, accs_history, participation, total_time, eval_time,
                 wall_time, energy_cumul, cfg, project_dir_name,
                 strategy=None, extra_tail_fn=None):
    """Sauvegarde le modele final, ecrit metrics_summary.csv, copie les CSV."""
    log_summary(total_time, accs_history, participation, num_clients=cfg["num_clients"])
    log_participation(participation, num_clients=cfg["num_clients"])

    rtc = rounds_to_convergence(accs_history, ratio=0.9)
    r50 = rounds_to_target(accs_history, 0.5)
    r70 = rounds_to_target(accs_history, 0.7)
    r90 = rounds_to_target(accs_history, 0.9)
    final_acc = accs_history[-1] if accs_history else 0.0
    extra = f" alpha={cfg['dir_alpha']}" if cfg["partitioning"].lower() != "iid" else ""
    tail = extra_tail_fn(len(accs_history), strategy) if extra_tail_fn else ""

    print(f"[done] total_time={total_time:.1f}s (eval_time={eval_time:.1f}s "
          f"wall={wall_time:.1f}s) final_acc={final_acc:.3f} "
          f"rtc90={rtc} r50={r50} r70={r70} r90={r90} "
          f"energy_total={energy_cumul:.1f}J "
          f"partitioning={cfg['partitioning']}{extra}{tail}")

    torch.save(arrays.to_torch_state_dict(),
               os.path.join(ensure_dir(), "final_model.pt"))
    print(f"[done] CSV -> {RESULTS_DIR}")

    dst = resolve_dst_results_dir(project_dir_name)
    if os.path.abspath(dst) != os.path.abspath(RESULTS_DIR):
        try:
            os.makedirs(dst, exist_ok=True)
            for fn in os.listdir(RESULTS_DIR):
                shutil.copy2(os.path.join(RESULTS_DIR, fn), os.path.join(dst, fn))
            print(f"[done] CSV copies dans {dst}")
        except Exception as e:
            print(f"[done] WARN copie CSV echouee: {e}")


# ============================================================================
# 6) Boucle principale
# ============================================================================

def run_federated_training(
    grid, cfg, algo_name, strategy_class, strategy_kwargs,
    train_config_fn, project_dir_name,
    extra_tail_fn=None, banner_extra="",
):
    """Boucle FL complete partagee par les 4 algos."""
    logging.getLogger("flwr").setLevel(logging.WARNING)
    reset_files()

    cfg = parse_config(cfg)
    if cfg["seed"] >= 0:
        set_seed(cfg["seed"])  # init torch RNG cote serveur (init Net global)
    print_banner(algo_name, cfg, banner_extra)
    device = get_device()

    # --- Strategie + callback agg_train --------------------------------------
    round_info = [None]  # rempli par agg_train a chaque round
    strategy = strategy_class(
        fraction_evaluate=0.0,                    # eval = serveur uniquement
        train_metrics_aggr_fn=make_agg_train(round_info),
        **strategy_kwargs,
    )
    start_kwargs = {"grid": grid, "num_rounds": 1}
    if cfg["straggler_sim"] and cfg["round_deadline_s"] > 0:
        start_kwargs["timeout"] = cfg["round_deadline_s"]

    # --- Etat de la boucle ---------------------------------------------------
    arrays = ArrayRecord(Net().state_dict())
    accs_history, participation = [], {}
    fl_time_total, eval_time_total, energy_cumul = 0.0, 0.0, 0.0
    best_acc, no_improve = 0.0, 0
    t_start = time.perf_counter()

    for r in range(1, cfg["num_rounds"] + 1):
        # 1) Un round FL pur (clients trainent uniquement)
        start_kwargs["initial_arrays"] = arrays
        start_kwargs["train_config"] = ConfigRecord(train_config_fn(r, cfg["lr"], cfg))
        t0 = time.perf_counter()
        result = strategy.start(**start_kwargs)
        round_time_s = time.perf_counter() - t0
        fl_time_total += round_time_s
        if result.arrays is not None:
            arrays = result.arrays

        # 2) Eval cote serveur (temps mesure separement, pas compte dans round_time_s)
        t_eval = time.perf_counter()
        ev = server_evaluate(arrays, device)
        eval_time_total += time.perf_counter() - t_eval
        accs_history.append(ev["accuracy"])

        # 3) Recup metriques clients + maj etat
        tr = round_info[0] or empty_round_info()
        for pid, n in tr["participation_add"].items():
            participation[pid] = participation.get(pid, 0) + n
        energy_cumul += tr["energy_j_round"]
        comm_mb = 2.0 * tr["n_clients"] * cfg["model_mb"]

        # 4) Log CSV
        log_round(
            r, ev["accuracy"], ev["loss"], ev["macro_recall"], ev["macro_f1"],
            ev["class_accs"],
            comm_cost_mb=comm_mb,
            round_time_s=round_time_s,
            mean_client_time_s=sum(tr["times"]) / max(len(tr["times"]), 1),
            max_client_time_s=max(tr["times"], default=0.0),
            mean_epochs_used=tr["mean_epochs"],
            mean_resource_tier=tr["mean_tier"],
            energy_j_round=tr["energy_j_round"],
            energy_j_cumulative=energy_cumul,
            local_loss=tr["local_loss"],
            local_acc=tr["local_acc"],
        )

        # 5) Print
        tail = extra_tail_fn(r, strategy) if extra_tail_fn else ""
        print_round(r, ev, tr, comm_mb, round_time_s,
                    tr["energy_j_round"], energy_cumul,
                    cfg["num_clients"], cfg["straggler_sim"], tail)

        # 6) Early stopping
        if cfg["es_patience"] > 0:
            if ev["accuracy"] > best_acc + cfg["es_min_delta"]:
                best_acc, no_improve = ev["accuracy"], 0
            else:
                no_improve += 1
                if no_improve >= cfg["es_patience"]:
                    print(f"[early-stop] convergence a r={r} "
                          f"(best_acc={best_acc:.3f}, patience={cfg['es_patience']})")
                    break

        round_info[0] = None  # reset pour le prochain round

    # --- Fin de run ----------------------------------------------------------
    finalize_run(
        arrays, accs_history, participation,
        total_time=fl_time_total,
        eval_time=eval_time_total,
        wall_time=time.perf_counter() - t_start,
        energy_cumul=energy_cumul,
        cfg=cfg, project_dir_name=project_dir_name,
        strategy=strategy, extra_tail_fn=extra_tail_fn,
    )
