"""Boucle serveur partagee par les 4 algos (FedAvg/FedProx/FedNova/FedSGD).

Pour chaque round :
  1. strategy.start(num_rounds=1) -> train cote clients seulement
  2. evaluate() cote SERVEUR sur le test set CIFAR-10 officiel (10k IID)
  3. log CSV (metrics_global.csv, metrics_per_class.csv) + prints

Pourquoi eval cote serveur ?
  En non-IID, si chaque client evalue sur sa propre partition, les metriques
  agregees sont biaisees (chaque client sur-represente ses classes majoritaires).
  Une eval centralisee sur un test set IID donne une mesure fiable de la
  qualite du modele global. Hypothese simu : OK d'avoir un test set serveur.
"""

import logging
import os
import shutil
import time

import torch
from flwr.app import ArrayRecord, ConfigRecord
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords
from torch.utils.data import DataLoader

from .data import Net, get_device, get_testset, model_size_bytes
from .metrics import (
    RESULTS_DIR, ensure_dir, log_participation, log_round, log_summary,
    reset_files, resolve_dst_results_dir, rounds_to_convergence, rounds_to_target,
)
from .training import test_with_class_accuracies


def server_evaluate(arrays, device, _cache={}):
    """Evaluation du modele global sur le test set CIFAR-10 (10k images IID)."""
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


def run_federated_training(
    grid, cfg, algo_name, strategy_class, strategy_kwargs,
    train_config_fn, project_dir_name,
    extra_tail_fn=None, banner_extra="",
):
    """Boucle complete. Pour chaque round : start(1) -> eval serveur -> log.

    Args:
        grid: Flower Grid (depuis @app.main()).
        cfg: context.run_config.
        algo_name: pour les logs ("FedAvg", ...).
        strategy_class: classe de strategie (FedAvgDropFilter, FedNovaStrategy, ...).
        strategy_kwargs: kwargs supplementaires pour strategy_class.__init__.
        train_config_fn: callable(round_idx, lr, cfg) -> dict pour ConfigRecord.
        project_dir_name: nom du dossier projet (pour copie CSV post-run).
        extra_tail_fn: optionnel, callable(r, strategy) -> str.
        banner_extra: chaine ajoutee au print [config] initial.
    """
    logging.getLogger("flwr").setLevel(logging.WARNING)
    reset_files()

    num_rounds = int(cfg["num-server-rounds"])
    lr = float(cfg["learning-rate"])
    num_clients = int(cfg.get("num-clients", 10))
    partitioning = str(cfg.get("partitioning", "noniid"))
    dir_alpha = float(cfg.get("dirichlet-alpha", 0.3))
    es_patience = int(cfg.get("early-stopping-patience", 0))
    es_min_delta = float(cfg.get("early-stopping-min-delta", 0.001))
    straggler_sim = int(cfg.get("straggler-sim", 0))
    round_deadline_s = float(cfg.get("round-deadline-s", 0.0))

    extra = f" alpha={dir_alpha}" if partitioning.lower() != "iid" else ""
    strag = f" straggler-sim=1 deadline={round_deadline_s}s" if straggler_sim else ""
    print(f"[config] algo={algo_name}{banner_extra} partitioning={partitioning}{extra} "
          f"num_rounds={num_rounds} lr={lr}{strag}")

    model_mb = model_size_bytes() / (1024.0 * 1024.0)
    device = get_device()

    # --- Callback agg_train : ne sert qu'a collecter les metriques de
    #     temps / comm / epochs des clients qui ont repondu.
    info = {"train": None}

    def agg_train(records, wk):
        recs = list(records)
        times, epochs_list, tier_list, details = [], [], [], []
        n_dropped, participation_add, energy_total = 0, {}, 0.0
        m = aggregate_metricrecords(recs, wk) if recs else {}
        for rec in recs:
            mr = next(iter(rec.metric_records.values()))
            pid = int(mr.get("partition_id", -1))
            if pid >= 0:
                participation_add[pid] = participation_add.get(pid, 0) + 1
            t = float(mr.get("local_time_s", 0.0))
            e = float(mr.get("epochs_used", 0.0))
            tier = float(mr.get("resource_tier", 1.0))
            energy = float(mr.get("energy_j", 0.0))
            energy_total += energy
            is_drop = int(float(mr.get("dropped", 0.0)) >= 0.5)
            n_dropped += is_drop
            times.append(t); epochs_list.append(e); tier_list.append(tier)
            details.append({
                "pid": pid, "n": int(mr.get("num-examples", 0)),
                "epochs": e, "tier": tier, "time": t,
                "net_tier": int(mr.get("net_tier", 1)),
                "comm_time": float(mr.get("comm_time_s", 0.0)),
                "dropped": is_drop, "energy": energy,
            })
        info["train"] = {
            "n_clients": len(recs), "times": times, "details": details,
            "mean_epochs": sum(epochs_list) / max(len(epochs_list), 1),
            "mean_tier": sum(tier_list) / max(len(tier_list), 1),
            "n_dropped": n_dropped, "participation_add": participation_add,
            "energy_j_round": energy_total,
        }
        return m

    # fraction_evaluate=0.0 : pas d'eval cote client (on evalue serveur-side)
    strategy = strategy_class(
        fraction_evaluate=0.0,
        train_metrics_aggr_fn=agg_train,
        **strategy_kwargs,
    )

    # Timeout serveur : si deadline > 0, grid.send_and_receive rend la main
    # au bout de `timeout` secondes. Les replies en retard sont discardes.
    start_kwargs = dict(grid=grid, num_rounds=1)
    if straggler_sim and round_deadline_s > 0:
        start_kwargs["timeout"] = float(round_deadline_s)

    # --- Etat de la boucle -------------------------------------------------
    t_start = time.perf_counter()
    t_last = t_start
    arrays = ArrayRecord(Net().state_dict())
    accs_history = []
    participation = {}
    best_acc, no_improve = 0.0, 0
    energy_cumul = 0.0

    for r in range(1, num_rounds + 1):
        # 1) Un round FL : uniquement TRAIN cote clients
        start_kwargs["initial_arrays"] = arrays
        start_kwargs["train_config"] = ConfigRecord(train_config_fn(r, lr, cfg))
        result = strategy.start(**start_kwargs)
        if result.arrays is not None:
            arrays = result.arrays

        # Snapshot du temps juste apres strategy.start (temps FL pur)
        now = time.perf_counter()
        round_time_s = now - t_last
        t_last = now

        # 2) Eval du modele global cote SERVEUR (test set CIFAR-10 IID)
        ev = server_evaluate(arrays, device)
        accs_history.append(ev["accuracy"])

        # 3) Recup metriques compute/comm/energie depuis le callback agg_train
        tr = info["train"] or {"n_clients": 0, "times": [], "details": [],
                               "mean_epochs": 0.0, "mean_tier": 0.0,
                               "n_dropped": 0, "participation_add": {},
                               "energy_j_round": 0.0}
        for pid, n in tr["participation_add"].items():
            participation[pid] = participation.get(pid, 0) + n
        energy_round = tr["energy_j_round"]
        energy_cumul += energy_round

        # 4) Log CSV
        times = tr["times"] or [0.0]
        comm_mb = 2.0 * tr["n_clients"] * model_mb
        log_round(
            r, ev["accuracy"], ev["loss"], ev["macro_recall"], ev["macro_f1"],
            ev["class_accs"],
            comm_cost_mb=comm_mb,
            round_time_s=round_time_s,
            mean_client_time_s=sum(times) / len(times),
            max_client_time_s=max(times),
            mean_epochs_used=tr["mean_epochs"],
            mean_resource_tier=tr["mean_tier"],
            energy_j_round=energy_round,
            energy_j_cumulative=energy_cumul,
        )

        # 5) Prints par-round
        tier_names = {0: "weak", 1: "medium", 2: "strong"}
        net_names = {0: "lora", 1: "lte", 2: "wifi"}
        print(f"[round {r}] clients participants ({tr['n_clients']}):")
        for cd in sorted(tr["details"], key=lambda x: x["pid"]):
            tname = tier_names.get(int(cd["tier"]), "?")
            nname = net_names.get(cd["net_tier"], "?")
            flag = " DROP" if cd["dropped"] else ""
            print(f"  pid={cd['pid']:>2}  n={cd['n']:>5}  E={cd['epochs']:.0f}  "
                  f"tier={tname:<6}  net={nname:<4}  t={cd['time']:.2f}s  "
                  f"comm={cd['comm_time']:.2f}s  energy={cd['energy']:.1f}J{flag}")
        if straggler_sim:
            n_timeout = max(0, num_clients - tr["n_clients"])
            print(f"[round {r}] stragglers dropped={tr['n_dropped']}/{tr['n_clients']} "
                  f"timeout={n_timeout}/{num_clients}")
        tail = extra_tail_fn(r, strategy) if extra_tail_fn else ""
        print(f"[round {r}] acc={ev['accuracy']:.3f} loss={ev['loss']:.3f} "
              f"recall={ev['macro_recall']:.3f} f1={ev['macro_f1']:.3f} "
              f"comm={comm_mb:.2f}MB n={tr['n_clients']} "
              f"round={round_time_s:.1f}s mean_ct={sum(times)/len(times):.2f}s "
              f"E={tr['mean_epochs']:.2f} tier={tr['mean_tier']:.2f} "
              f"energy={energy_round:.1f}J (cumul={energy_cumul:.1f}J){tail}")

        # 6) Early stopping
        if es_patience > 0:
            if ev["accuracy"] > best_acc + es_min_delta:
                best_acc = ev["accuracy"]; no_improve = 0
            else:
                no_improve += 1
                if no_improve >= es_patience:
                    print(f"[early-stop] convergence a r={r} "
                          f"(best_acc={best_acc:.3f}, patience={es_patience})")
                    break

        info["train"] = None

    # --- Fin de run : summary + CSV + copie -------------------------------
    total_time = time.perf_counter() - t_start
    log_summary(total_time, accs_history, participation, num_clients=num_clients)
    log_participation(participation, num_clients=num_clients)

    rtc = rounds_to_convergence(accs_history, ratio=0.9)
    r50 = rounds_to_target(accs_history, 0.5)
    r70 = rounds_to_target(accs_history, 0.7)
    r90 = rounds_to_target(accs_history, 0.9)
    final_acc = accs_history[-1] if accs_history else 0.0
    tail = extra_tail_fn(len(accs_history), strategy) if extra_tail_fn else ""
    print(f"[done] total_time={total_time:.1f}s final_acc={final_acc:.3f} "
          f"rtc90={rtc} r50={r50} r70={r70} r90={r90} "
          f"energy_total={energy_cumul:.1f}J "
          f"partitioning={partitioning}{extra}{tail}")
    torch.save(arrays.to_torch_state_dict(), os.path.join(ensure_dir(), "final_model.pt"))
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
