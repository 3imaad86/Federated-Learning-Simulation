"""Boucle serveur partagee (FedAvg/FedProx/FedNova/FedSGD).

Factorise le code strictement identique entre les 4 server_app.py :
  * init state dict
  * agg_train / agg_eval (log CSV + prints [round N])
  * boucle rounds + early stopping
  * save final model + copie CSV

Seule nouveaute vs projets originaux : appel a `central_evaluate()`
apres chaque round pour calculer l'accuracy sur le test set CIFAR-10
officiel. Les CSV recoivent 4 colonnes supplementaires (central_*).
"""

import logging
import os
import shutil
import time

import torch
from flwr.app import ArrayRecord, ConfigRecord

from .data import Net, get_device, get_testset, model_size_bytes
from .metrics import (
    RESULTS_DIR, ensure_dir, log_participation, log_round, log_summary,
    reset_files, resolve_dst_results_dir, rounds_to_convergence, rounds_to_target,
)
from .training import test_with_class_accuracies


def central_evaluate(arrays, device, loader_cache={}):
    """Evaluation centralisee sur le test set CIFAR-10 officiel (10k images IID).

    ATTENTION : cette evaluation viole la contrainte FL en production
    (aucune donnee ne doit atteindre le serveur). Elle sert UNIQUEMENT
    a comparer les algos en simulation avec une metrique non biaisee
    par la distribution non-IID. En deploiement reel, utiliser plutot
    des held-out clients ou secure aggregation.
    """
    from torch.utils.data import DataLoader

    if "loader" not in loader_cache:
        testset = get_testset()
        loader_cache["loader"] = DataLoader(testset, batch_size=256, shuffle=False)
    loader = loader_cache["loader"]

    net = Net()
    net.load_state_dict(arrays.to_torch_state_dict())
    net.to(device)
    loss, acc, class_accs, macro_recall, macro_f1 = test_with_class_accuracies(
        net, loader, device
    )
    return {
        "central_loss": float(loss),
        "central_accuracy": float(acc),
        "central_class_accs": [float(a) for a in class_accs],
        "central_macro_recall": float(macro_recall),
        "central_macro_f1": float(macro_f1),
    }


def _build_state(t_start, num_clients_expected):
    return {
        "round": 0,
        "accs_history": [],
        "central_accs_history": [],
        "t_last": t_start,
        "last_train_times": [],
        "n_train_clients_round": 0,
        "n_aggregated_clients_round": 0,
        "n_timeout_round": 0,
        "num_clients_expected": num_clients_expected,
        "participation": {"replied": {}, "aggregated": {}},
        "mean_epochs_used": 0.0,
        "mean_resource_tier": 0.0,
        "clients_detail": [],
        "best_acc": 0.0,
        "no_improve": 0,
        "early_stop": False,
        "last_central": {
            "central_accuracy": 0.0, "central_loss": 0.0,
            "central_macro_recall": 0.0, "central_macro_f1": 0.0,
        },
    }


def _make_agg_train(state, aggregate_metricrecords):
    def agg_train(records, wk):
        recs = list(records)
        state["n_train_clients_round"] = len(recs)
        # Clients qui n'ont PAS repondu dans le timeout serveur
        # (grid.send_and_receive a rendu la main avant de les recevoir).
        state["n_timeout_round"] = max(
            0, state["num_clients_expected"] - len(recs)
        )
        if not recs:
            # Tous les clients ont timeout : pas d'agregation possible.
            # On reset l'etat par-round pour que agg_eval affiche du vide.
            state["last_train_times"] = []
            state["clients_detail"] = []
            state["mean_epochs_used"] = 0.0
            state["mean_resource_tier"] = 0.0
            state["n_dropped_round"] = 0
            state["n_aggregated_clients_round"] = 0
            return None
        m = aggregate_metricrecords(recs, wk)
        times, epochs_list, tier_list = [], [], []
        clients_detail = []
        n_dropped = 0
        n_aggregated = 0
        for rec in recs:
            mr = next(iter(rec.metric_records.values()))
            pid = int(mr.get("partition_id", -1))
            if pid >= 0:
                replied = state["participation"]["replied"]
                replied[pid] = replied.get(pid, 0) + 1
            is_dropped = int(float(mr.get("dropped", 0.0)) >= 0.5)
            n_dropped += is_dropped
            if not is_dropped and pid >= 0:
                n_aggregated += 1
                aggregated = state["participation"]["aggregated"]
                aggregated[pid] = aggregated.get(pid, 0) + 1
                times.append(float(mr.get("local_time_s", 0.0)))
                epochs_list.append(float(mr.get("epochs_used", 0.0)))
                tier_list.append(float(mr.get("resource_tier", 1.0)))
            clients_detail.append({
                "pid": pid,
                "n": int(mr.get("num-examples", 0)),
                "epochs": float(mr.get("epochs_used", 0.0)),
                "tier": float(mr.get("resource_tier", 1.0)),
                "time": float(mr.get("local_time_s", 0.0)),
                "net_tier": int(mr.get("net_tier", 1)),
                "comm_time": float(mr.get("comm_time_s", 0.0)),
                "dropped": is_dropped,
            })
        state["last_train_times"] = times
        state["clients_detail"] = clients_detail
        state["mean_epochs_used"] = sum(epochs_list) / max(len(epochs_list), 1)
        state["mean_resource_tier"] = sum(tier_list) / max(len(tier_list), 1)
        state["n_dropped_round"] = n_dropped
        state["n_aggregated_clients_round"] = n_aggregated
        return m
    return agg_train


def _make_agg_eval(state, model_mb, straggler_sim, es_patience, es_min_delta,
                   extra_tail_fn, aggregate_metricrecords):
    """Usine a agg_eval. extra_tail_fn(r) renvoie la chaine a accoler en fin de print."""

    def agg_eval(records, wk):
        recs = list(records)
        # Si tous les clients ont timeout sur l'eval, on saute l'aggregation
        # des metrics (sinon aggregate_metricrecords crash). On affiche
        # quand meme la ligne [round N] avec des valeurs 0 pour garder la
        # visibilite du timeout.
        if recs:
            m = aggregate_metricrecords(recs, wk)
        else:
            m = {}
        state["round"] += 1
        r = state["round"]

        global_acc = float(m.get("accuracy", 0.0))
        global_loss = float(m.get("loss", 0.0))
        macro_recall = float(m.get("macro_recall", 0.0))
        macro_f1 = float(m.get("macro_f1", 0.0))
        state["accs_history"].append(global_acc)

        n_eval_clients = len(recs)
        n_train_clients = int(state.get("n_train_clients_round", n_eval_clients))
        n_aggregated_clients = int(state.get("n_aggregated_clients_round", n_train_clients))
        comm_mb = 2.0 * n_train_clients * model_mb

        now = time.perf_counter()
        round_time_s = now - state["t_last"]
        state["t_last"] = now
        tt = state["last_train_times"] or [0.0]
        mean_ct = float(sum(tt) / len(tt))
        max_ct = float(max(tt))

        client_accs = []
        for rec in recs:
            mr = next(iter(rec.metric_records.values()))
            client_accs.append(float(mr.get("accuracy", 0.0)))

        class_accs = list(m.get("class_accuracies", [0.0] * 10))

        c = state["last_central"]
        log_round(
            r, global_acc, global_loss, comm_mb, macro_recall, macro_f1,
            client_accs, class_accs,
            round_time_s=round_time_s, mean_client_time_s=mean_ct,
            max_client_time_s=max_ct,
            mean_epochs_used=state["mean_epochs_used"],
            mean_resource_tier=state["mean_resource_tier"],
            n_train_replied=n_train_clients,
            n_train_aggregated=n_aggregated_clients,
            central_accuracy=c["central_accuracy"],
            central_loss=c["central_loss"],
            central_macro_recall=c["central_macro_recall"],
            central_macro_f1=c["central_macro_f1"],
        )

        tier_names = {0: "weak", 1: "medium", 2: "strong"}
        net_names = {0: "lora", 1: "lte", 2: "wifi"}
        print(f"[round {r}] clients replied ({n_train_clients}):")
        for cd in sorted(state["clients_detail"], key=lambda x: x["pid"]):
            tname = tier_names.get(int(cd["tier"]), "?")
            nname = net_names.get(int(cd.get("net_tier", 1)), "?")
            flag = " DROP" if cd.get("dropped", 0) else ""
            print(f"  pid={cd['pid']:>2}  n={cd['n']:>5}  E={cd['epochs']:.0f}  "
                  f"tier={tname:<6}  net={nname:<4}  t={cd['time']:.2f}s  "
                  f"comm={cd.get('comm_time', 0.0):.2f}s{flag}")
        if straggler_sim:
            n_timeout = state.get("n_timeout_round", 0)
            n_expected = state.get("num_clients_expected", n_train_clients)
            print(f"[round {r}] stragglers dropped={state.get('n_dropped_round', 0)}/{n_train_clients} "
                  f"timeout={n_timeout}/{n_expected} aggregated={n_aggregated_clients}")
        tail = extra_tail_fn(r) if extra_tail_fn else ""
        gap = global_acc - c["central_accuracy"]
        print(f"[round {r}] acc_local={global_acc:.3f} acc_central={c['central_accuracy']:.3f} "
              f"gap={gap:+.3f} loss={global_loss:.3f} "
              f"recall={macro_recall:.3f} f1={macro_f1:.3f} "
              f"comm={comm_mb:.2f}MB replied={n_train_clients} aggregated={n_aggregated_clients} "
              f"round={round_time_s:.1f}s mean_ct={mean_ct:.2f}s "
              f"E={state['mean_epochs_used']:.2f} tier={state['mean_resource_tier']:.2f}{tail}")

        if es_patience > 0:
            if global_acc > state["best_acc"] + es_min_delta:
                state["best_acc"] = global_acc
                state["no_improve"] = 0
            else:
                state["no_improve"] += 1
                if state["no_improve"] >= es_patience:
                    state["early_stop"] = True
                    print(f"[early-stop] convergence detectee a r={r} "
                          f"(best_acc={state['best_acc']:.3f}, "
                          f"patience={es_patience}, min_delta={es_min_delta})")
        return m
    return agg_eval


def run_federated_training(
    grid,
    cfg,
    algo_name,
    strategy_factory,
    train_config_fn,
    project_dir_name,
    extra_tail_fn=None,
    banner_extra="",
):
    """Boucle serveur complete partagee par les 4 algos.

    Args:
        grid: objet Flower Grid (fourni par @app.main()).
        cfg: context.run_config.
        algo_name: "FedAvg" | "FedProx" | "FedNova" | "FedSGD" (pour logs).
        strategy_factory: callable(agg_train, agg_eval, frac_eval) -> Strategy.
        train_config_fn: callable(round_idx, lr, cfg) -> dict pour ConfigRecord.
        project_dir_name: nom du dossier projet (pour copie CSV post-run).
        extra_tail_fn: optionnel, callable(r) -> str accolee en fin de print [round N].
        banner_extra: chaine accolee au print [config] de demarrage.
    """
    from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

    logging.getLogger("flwr").setLevel(logging.WARNING)
    reset_files()

    num_rounds = int(cfg["num-server-rounds"])
    frac_eval = float(cfg["fraction-evaluate"])
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
    t_start = time.perf_counter()
    state = _build_state(t_start, num_clients_expected=num_clients)

    agg_train = _make_agg_train(state, aggregate_metricrecords)
    agg_eval = _make_agg_eval(
        state, model_mb, straggler_sim, es_patience, es_min_delta,
        extra_tail_fn, aggregate_metricrecords,
    )
    strategy = strategy_factory(agg_train, agg_eval, frac_eval)

    # Timeout serveur : si un round_deadline est fixe, le serveur n'attendra
    # PAS indefiniment les clients en retard. grid.send_and_receive rend la
    # main au bout de `timeout` secondes et l'agregation se fait avec les
    # replies effectivement recus. Les clients "en timeout" sont simplement
    # absents de l'agregation (differents des "dropped=1" qui ont repondu
    # avec un flag). Ce timeout s'applique a TRAIN ET EVAL de chaque round.
    # Si round_deadline_s <= 0, on laisse Flower attendre (default 3600s).
    start_kwargs = dict(
        grid=grid,
        initial_arrays=None,  # patche par round
        train_config=None,
        num_rounds=1,
    )
    if straggler_sim and round_deadline_s > 0:
        start_kwargs["timeout"] = float(round_deadline_s)

    arrays = ArrayRecord(Net().state_dict())
    result = None
    for round_idx in range(1, num_rounds + 1):
        train_cfg_dict = train_config_fn(round_idx, lr, cfg)
        start_kwargs["initial_arrays"] = arrays
        start_kwargs["train_config"] = ConfigRecord(train_cfg_dict)
        result = strategy.start(**start_kwargs)
        # Si TOUS les replies ont timeout, result.arrays peut etre None :
        # on garde les arrays precedents dans ce cas.
        if result.arrays is not None:
            arrays = result.arrays
        # Evaluation centralisee serveur-side : mise a jour du cache AVANT
        # que agg_eval du round courant ne lise state["last_central"].
        # (En pratique, agg_eval a deja tourne avant cette ligne pour le round
        #  courant ; on stocke pour qu'agg_eval du round suivant le log, et on
        #  reecrit la derniere ligne du CSV ci-dessous pour synchroniser.)
        central = central_evaluate(arrays, device)
        state["last_central"] = central
        state["central_accs_history"].append(central["central_accuracy"])
        # Rewrite la derniere ligne du CSV pour y inclure le central_* du round
        # qui vient de finir (le log_round a ete fait avec le central du round
        # precedent).
        _patch_last_global_csv_row(central)
        if state["early_stop"]:
            break

    total_time = time.perf_counter() - t_start
    log_summary(total_time, state["accs_history"], state["participation"], num_clients=num_clients)
    log_participation(state["participation"], num_clients=num_clients)

    rtc = rounds_to_convergence(state["accs_history"], ratio=0.9)
    r50 = rounds_to_target(state["accs_history"], 0.5)
    r70 = rounds_to_target(state["accs_history"], 0.7)
    r90 = rounds_to_target(state["accs_history"], 0.9)
    c_rtc = rounds_to_convergence(state["central_accs_history"], ratio=0.9)
    c_r50 = rounds_to_target(state["central_accs_history"], 0.5)
    c_r70 = rounds_to_target(state["central_accs_history"], 0.7)
    final_central = state["last_central"]["central_accuracy"]
    tail = extra_tail_fn(state["round"]) if extra_tail_fn else ""
    print(f"[done] total_time={total_time:.1f}s | LOCAL rtc90={rtc} "
          f"r50={r50} r70={r70} r90={r90} | CENTRAL final={final_central:.3f} "
          f"rtc90={c_rtc} r50={c_r50} r70={c_r70} | "
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


def _patch_last_global_csv_row(central):
    """Met a jour les 4 dernieres colonnes de la derniere ligne de
    metrics_global.csv avec les valeurs central_* du round qui vient
    de finir. Necessaire car log_round() est appele DANS agg_eval()
    (i.e. AVANT que le round soit termine et que central_evaluate()
    puisse tourner sur les nouveaux arrays).
    """
    import csv as _csv
    from .metrics import GLOBAL_CSV, GLOBAL_HEADER
    if not os.path.exists(GLOBAL_CSV):
        return
    with open(GLOBAL_CSV, "r", newline="", encoding="utf-8") as f:
        rows = list(_csv.reader(f))
    if len(rows) < 2:
        return
    last = rows[-1]
    if len(last) != len(GLOBAL_HEADER):
        return
    last[-4] = str(float(central["central_accuracy"]))
    last[-3] = str(float(central["central_loss"]))
    last[-2] = str(float(central["central_macro_recall"]))
    last[-1] = str(float(central["central_macro_f1"]))
    with open(GLOBAL_CSV, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerows(rows)
