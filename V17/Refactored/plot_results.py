"""Plots de comparaison FedAvg / FedProx / FedNova / FedSGD / SCAFFOLD / q-FedAvg / FairFed / BEFL.

Lit les CSV dans chaque `<algo>/results/` et produit des PNG dans `Refactored/plots/`.

Usage :
    conda activate FL
    cd Refactored
    python plot_results.py                          # tous les algos trouves
    python plot_results.py fedavg befl              # uniquement ceux listes
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

ALGOS = ["fedavg", "fedprox", "fednova", "fedsgd", "scaffold", "qfedavg", "fairfed", "befl"]
COLORS = {"fedavg": "tab:blue", "fedprox": "tab:orange",
          "fednova": "tab:green", "fedsgd": "tab:red",
          "scaffold": "tab:purple", "qfedavg": "tab:brown",
          "fairfed": "tab:pink", "befl": "tab:olive"}

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "plots")
os.makedirs(OUT_DIR, exist_ok=True)


def load_csvs(algo):
    """Retourne (global_df, per_class_df, summary_df) ou None si absent."""
    results = os.path.join(ROOT, algo, "results")
    g = os.path.join(results, "metrics_global.csv")
    if not os.path.exists(g):
        return None
    gdf = pd.read_csv(g)
    pdf = pd.read_csv(os.path.join(results, "metrics_per_class.csv"))
    sdf = pd.read_csv(os.path.join(results, "metrics_summary.csv"))
    return gdf, pdf, sdf


def plot_curve(data, y_col, ylabel, title, filename, cumulative=False):
    """Plot une courbe par algo pour la colonne `y_col`."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, (gdf, _, _) in data.items():
        y = gdf[y_col].cumsum() if cumulative else gdf[y_col]
        ax.plot(gdf["round"], y, marker="o", label=algo.upper(),
                color=COLORS[algo], linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=120)
    plt.close(fig)
    print(f"  -> {filename}")


def plot_server_vs_local(data, server_col, local_col, ylabel, title, filename):
    """Overlay server (ligne pleine) vs local agrege (ligne pointillee) par algo.

    Le gap entre les 2 courbes visualise le biais non-IID : chaque client
    overfitte sa propre distribution -> sa metrique locale est sur-estimee
    par rapport a l'eval serveur sur un test set IID.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, (gdf, _, _) in data.items():
        color = COLORS[algo]
        ax.plot(gdf["round"], gdf[server_col], marker="o", linestyle="-",
                color=color, linewidth=2, label=f"{algo.upper()} (server)")
        if local_col in gdf.columns:
            ax.plot(gdf["round"], gdf[local_col], marker="x", linestyle="--",
                    color=color, linewidth=1.5, alpha=0.8,
                    label=f"{algo.upper()} (local agg)")
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=120)
    plt.close(fig)
    print(f"  -> {filename}")


def plot_per_class_heatmap(data):
    """Grille de heatmaps per-class accuracy par algo."""
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    for i, (algo, (_, pdf, _)) in enumerate(data.items()):
        ax = axes[0, i]
        class_cols = [c for c in pdf.columns if c.startswith("class_")]
        mat = pdf[class_cols].values.T  # classes en lignes, rounds en colonnes
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=0.0, vmax=1.0,
                       extent=[0.5, len(pdf) + 0.5, 9.5, -0.5])
        ax.set_title(algo.upper())
        ax.set_xlabel("Round")
        ax.set_ylabel("Classe CIFAR-10")
        ax.set_yticks(range(10))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Accuracy par classe (eval serveur sur CIFAR-10 IID)")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "per_class_heatmap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  -> per_class_heatmap.png")


def plot_summary_bars(data):
    """Bar chart des rounds-to-target (rtc90, r50, r70, r90) par algo."""
    metrics = ["rtc90", "rounds_to_50", "rounds_to_70", "rounds_to_90"]
    labels = ["RTC90", "R->0.50", "R->0.70", "R->0.90"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(metrics))
    width = 0.8 / max(len(data), 1)
    for i, (algo, (_, _, sdf)) in enumerate(data.items()):
        vals = []
        for m in metrics:
            v = sdf[m].iloc[0] if m in sdf.columns else None
            vals.append(float(v) if pd.notna(v) else 0)
        ax.bar([xi + i * width for xi in x], vals, width=width,
               label=algo.upper(), color=COLORS[algo])
    ax.set_xticks([xi + width * (len(data) - 1) / 2 for xi in x])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Nombre de rounds (0 = non atteint)")
    ax.set_title("Vitesse de convergence par algorithme")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "summary_bars.png"), dpi=120)
    plt.close(fig)
    print(f"  -> summary_bars.png")


def main():
    algos = sys.argv[1:] if len(sys.argv) > 1 else ALGOS
    data = {}
    for algo in algos:
        res = load_csvs(algo)
        if res is None:
            print(f"[skip] {algo} : metrics_global.csv absent")
            continue
        data[algo] = res
        print(f"[ok]   {algo} : {len(res[0])} rounds")

    if not data:
        print("Aucun resultat a tracer. Lance d'abord `flwr run .` dans un dossier d'algo.")
        return

    print(f"\nSauvegarde dans {OUT_DIR}:")
    plot_curve(data, "accuracy", "Accuracy",
               "Accuracy (eval serveur CIFAR-10 IID)", "accuracy_curve.png")
    plot_curve(data, "loss", "Loss",
               "Loss (eval serveur)", "loss_curve.png")
    plot_curve(data, "macro_f1", "Macro F1",
               "Macro F1", "macro_f1_curve.png")
    plot_curve(data, "jfi_classes", "Jain's Fairness Index (classes)",
               "Fairness inter-classes", "fairness_curve.png")
    plot_curve(data, "min_max_class_gap", "max(class_acc) - min(class_acc)",
               "Ecart entre meilleure et pire classe", "class_gap_curve.png")
    plot_curve(data, "comm_cost_mb", "Comm cumule (MB)",
               "Cout de communication cumule", "comm_cost_curve.png",
               cumulative=True)
    plot_curve(data, "round_time_s", "Duree round (s)",
               "Temps par round", "round_time_curve.png")
    plot_curve(data, "energy_j_round", "Energie (J) / round",
               "Energie par round (edge IoT)", "energy_round_curve.png")
    plot_curve(data, "energy_j_cumulative", "Energie cumulee (J)",
               "Energie cumulee (edge IoT)", "energy_cumulative_curve.png")

    # Overlays server vs local agrege (visualise le biais non-IID)
    plot_server_vs_local(data, "accuracy", "local_acc", "Accuracy",
                         "Server (IID) vs local agg (non-IID bias)",
                         "accuracy_server_vs_local.png")
    plot_server_vs_local(data, "loss", "local_loss", "Loss",
                         "Server (IID) vs local agg (non-IID bias)",
                         "loss_server_vs_local.png")

    plot_per_class_heatmap(data)
    plot_summary_bars(data)
    print(f"\n{len(data)} algo(s) tracees.")


if __name__ == "__main__":
    main()
