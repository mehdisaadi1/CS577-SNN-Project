"""Aggregate results/runs/*.json into a summary table and figures.

Outputs:
    results/summary.md
    results/figures/D_vs_epoch.png
    results/figures/layer_stats.png
    results/figures/val_curves.png
"""

import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

RUNS_DIR = "results/runs"
FIG_DIR = "results/figures"
COLORS = {"selu": "#257de0", "elu": "#f47230"}
LABELS = {"selu": "SELU", "elu": "ELU"}


def load_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        runs.setdefault(r["activation"], []).append(r)
    return runs


def mean_std(values):
    m = st.mean(values)
    s = st.stdev(values) if len(values) > 1 else 0.0
    return m, s


def D_terms(h):
    """Split D into its mean part and its variance part."""
    L = len(h["layer_mean"])
    mean_part = sum(abs(m) for m in h["layer_mean"]) / L
    var_part = sum(abs(v - 1) for v in h["layer_var"]) / L
    return mean_part, var_part


def summary_table(runs):
    rows = [
        ("D at init", lambda r: r["history"][0]["D"]),
        ("D after training", lambda r: r["history"][-1]["D"]),
        ("  mean part (1/L) sum |mu_l|",
         lambda r: D_terms(r["history"][-1])[0]),
        ("  variance part (1/L) sum |var_l - 1|",
         lambda r: D_terms(r["history"][-1])[1]),
        ("Final train cross-entropy", lambda r: r["history"][-1]["train_loss"]),
        ("Final val cross-entropy", lambda r: r["history"][-1]["val_loss"]),
        ("Final val accuracy (%)", lambda r: 100 * r["history"][-1]["val_acc"]),
        ("Training time per run (s)", lambda r: r["wall_time_s"]),
    ]
    acts = [a for a in ("selu", "elu") if a in runs]
    cfg = runs[acts[0]][0]["config"]
    n_updates = runs[acts[0]][0]["history"][-1]["updates"]
    seeds = sorted(r["seed"] for r in runs[acts[0]])

    lines = [
        "# SELU vs ELU: matched ablation",
        "",
        f"Config: optimizer={cfg['optimizer']}, lr={cfg['lr']}, "
        f"batch={cfg['batch_size']}, epochs={cfg['epochs']} "
        f"({n_updates} updates), seeds={seeds}. "
        "Values are mean ± std over seeds; statistics measured on the "
        "2,000 validation images.",
        "",
        "| Metric | " + " | ".join(LABELS[a] for a in acts) + " |",
        "|---|" + "---|" * len(acts),
    ]
    for name, fn in rows:
        cells = []
        for a in acts:
            m, s = mean_std([fn(r) for r in runs[a]])
            cells.append(f"{m:.4f} ± {s:.4f}" if abs(m) < 10
                         else f"{m:.2f} ± {s:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def curve(runs, act, key):
    """Seed-averaged curve (and std) of history[key] over epochs."""
    n_epochs = len(runs[act][0]["history"])
    means, stds = [], []
    for e in range(n_epochs):
        m, s = mean_std([r["history"][e][key] for r in runs[act]])
        means.append(m)
        stds.append(s)
    return list(range(n_epochs)), means, stds


def style(ax, integer_x=False):
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_D(runs):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for act in ("selu", "elu"):
        x, m, s = curve(runs, act, "D")
        ax.plot(x, m, color=COLORS[act], linewidth=2, label=LABELS[act])
        ax.fill_between(x, [a - b for a, b in zip(m, s)],
                        [a + b for a, b in zip(m, s)],
                        color=COLORS[act], alpha=0.2, linewidth=0)
        ax.annotate(LABELS[act], (x[-1], m[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_xlabel("Epoch (0 = initialization)")
    ax.set_ylabel("Normalization error D")
    ax.set_title("Layerwise normalization error during training")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    style(ax, integer_x=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "D_vs_epoch.png"), dpi=150)
    plt.close(fig)


def plot_layer_stats(runs):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    layers = list(range(1, 9))
    for key, ax, target, title in (
        ("layer_mean", axes[0], 0.0, "Mean of activations"),
        ("layer_var", axes[1], 1.0, "Variance of activations"),
    ):
        ax.axhline(target, color="#9c9c9c", linewidth=1, linestyle=":")
        for act in ("selu", "elu"):
            for idx, ls, when in ((0, "--", "init"), (-1, "-", "trained")):
                vals = [st.mean(r["history"][idx][key][l] for r in runs[act])
                        for l in range(8)]
                ax.plot(layers, vals, color=COLORS[act], linestyle=ls,
                        linewidth=2, marker="o", markersize=4,
                        label=f"{LABELS[act]} ({when})")
        ax.set_xlabel("Hidden layer")
        ax.set_title(title)
        ax.set_xticks(layers)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9)
    fig.suptitle("Per-layer statistics, averaged over seeds "
                 "(dotted line = SNN fixed point)", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "layer_stats.png"), dpi=150)
    plt.close(fig)


def plot_val_curves(runs):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for key, ax, title, scale in (
        ("val_acc", axes[0], "Validation accuracy (%)", 100),
        ("val_loss", axes[1], "Validation cross-entropy", 1),
    ):
        for act in ("selu", "elu"):
            x, m, s = curve(runs, act, key)
            m = [v * scale for v in m]
            s = [v * scale for v in s]
            ax.plot(x, m, color=COLORS[act], linewidth=2, label=LABELS[act])
            ax.fill_between(x, [a - b for a, b in zip(m, s)],
                            [a + b for a, b in zip(m, s)],
                            color=COLORS[act], alpha=0.2, linewidth=0)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        style(ax, integer_x=True)
    axes[0].set_ylim(80, 100)
    axes[1].set_ylim(0, 0.6)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "val_curves.png"), dpi=150)
    plt.close(fig)


def main():
    runs = load_runs()
    os.makedirs(FIG_DIR, exist_ok=True)
    table = summary_table(runs)
    with open("results/summary.md", "w") as f:
        f.write(table)
    print(table)
    plot_D(runs)
    plot_layer_stats(runs)
    plot_val_curves(runs)
    print(f"Figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
