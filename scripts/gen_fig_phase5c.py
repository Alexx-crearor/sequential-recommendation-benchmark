"""Reproducible academic figures from verified Phase 5C aggregate CSVs only."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import _bootstrap
from src.analysis.phase5c import MODELS, LABELS

COLORS = ["#0072B2", "#D55E00", "#009E73"]
STYLES = ["-", "--", "-."]


def generate_figures(root):
    root = Path(root)
    tables, output = root / "results/tables/phase5c", root / "results/figures/phase5c"
    output.mkdir(parents=True, exist_ok=True)
    trials = pd.read_csv(tables / "tuning_trials.csv")
    history = pd.read_csv(tables / "epoch_history.csv")
    validation = pd.read_csv(tables / "validation_curves.csv")
    selected = trials[trials.selected].set_index("model").loc[list(MODELS)]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                         "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .16, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 300, "figure.dpi": 120})
    files = []

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            files.append(path)
        plt.close(fig)

    for kind, data, metric, ylabel in (("validation_curves", validation, "ndcg@10", "Validation NDCG@10"),
                                        ("training_loss", history, "train_loss", "Training objective (model-specific)")):
        fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.8), layout="constrained")
        for model, ax in zip(MODELS, axes.flat):
            winning = int(selected.loc[model, "trial"])
            for trial in (0, 1, 2):
                points = data[(data.model == model) & (data.trial == trial)]
                label = f"T{trial}" + (" (selected)" if trial == winning else "")
                ax.plot(points.epoch_display, points[metric], color=COLORS[trial], linestyle=STYLES[trial],
                        linewidth=2.1 if trial == winning else 1.1, label=label)
            if model == "fenrec":
                ax.axvline(20, color=".5", linewidth=.8, linestyle=":", label="Warm-up budget")
            ax.set(title=LABELS[model], xlabel="Completed epoch (1-based)", ylabel=ylabel)
            ax.legend(loc="best")
        axes.flat[-1].axis("off")
        axes.flat[-1].text(0, .85, "Beauty · seed 7 · three trials/model\n\nValidation shows actual checks only.\nSkipped checks are not observations.\n\nNo test scores or uncertainty estimates.\nLoss scales differ across objectives.",
                           transform=axes.flat[-1].transAxes, va="top", linespacing=1.5)
        save(fig, kind)

    fig, axes = plt.subplots(1, 5, figsize=(10.2, 2.8), layout="constrained")
    for model, ax in zip(MODELS, axes):
        rows = trials[trials.model == model].sort_values("trial")
        bars = ax.bar(np.arange(3), rows["validation_ndcg@10"], color=COLORS, width=.65)
        for bar, row in zip(bars, rows.itertuples()):
            if row.selected:
                bar.set_edgecolor("black")
                bar.set_linewidth(1.6)
                bar.set_hatch("//")
            ax.annotate(f"{rows.loc[rows.trial == row.trial, 'validation_ndcg@10'].iloc[0]:.4f}",
                        (bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
        ax.set_xticks(range(3), ["T0", "T1", "T2"])
        ax.set_title(LABELS[model])
        ax.set_ylim(0, .085)
        ax.set_xlabel("Trial (hatched = selected)")
    axes[0].set_ylabel("Best validation NDCG@10")
    save(fig, "trial_validation_comparison")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), layout="constrained")
    positions = np.arange(len(MODELS))
    bottom = np.zeros(len(MODELS))
    components = [(selected.training_seconds-selected.clustering_seconds, "Optimization", "#0072B2"),
                  (selected.clustering_seconds, "Clustering (inside training)", "#E69F00"),
                  (selected.validation_seconds, "Validation checks", "#009E73"),
                  (selected.checkpoint_seconds, "Checkpoint writes", "#CC79A7")]
    for values, label, color in components:
        axes[0].bar(positions, values/60, bottom=bottom, label=label, color=color)
        bottom += values.to_numpy()/60
    axes[0].set(ylabel="Logged time (minutes)", title="Selected trials: recorded components")
    axes[0].legend(fontsize=6.5)
    axes[1].bar(positions, selected.peak_gpu_memory_mib, color="#0072B2")
    axes[1].set(ylabel="Peak allocated GPU memory (MiB)", title="Selected trials: Tesla T4")
    axes[2].bar(positions-.18, selected.epochs_completed, width=.36, color="#0072B2", label="Completed")
    axes[2].bar(positions+.18, selected.best_epoch, width=.36, color="#009E73", label="Best validation")
    axes[2].scatter(positions, selected.max_epochs, marker="_", s=160, color="black", label="Epoch cap")
    axes[2].set(ylabel="Epoch (1-based)", title="Stopping budgets and selected epochs")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(positions, [LABELS[m] for m in MODELS], rotation=35, ha="right")
    save(fig, "selected_trial_resources")
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    args = parser.parse_args()
    for path in generate_figures(args.root):
        print(path)
