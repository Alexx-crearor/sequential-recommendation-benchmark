"""Final-course publication figures; no training, model inference or GPU profiling."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import _bootstrap
from gen_fig_master_session1 import generate as common_figures
import matplotlib.pyplot as plt

VARIANTS = ("A1", "A2", "A3", "A4")
COLORS = ("#56B4E9", "#0072B2", "#E69F00", "#009E73")


def generate(root):
    root = Path(root)
    files = common_figures(root, "course_final", True)
    tables, out = root / "results/tables/course_final", root / "results/figures/course_final"
    summary, runs = pd.read_csv(tables / "metric_summary.csv"), pd.read_csv(tables / "runs.csv")
    e2 = summary[(summary.experiment == "E2") & (summary.group == "full")]

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = out / f"{name}.{extension}"
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
            fig.savefig(path, dpi=300, bbox_inches="tight", metadata=metadata, facecolor="white")
            files.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), layout="constrained")
    for ax, metric in zip(axes, ("recall@10", "ndcg@10", "mrr@10")):
        data = e2[e2.metric == metric].set_index("variant").loc[list(VARIANTS)]
        ax.bar(range(4), data["mean"], yerr=data.sd, color=COLORS, capsize=3)
        ax.set(title=metric.upper(), ylabel="Test metric", xticks=range(4), xticklabels=VARIANTS)
    fig.suptitle("E2 complete: mean ± sample SD across seeds 7, 42, 2026", fontsize=10)
    save(fig, "e2_test_metrics")

    bootstrap = pd.read_csv(tables / "paired_bootstrap.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), layout="constrained")
    contrasts = (("A2", "A1"), ("A3", "A2"), ("A4", "A3"))
    for ax, metric in zip(axes, ("recall@10", "ndcg@10", "mrr@10")):
        for i, (left, right) in enumerate(contrasts):
            row = bootstrap[(bootstrap.experiment == "E2") & (bootstrap.group == "full") & (bootstrap.metric == metric)
                            & (bootstrap.left == left) & (bootstrap.right == right)].iloc[0]
            ax.errorbar(row.difference, i, xerr=[[row.difference-row.ci_low], [row.ci_high-row.difference]], fmt="o", color=COLORS[i+1], capsize=4)
        ax.axvline(0, color="black", linestyle="--", linewidth=.8)
        ax.set_yticks(range(3), ["A2 − A1", "A3 − A2", "A4 − A3"])
        ax.set(xlabel="Test metric difference", title=metric.upper())
    fig.suptitle("Paired user bootstrap: pointwise 95% CI, 10,000 resamples; conditional on fitted seeds", fontsize=9)
    save(fig, "e2_paired_contrasts")

    history, validation = pd.read_csv(tables / "epoch_history.csv"), pd.read_csv(tables / "validation_curves.csv")
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.2), layout="constrained")
    for column, variant in enumerate(VARIANTS):
        for seed, style in zip((7, 42, 2026), ("-", "--", ":")):
            h = history[(history.experiment == "E2") & (history.variant == variant) & (history.seed == seed)]
            v = validation[(validation.experiment == "E2") & (validation.variant == variant) & (validation.seed == seed)]
            axes[0, column].plot(h.epoch+1, h.train_loss, linestyle=style, label=str(seed))
            axes[1, column].plot(v.epoch+1, v["ndcg@10"], linestyle=style)
        axes[0, column].set_title(variant)
        axes[0, column].legend(title="Seed", fontsize=6)
        axes[1, column].set_xlabel("Completed epoch (1-based)")
    axes[0, 0].set_ylabel("Training objective (different scales)")
    axes[1, 0].set_ylabel("Validation NDCG@10")
    save(fig, "e2_training_validation")

    e2_runs = runs[runs.experiment == "E2"].copy()
    e2_runs["training_minutes"] = e2_runs.training_seconds/60
    e2_runs["seconds_per_epoch"] = e2_runs.training_seconds/e2_runs.epochs_completed
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 3.1), layout="constrained")
    for ax, metric, ylabel in zip(axes, ("training_minutes", "seconds_per_epoch", "inference_mean_ms", "peak_gpu_memory_mib"),
                                  ("Training per run (min)", "Training per epoch (s)", "Batch-1 scoring latency (ms)", "Peak allocated GPU memory (MiB)")):
        data = e2_runs.groupby("variant")[metric].agg(["mean", "std"]).loc[list(VARIANTS)]
        ax.bar(range(4), data["mean"], yerr=data["std"], color=COLORS, capsize=3)
        ax.set(xticks=range(4), xticklabels=VARIANTS, ylabel=ylabel)
    fig.suptitle("Three recorded Kaggle runs/variant; sample SD; dedicated GPU profiling remains deferred", fontsize=9)
    save(fig, "e2_recorded_efficiency")
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    for path in generate(parser.parse_args().root):
        print(path)
