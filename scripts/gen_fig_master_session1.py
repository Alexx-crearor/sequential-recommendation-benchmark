"""Academic vector/300-DPI plots from verified session-one aggregate tables."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _bootstrap
from src.analysis.master_session1 import MODELS, LABELS, SEEDS

COLORS = dict(zip(MODELS, ["#777777", "#56B4E9", "#E69F00", "#0072B2", "#009E73", "#D55E00"]))


def generate(root, collection="master_session1", e2_complete=False):
    root = Path(root)
    tables, out = root / "results/tables" / collection, root / "results/figures" / collection
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(tables / "metric_summary.csv")
    runs = pd.read_csv(tables / "runs.csv")
    per_seed = pd.read_csv(tables / "per_seed_metrics.csv")
    bootstrap = pd.read_csv(tables / "paired_bootstrap.csv")
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 7, "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .15, "pdf.fonttype": 42})
    files = []

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = out / f"{name}.{extension}"
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", metadata=metadata)
            files.append(path)
        plt.close(fig)

    e1 = summary[(summary.experiment == "E1") & (summary.group == "full")]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), layout="constrained")
    for ax, metric in zip(axes, ("recall@10", "ndcg@10", "mrr@10")):
        data = e1[e1.metric == metric].set_index("model").loc[list(MODELS)]
        ax.bar(range(6), data["mean"], color=[COLORS[m] for m in MODELS])
        for index, row in enumerate(data.itertuples()):
            if row.n_runs > 1:
                ax.errorbar(index, row.mean, yerr=row.sd, fmt="none", color="black", capsize=3)
        ax.set(title=metric.upper(), ylabel="Test metric")
        ax.set_xticks(range(6), [LABELS[m] for m in MODELS], rotation=35, ha="right")
    fig.suptitle("E1 Beauty: mean ± sample SD over three neural seeds; MostPopular measured once", fontsize=10)
    save(fig, "e1_test_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), layout="constrained")
    for ax, family in zip(axes, ("recall", "ndcg")):
        for model in MODELS:
            rows = e1[(e1.model == model) & e1.metric.str.startswith(family+"@")].copy()
            rows["k"] = rows.metric.str.split("@").str[1].astype(int)
            rows = rows.sort_values("k")
            ax.plot(rows.k, rows["mean"], color=COLORS[model], marker="o", label=LABELS[model])
            if rows.n_runs.min() > 1:
                ax.fill_between(rows.k, rows["mean"]-rows.sd, rows["mean"]+rows.sd, color=COLORS[model], alpha=.1)
        ax.set(xlabel="K", ylabel=f"Test {family.upper()}@K", xticks=[5, 10, 20])
    axes[0].legend(ncol=2, fontsize=6.5)
    save(fig, "e1_recall_ndcg_at_k")

    fig, ax = plt.subplots(figsize=(7.2, 3.2), layout="constrained")
    for i, model in enumerate(m for m in MODELS if m != "sasrec_ce"):
        pair = bootstrap[(bootstrap.experiment == "E1") & (bootstrap.metric == "ndcg@10")
                         & (((bootstrap.left == model) & (bootstrap.right == "sasrec_ce")) | ((bootstrap.right == model) & (bootstrap.left == "sasrec_ce")))].iloc[0]
        delta, low, high = pair.difference, pair.ci_low, pair.ci_high
        if pair.left != model:
            delta, low, high = -delta, -high, -low
        ax.errorbar(delta, i, xerr=[[delta-low], [high-delta]], fmt="o", capsize=4, color=COLORS[model])
    ax.axvline(0, color="black", linestyle="--", linewidth=.9)
    ax.set_yticks(range(5), [LABELS[m]+" − SASRec-CE" for m in MODELS if m != "sasrec_ce"])
    ax.set(xlabel="Test NDCG@10 difference; pointwise paired-user 95% percentile CI",
           title="10,000 resamples; contributions averaged within user across seeds")
    save(fig, "e1_paired_bootstrap")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1), layout="constrained")
    data = per_seed[(per_seed.experiment == "E2") & (per_seed.group == "full") & (per_seed.metric == "ndcg@10")]
    for index, seed in enumerate(SEEDS):
        rows = data[data.seed == seed]
        x = rows.variant.map({f"A{i}": i-1 for i in range(1, 5)})
        axes[0].scatter(x+(index-1)*.12, rows.value, marker=["o", "s", "^"][index], label=f"seed {seed}")
    axes[0].set(xticks=range(4), xticklabels=["A1", "A2", "A3", "A4"], ylabel="Observed test NDCG@10",
                title="E2 complete: all twelve observed runs" if e2_complete else "E2 partial: six observed runs only")
    axes[0].legend()
    matrix = np.zeros((3, 4))
    for i, seed in enumerate(SEEDS):
        for j in range(4):
            matrix[i, j] = int(((data.seed == seed) & (data.variant == f"A{j+1}")).any())
    axes[1].pcolormesh(np.arange(5)-.5, np.arange(4)-.5, matrix,
                       cmap=matplotlib.colors.ListedColormap(["#eeeeee", "#b3d6c2"]), vmin=0, vmax=1,
                       edgecolors="white", linewidth=.8, rasterized=False)
    axes[1].set_xlim(-.5, 3.5)
    axes[1].set_ylim(2.5, -.5)
    for i in range(3):
        for j in range(4):
            axes[1].text(j, i, "SUCCESS" if matrix[i, j] else "PENDING", ha="center", va="center", fontsize=8)
    axes[1].set(xticks=range(4), xticklabels=["A1", "A2", "A3", "A4"], yticks=range(3), yticklabels=list(SEEDS), title="Completion, not estimated performance")
    axes[1].grid(False)
    save(fig, "e2_complete_results" if e2_complete else "e2_partial_results")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), layout="constrained")
    for ax, metric, label in zip(axes, ("training_seconds", "inference_mean_ms", "peak_gpu_memory_mib"),
                                ("Logged training seconds (log scale)", "Batch-1 scoring latency (ms)", "Peak allocated GPU memory (MiB)")):
        for i, model in enumerate(MODELS):
            values = runs.loc[(runs.experiment == "E1") & (runs.model == model), metric].dropna()
            if len(values):
                ax.bar(i, values.mean(), color=COLORS[model])
                if len(values) > 1:
                    ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt="none", color="black", capsize=3)
            else:
                ax.text(i, .02, "CPU/N/A", transform=ax.get_xaxis_transform(), rotation=90, ha="center", fontsize=7)
        ax.set(ylabel=label)
        ax.set_xticks(range(6), [LABELS[m] for m in MODELS], rotation=35, ha="right")
    axes[0].set_yscale("log")
    fig.suptitle("Archived timings: MostPopular CPU fit/scoring; neural Tesla T4. Dedicated E7 DEFERRED.", fontsize=9)
    save(fig, "recorded_efficiency")

    groups = pd.read_csv(tables / "e8_error_groups.csv")
    grouped = groups.groupby("category").fraction.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(7.6, 3.1), layout="constrained")
    names = ["sasrec_only_vs_bert", "bert_only_vs_sasrec", "fenrec_only_vs_sasrec", "all_six_incorrect"]
    ax.barh(range(4), grouped.loc[names, "mean"]*100, xerr=grouped.loc[names, "std"]*100, color=["#0072B2", "#009E73", "#D55E00", "#777777"], capsize=3)
    ax.set_yticks(range(4), ["SASRec hits, BERT misses", "BERT hits, SASRec misses", "FENRec hits, SASRec misses", "All six models miss"])
    ax.set(xlabel="User fraction (%) at K=10; mean ± sample SD over three seeds", title="E8 descriptive groups; pairwise groups may overlap")
    save(fig, "e8_error_groups")

    segments = pd.read_csv(tables / "e8_segments.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), layout="constrained")
    for ax, dimension, labels in zip(axes, ("history_length", "target_training_popularity"),
                                    (["1-5", "6-10", "11-20", "21-50", "51+"], ["cold_0", "1-5", "6-10", "11-20", "21+"])):
        for model in MODELS:
            data = segments[(segments.model == model) & (segments.dimension == dimension) & (segments.metric == "ndcg@10")]
            values = data.groupby("bin").value.mean().reindex(labels)
            ax.plot(range(5), values, marker="o", label=LABELS[model], color=COLORS[model], linewidth=1.3)
        ax.set_xticks(range(5), labels)
        ax.set(xlabel="History length" if dimension == "history_length" else "Target frequency in training interactions", ylabel="Test NDCG@10 within segment")
    axes[0].legend(ncol=2, fontsize=6.5)
    save(fig, "e8_history_popularity")

    history, validation = pd.read_csv(tables / "epoch_history.csv"), pd.read_csv(tables / "validation_curves.csv")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.4), layout="constrained")
    neural = [m for m in MODELS if m != "mostpopular"]
    for column, model in enumerate(neural):
        for seed, style in zip(SEEDS, ("-", "--", ":")):
            h = history[(history.experiment == "E1") & (history.model == model) & (history.seed == seed)]
            v = validation[(validation.experiment == "E1") & (validation.model == model) & (validation.seed == seed)]
            axes[0, column].plot(h.epoch+1, h.train_loss, linestyle=style, label=str(seed))
            axes[1, column].plot(v.epoch+1, v["ndcg@10"], linestyle=style, label=str(seed))
        axes[0, column].set_title(LABELS[model])
        axes[1, column].set_xlabel("Completed epoch (1-based)")
        axes[0, column].legend(title="Seed", fontsize=6)
    axes[0, 0].set_ylabel("Training objective (different scales)")
    axes[1, 0].set_ylabel("Validation NDCG@10")
    save(fig, "e1_training_validation")
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    for path in generate(parser.parse_args().root):
        print(path)
