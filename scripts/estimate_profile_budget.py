"""Estimate profile costs from measured pipeline benchmarks; unknown axes stay unknown."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import _bootstrap
from src.utils.profiles import load_profile, profile_overrides
from src.utils.grid import expand_grid
from src.utils.configuration import resolve_config
from src.utils.artifacts import fingerprint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--full-epochs", type=Path, help="Measured complete optimized epoch CSV; preferred over short-batch extrapolation")
    parser.add_argument("--output", type=Path, default=Path("results/phase5a/budget"))
    args = parser.parse_args()
    frame = pd.read_csv(args.benchmark/"measurements.csv")
    optimized = frame.loc[(frame["mode"] == "optimized") & ~frame.amp]
    means = optimized.groupby("model").mean(numeric_only=True)
    full_epochs = pd.read_csv(args.full_epochs).set_index("model") if args.full_epochs else None
    train = pd.read_parquet(args.bundle/"train.parquet")
    counts = train.groupby("user_id").size().to_numpy()
    summaries = []
    args.output.mkdir(parents=True, exist_ok=True)
    for name, final in [("pilot", False), ("core_research", False), ("core_research", True), ("full_research", True)]:
        grid, profile = load_profile(name, final)
        entries = expand_grid(grid)
        seen, rows = set(), []
        for row in entries:
            row = dict(row)
            row["profile"], row["final_seeds"] = name, final
            row["estimate_kind"] = "not_training"
            if "task" in row:
                row["estimate_kind"] = "analysis_not_estimated"
            elif row["model"] == "tiger":
                row["estimate_kind"] = "unknown_unimplemented_tiger"
            elif row["dataset"] != "beauty":
                row["estimate_kind"] = "unknown_unmeasured_dataset"
            elif row["model"] == "mostpopular":
                row["estimated_training_hours"] = 0.
                row["estimate_kind"] = "deterministic_fit_negligible_evaluation_extra"
            else:
                variant = grid["experiments"][row["experiment"]].get("variants", {}).get(row["variant"], {})
                cfg = resolve_config(row["model"], {**profile_overrides(profile, row["model"]), **variant, "seed": row["seed"],
                                    "variant": row["variant"], "retention": row["retention"], "max_seq_length": row["max_seq_length"]})
                runtime_id = fingerprint(cfg)
                model = row["model"]
                surrogate = "bert4rec" if model == "attention_ablation" and cfg["objective"] == "masked_ce" else "sasrec_ce" if model == "attention_ablation" and cfg["objective"] == "next_ce" else "sasrec_bce" if model == "attention_ablation" else model
                if surrogate not in means.index:
                    row["estimate_kind"] = "unknown_unbenchmarked_model"
                    rows.append(row); continue
                measured = means.loc[surrogate]
                if full_epochs is not None and surrogate in full_epochs.index:
                    observed = full_epochs.loc[surrogate]
                    measured = measured.copy()
                    for field in ("seconds_per_batch", "batch_size", "cluster_seconds", "full_validation_seconds"):
                        measured[field] = observed[field]
                retained = np.maximum(1, np.floor(counts*cfg["retention"]).astype(int))
                samples = int(np.maximum(0, retained-1).sum()) if model == "fenrec" else len(counts) if cfg["objective"] == "masked_ce" else int((retained>=2).sum())
                # Linear length scaling is a labeled surrogate, not a measured L^2 model.
                scale = cfg["max_seq_length"]/50
                batches = math.ceil(samples/cfg["batch_size"])
                if cfg["pilot"] and cfg["max_train_batches"] is not None:
                    batches = min(batches, cfg["max_train_batches"])
                train_seconds = batches*float(measured.seconds_per_batch)
                # Times are used at the benchmark's physical batch size only.
                if cfg["batch_size"] != int(measured.batch_size):
                    train_seconds *= cfg["batch_size"]/float(measured.batch_size)
                train_seconds *= scale
                cluster = 0.
                if model == "fenrec" and cfg["contrastive"] != "none" and cfg["beta_0"]:
                    full_prefixes = int(np.maximum(0, counts-1).sum())
                    cluster = float(measured.cluster_seconds) * (samples/full_prefixes) * scale if pd.notna(measured.cluster_seconds) else 0.
                full_validation = measured.full_validation_seconds
                if pd.isna(full_validation):
                    full_validation = measured.validation_seconds*len(counts)/measured.measured_validation_users
                validation = float(full_validation)*scale
                if cfg["pilot"] and cfg["validation_user_limit"] is not None:
                    validation *= min(1., cfg["validation_user_limit"]/len(counts))
                checks = 1 + cfg["max_epochs"]//cfg["validation_every"]
                max_seconds = (train_seconds+cluster)*cfg["max_epochs"] + checks*validation
                minimum = max(cfg["min_epochs"], cfg.get("warm_up_epochs", 0)+cfg.get("min_post_warmup_epochs", 0) if model == "fenrec" and cfg["enduring"] else 1)
                minimum = min(minimum, cfg["max_epochs"])
                floor_seconds = (train_seconds+cluster)*minimum + (1+minimum//cfg["validation_every"])*validation
                row.update(estimated_training_hours=max_seconds/3600, minimum_budget_hours=floor_seconds/3600,
                           estimated_epoch_training_seconds=train_seconds+cluster, estimated_full_validation_seconds=validation,
                           max_epochs=cfg["max_epochs"], validation_every=cfg["validation_every"], benchmark_surrogate=surrogate,
                           estimate_kind="extrapolated_beauty_length_retention_and_variant_surrogates")
                if runtime_id in seen:
                    row["estimated_training_hours"] = 0.
                    row["minimum_budget_hours"] = 0.
                    row["estimate_kind"] = "reuse_identical_config"
                seen.add(runtime_id)
            rows.append(row)
        label = name + ("_final" if final else "_initial")
        pd.DataFrame(rows).to_csv(args.output/f"{label}_manifest.csv", index=False)
        summaries.append({"profile": label, "entries": len(rows), "known_beauty_budget_floor_hours": sum(row.get("minimum_budget_hours", 0) for row in rows),
                          "known_beauty_max_budget_hours": sum(row.get("estimated_training_hours", 0) for row in rows),
                          "unknown_entries": sum(row["estimate_kind"].startswith("unknown") for row in rows),
                          "reused_entries": sum(row["estimate_kind"] == "reuse_identical_config" for row in rows),
                          "note": "Budget floor is minimum permitted training, not predicted convergence. Excludes tuning, final test evaluation, E7/E8, setup and export. Full-profile unknown costs are additional, not zero."})
    write_json(args.output/"profile_budget.json", {"benchmark": str(args.benchmark), "full_epoch_measurements": str(args.full_epochs) if args.full_epochs else None, "estimates": summaries})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__": main()
