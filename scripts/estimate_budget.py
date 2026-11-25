"""Measured-pilot extrapolation, explicitly not a throughput guarantee."""
import argparse
import json
import math
from pathlib import Path
import pandas as pd
import yaml
import _bootstrap
from src.utils.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/pilots"))
    parser.add_argument("--output", type=Path, default=Path("results/reports/pilot_budget_estimate.json"))
    args = parser.parse_args()
    rows = []
    for status_path in args.root.rglob("status.json"):
        directory = status_path.parent
        status = json.loads(status_path.read_text())
        if status["state"] != "SUCCESS":
            continue
        config = yaml.safe_load((directory / "config.yaml").read_text())
        if config["model"] == "mostpopular":
            continue
        efficiency = json.loads((directory / "efficiency.json").read_text())
        history = pd.read_csv(directory / "train_history.csv")
        batch_count = math.ceil(efficiency["training_samples"] / config["batch_size"])
        observed_batches = float(history.batches.mean())
        if "optimization_seconds" in history:
            compute = float(history.optimization_seconds.mean()) * batch_count / observed_batches + float(history.cluster_seconds.mean())
            method = "measured_optimization_scaled_plus_measured_clustering"
        elif config["model"] != "fenrec":
            compute = float(history.epoch_seconds.mean()) * batch_count / observed_batches
            method = "entire_epoch_scaled_including_clustering_upper_extrapolation"
        else:
            compute = None
            method = "unavailable_fixed_clustering_cost_not_separated_in_this_earlier_pilot"
        # All Beauty users, based on deterministic 512-user validation pilot.
        test_users = json.loads((directory / "test_metrics.json").read_text())["full"]["users"]
        val_users = int(pd.read_csv(directory / "validation_metrics.csv").users.iloc[0])
        validation = float(history.validation_seconds.mean()) * test_users / val_users
        rows.append({"model": config["model"], "variant": config["variant"], "source_run": str(directory), "device_type": config["device_type"],
                     "pilot_batch_size": config["batch_size"], "observed_batches_per_epoch": observed_batches, "full_batches_per_epoch": batch_count,
                     "estimated_full_epoch_training_seconds": compute, "estimated_full_validation_seconds": validation,
                     "estimated_300_epoch_training_and_validation_hours": (compute+validation)*300/3600 if compute is not None else None,
                     "method": method, "warm_up_epochs": config.get("warm_up_epochs"),
                     "limits": "Beauty only; same batch size and hardware; no early stopping credit; not measured post-warmup cost unless warm_up_epochs=0; excludes tuning and one-time final evaluation"})
    write_json(args.output, {"estimates": rows, "grid_extrapolation": "Do not extrapolate Beauty timings to Sports/Toys or sequence-length/retention settings without pilots. FENRec warm-up and full batch-size scaling need additional measurements."})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
