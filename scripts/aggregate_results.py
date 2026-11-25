"""Aggregate only verified real run artifacts. No placeholder scores."""
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
import _bootstrap
from src.utils.runs import verify_run
from src.utils.artifacts import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/runs"))
    parser.add_argument("--output", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    rows = []
    for status in args.root.rglob("status.json"):
        try:
            verify_run(status.parent)
        except Exception as error:
            print(f"Excluded {status.parent}: {error}")
            continue
        config = yaml.safe_load((status.parent / "config.yaml").read_text())
        if config.get("pilot") or config.get("profile") == "pilot" or "pilots" in status.parts or "PILOT" in status.parts:
            print(f"Excluded pilot from final research tables: {status.parent}")
            continue
        metrics = json.loads((status.parent / "test_metrics.json").read_text())
        experiment = status.parent.relative_to(args.root).parts[0]
        comparable = {key: value for key, value in config.items() if key not in ("seed", "retention_fingerprint")}
        rows.append({"variant": "default", "retention": 1., "max_seq_length": 0, **config, **metrics["full"],
                     "experiment": experiment, "comparison_hash": fingerprint(comparable), "run_path": str(status.parent)})
    if not rows:
        print("No verified runs; no performance table generated.")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "per_seed_results.csv", index=False)
    metric_keys = [key for key in frame if "@" in key]
    groups = ["dataset", "model", "variant", "retention", "max_seq_length", "comparison_hash", "split_version"]
    filenames = {"E1": "table_02_overall_performance.csv", "E2": "table_03_attention_ablation.csv", "E3": "table_04_data_sparsity.csv",
                 "E4": "table_05_fenrec_ablation.csv", "E5": "table_06_sequence_length.csv"}
    for experiment, filename in filenames.items():
        subset = frame.loc[frame.experiment == experiment]
        if len(subset):
            subset.groupby(groups, dropna=False)[metric_keys].agg(["mean", "std", "count"]).to_csv(args.output / filename)


if __name__ == "__main__":
    main()
