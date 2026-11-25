"""Run only the six requested seed-7 pipeline pilots, sequentially."""
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
import _bootstrap
from run_train import run
from src.training.trainer import run_neural
from src.utils.configuration import resolve_config
from src.utils.runs import verify_run
from src.utils.artifacts import write_json
from src.models.popularity import MostPopular
from src.evaluation.efficiency import inference_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="+", default=["mostpopular", "gru4rec_ce", "sasrec_bce", "sasrec_ce", "bert4rec", "fenrec"])
    parser.add_argument("--results-root", type=Path, default=Path("results/pilots"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    overrides = yaml.safe_load((_bootstrap.ROOT / "configs/experiments/pilot.yaml").read_text())
    rows = []
    for name in args.models:
        if name == "mostpopular":
            output = run(args.bundle, args.results_root, "PILOT")
        else:
            config = resolve_config(name, {**overrides, "seed": 7})
            output = run_neural(args.bundle, config, args.results_root, "PILOT", args.device, args.resume)
        verify_run(output)
        efficiency = json.loads((output / "efficiency.json").read_text())
        metrics = json.loads((output / "test_metrics.json").read_text())
        history = pd.read_csv(output / "train_history.csv")
        validation = pd.read_csv(output / "validation_metrics.csv")
        latency = efficiency.get("inference_latency")
        if latency is None and name == "mostpopular":
            predictions = pd.read_parquet(output / "predictions.parquet")
            latency = inference_profile(MostPopular.load(output / "checkpoint.pt"), predictions.iloc[0].history)
            write_json(output.parent / "mostpopular_latency_analysis.json", {"source_run": str(output), **latency})
        rows.append({"model": name, "seed": 7, "pilot": True, "final_train_loss": float(history.iloc[-1]["train_loss"]) if "train_loss" in history else None,
                     "best_validation_ndcg@10": float(validation["ndcg@10"].max()),
                     "test_ndcg@10": metrics["full"]["ndcg@10"],
                     "training_seconds": efficiency.get("total_training_seconds", efficiency.get("fit_seconds")),
                     "mean_epoch_seconds": efficiency.get("mean_epoch_seconds"),
                     "batch1_latency_ms": latency["batch1_mean_seconds"]*1000,
                     "peak_gpu_memory_mib": efficiency["peak_gpu_memory_bytes"]/1048576 if efficiency["peak_gpu_memory_bytes"] is not None else 0.,
                     "parameters": efficiency["total_parameters"], "epochs": efficiency.get("epochs_completed", 0),
                     "training_samples": efficiency.get("training_samples"), "run_path": str(output)})
        table = args.results_root / "pilot_summary.csv"
        # Merge by source run, preserving previously completed model pilot summaries.
        previous = pd.read_csv(table).to_dict("records") if table.exists() else []
        combined = {row["run_path"]: row for row in previous + rows}
        pd.DataFrame(list(combined.values())).to_csv(table, index=False)
        print(f"Recorded {name}: {output}", flush=True)


if __name__ == "__main__":
    main()
