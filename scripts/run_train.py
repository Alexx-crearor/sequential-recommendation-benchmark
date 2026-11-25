"""Phase 2 MostPopular fitting and full-catalogue evaluation."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path
import pandas as pd
import yaml
import _bootstrap
from src.data.splitting import validate_splits
from src.models.popularity import MostPopular
from src.evaluation.evaluate import evaluate
from src.utils.artifacts import fingerprint, file_hash, write_json, verify_bundle
from src.utils.runs import verify_run, REQUIRED


def run(bundle, results_root=Path("results/runs"), experiment="E1", device="cpu"):
    bundle = Path(bundle)
    manifest = verify_bundle(bundle)
    parts = {key: pd.read_parquet(bundle / f"{key}.parquet") for key in ("train", "validation", "test")}
    validate_splits(parts)
    mappings = json.loads((bundle / "mappings.json").read_text())
    config = {"model": "mostpopular", "dataset": manifest["policy"]["dataset"], "split_version": manifest["split_version"],
              "seed": 7, "ranking": "full_catalogue_history_mask_target_exempt_ties_id", "schema": 1, "implementation": "phase2-v1"}
    config_hash = fingerprint(config)
    output = Path(results_root) / experiment / config["dataset"] / "mostpopular" / config_hash / "7"
    if output.exists():
        verify_run(output)  # preserve completed runs; incomplete ones require a new attempt path
        print(f"SKIPPED verified completed run: {output}")
        return output
    output.mkdir(parents=True)
    write_json(output / "status.json", {"state": "RUNNING", "config_hash": config_hash})
    try:
        (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        write_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "device": "cpu", "requested_device": device})
        start = time.perf_counter()
        model = MostPopular(len(mappings["items"])).fit(parts["train"])
        elapsed = time.perf_counter() - start
        model.save(output / "checkpoint.pt")
        validation, _ = evaluate(model, parts, "validation")
        metrics, predictions = evaluate(model, parts, "test")
        write_json(output / "test_metrics.json", metrics)
        predictions.to_parquet(output / "predictions.parquet", index=False)
        pd.DataFrame([{"epoch": 0, "fit_seconds": elapsed, "converged": True}]).to_csv(output / "train_history.csv", index=False)
        pd.DataFrame([{"epoch": 0, **validation["full"]}]).to_csv(output / "validation_metrics.csv", index=False)
        from src.evaluation.efficiency import inference_profile
        profile = inference_profile(model, predictions.iloc[0].history)
        write_json(output / "efficiency.json", {"fit_seconds": elapsed, "total_parameters": 0, "trainable_parameters": 0, "count_entries": len(model.counts), "checkpoint_bytes": (output / "checkpoint.pt").stat().st_size, "peak_gpu_memory_bytes": None, "inference_latency": profile})
        (output / "run.log").write_text("MostPopular fitted from training events only. No stochastic training. Checkpoint is NumPy NPZ despite unified .pt suffix. Inference microbenchmark pending.\n", encoding="utf-8")
        write_json(output / "status.json", {"state": "SUCCESS", "config_hash": config_hash, "files": {name: file_hash(output / name) for name in REQUIRED}})
        verify_run(output)
    except Exception as error:
        write_json(output / "status.json", {"state": "FAILED", "error": repr(error), "config_hash": config_hash})
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--experiment", default="E1", choices=["E1", "E2", "E3", "E4", "E5", "PILOT"])
    parser.add_argument("--model", default="mostpopular", choices=["mostpopular", "gru4rec_ce", "sasrec_bce", "sasrec_ce", "bert4rec", "attention_ablation", "fenrec"])
    parser.add_argument("--seed", type=int, default=7, choices=[7, 42, 2026])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path, help="YAML overrides")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epochs", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--variant")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    args = parser.parse_args()
    if args.model == "mostpopular":
        print(run(args.bundle, args.results_root, args.experiment, args.device))
        return
    from src.utils.configuration import resolve_config
    from src.training.trainer import run_neural
    overrides = yaml.safe_load(args.config.read_text()) if args.config else {}
    for setting in args.set:
        key, value = setting.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    overrides.update(seed=args.seed, pilot=args.pilot or overrides.get("pilot", False))
    if args.variant:
        overrides["variant"] = args.variant
    config = resolve_config(args.model, overrides)
    print(run_neural(args.bundle, config, args.results_root, args.experiment, args.device, args.resume, args.stop_after_epochs))


if __name__ == "__main__":
    main()
