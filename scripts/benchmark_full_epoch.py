"""One optimized complete training epoch per selected model; validation only, no grid."""
import argparse
import gc
import json
import time
from pathlib import Path
import pandas as pd
import torch
import _bootstrap
from src.utils.configuration import resolve_config
from src.utils.artifacts import verify_bundle, write_json
from src.training.trainer import training_epoch, fit_training_clusters
from src.training.checkpoint import seed_everything
from src.models.factory import build_model
from src.data.neural_dataset import TrainingDataset
from src.evaluation.neural_ranking import evaluate_neural
from src.evaluation.efficiency import synchronize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--models", nargs="+", default=["gru4rec_ce", "sasrec_bce", "sasrec_ce", "bert4rec", "fenrec"])
    args = parser.parse_args()
    if args.output.exists(): parser.error("Choose a new output directory")
    args.output.mkdir(parents=True)
    manifest = verify_bundle(args.bundle)
    parts = {key: pd.read_parquet(args.bundle/f"{key}.parquet") for key in ("train", "validation", "test")}
    num_items = len(json.loads((args.bundle/"mappings.json").read_text())["items"])
    rows = []
    torch.set_num_threads(4)
    for name in args.models:
        seed_everything(7)
        config = resolve_config(name, {"pilot": True, "batch_size": args.batch_size, "max_train_batches": None})
        dataset = TrainingDataset(parts["train"], num_items, config)
        model = build_model(name, num_items, config).to(args.device)
        synchronize(args.device); start = time.perf_counter()
        cluster = fit_training_clusters(model, dataset, config, args.device) or {}
        synchronize(args.device); cluster_seconds = time.perf_counter()-start
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(.9, config["adam_beta2"]))
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        if args.device.startswith("cuda"): torch.cuda.reset_peak_memory_stats(args.device)
        synchronize(args.device); start = time.perf_counter()
        # Epoch index 21 exercises the post-warm-up objective; this is timing, not convergence.
        values = training_epoch(model, dataset, optimizer, scaler, config, 21, args.device)
        synchronize(args.device); training_seconds = time.perf_counter()-start
        synchronize(args.device); start = time.perf_counter()
        evaluate_neural(model, parts, "validation", config["item_batch_size"])
        synchronize(args.device); validation_seconds = time.perf_counter()-start
        rows.append({"model": name, "batch_size": args.batch_size, "full_epoch_training_seconds": training_seconds,
                     "cluster_seconds": cluster_seconds, **cluster, "full_validation_seconds": validation_seconds,
                     "batches": values["batches"], "examples": values["examples"], "examples_per_second": values["examples"]/training_seconds,
                     "seconds_per_batch": training_seconds/values["batches"], "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(args.device)/1048576 if args.device.startswith("cuda") else 0.})
        pd.DataFrame(rows).to_csv(args.output/"full_epoch_measurements.csv", index=False)
        print(f"{name}: full epoch={training_seconds:.2f}s clustering={cluster_seconds:.2f}s full validation={validation_seconds:.2f}s", flush=True)
        del dataset, model, optimizer, scaler
        gc.collect()
        if args.device.startswith("cuda"): torch.cuda.empty_cache()
    write_json(args.output/"provenance.json", {"split_version": manifest["split_version"], "torch": str(torch.__version__),
               "device": args.device, "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
               "epoch_index_for_timing": 21, "test_evaluation": False, "research_result": False})


if __name__ == "__main__": main()
