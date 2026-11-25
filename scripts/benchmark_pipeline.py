"""Controlled short real-data benchmark; no research runs and no test evaluation."""
import argparse
import copy
import gc
import json
import subprocess
import threading
import time
import types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import _bootstrap
from src.utils.configuration import resolve_config
from src.utils.artifacts import verify_bundle, write_json
from src.models.factory import build_model
from src.training.checkpoint import seed_everything, atomic_torch_save
from src.training.trainer import training_epoch, fit_training_clusters
from src.data.neural_dataset import TrainingDataset
from src.evaluation.neural_ranking import evaluate_neural
from src.evaluation.efficiency import synchronize
from benchmarks.phase34.dataset import TrainingDataset as OriginalDataset
from benchmarks.phase34.trainer import training_epoch as original_epoch, fit_training_clusters as original_clusters
from benchmarks.phase34.ranking import evaluate_neural as original_evaluate
from benchmarks.phase34.base import SequentialModel as OriginalBase
from benchmarks.phase34.gru4rec import GRU4RecCE as OriginalGRU


class GpuSampler:
    def __init__(self):
        self.samples, self.stop_event = [], threading.Event()
    def start(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self
    def sample(self):
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
                for index, line in enumerate(result.stdout.strip().splitlines()):
                    utilization, memory = map(float, line.split(","))
                    self.samples.append({"gpu": index, "utilization_percent": utilization, "driver_memory_mib": memory})
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self.stop_event.wait(.5)
    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="+", default=["gru4rec_ce", "sasrec_bce", "sasrec_ce", "bert4rec", "fenrec"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--users", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cluster-benchmark", action="store_true")
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Optimized-only optional AMP measurement; FP32 control still runs")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Benchmark output exists; choose a new directory")
    args.output.mkdir(parents=True)
    manifest = verify_bundle(args.bundle)
    parts = {key: pd.read_parquet(args.bundle / f"{key}.parquet") for key in ("train", "validation", "test")}
    num_items = len(json.loads((args.bundle / "mappings.json").read_text())["items"])
    torch.set_num_threads(4)
    sampler = GpuSampler().start()
    rows, correctness = [], []
    write_json(args.output / "environment.json", {"torch": str(torch.__version__), "device": args.device,
               "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None, "split_version": manifest["split_version"], "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}})
    try:
        for name in args.models:
            config = resolve_config(name, {"seed": 7, "pilot": True, "batch_size": args.batch_size, "max_train_batches": args.batches,
                          "validation_user_limit": args.users, "max_epochs": 1, "min_epochs": 1, "dropout": .2 if name != "fenrec" else .5})
            seed_everything(7)
            initial = build_model(name, num_items, config).state_dict()
            initial = {key: value.clone() for key, value in initial.items()}
            result_predictions = {}
            for mode in ("original", "optimized"):
                cls = OriginalDataset if mode == "original" else TrainingDataset
                start = time.perf_counter()
                dataset = cls(parts["train"], num_items, config)
                construction_seconds = time.perf_counter()-start
                print(f"{name} {mode} dataset seconds={construction_seconds:.3f}", flush=True)
                for repetition in range(args.repeats):
                    gc.collect()
                    if args.device.startswith("cuda"): torch.cuda.empty_cache()
                    seed_everything(7)
                    runtime = dict(config)
                    runtime["amp"] = args.amp and mode == "optimized" and name != "fenrec"
                    model = OriginalGRU(num_items, runtime) if mode == "original" and name == "gru4rec_ce" else build_model(name, num_items, runtime)
                    if mode == "original":
                        model.representations = types.MethodType(OriginalBase.representations, model)
                    model.load_state_dict(initial)
                    model.to(args.device)
                    cluster_seconds = None
                    if name == "fenrec":
                        if args.cluster_benchmark and repetition == 0:
                            synchronize(args.device); start = time.perf_counter()
                            (original_clusters if mode == "original" else fit_training_clusters)(model, dataset, runtime, args.device)
                            synchronize(args.device); cluster_seconds = time.perf_counter()-start
                            centers = model.cluster_centers.detach().cpu()
                            atomic_torch_save(args.output / f"{mode}_centers.pt", centers)
                        else:
                            path = args.output / f"{mode}_centers.pt"
                            model.set_clusters(torch.load(path, weights_only=True) if path.exists() else torch.randn(runtime["intent_num"], runtime["hidden_dim"]))
                    optimizer = torch.optim.Adam(model.parameters(), lr=runtime["learning_rate"], betas=(.9, runtime["adam_beta2"]))
                    scaler = torch.amp.GradScaler("cuda", enabled=runtime["amp"])
                    function = original_epoch if mode == "original" else training_epoch
                    # One sacrificial warm-up update, then restore all model/optimizer/RNG state.
                    warm_config = {**runtime, "max_train_batches": 1}
                    function(model, dataset, optimizer, scaler, warm_config, 21, args.device)
                    model.load_state_dict(initial)
                    optimizer = torch.optim.Adam(model.parameters(), lr=runtime["learning_rate"], betas=(.9, runtime["adam_beta2"]))
                    scaler = torch.amp.GradScaler("cuda", enabled=runtime["amp"])
                    seed_everything(7)
                    if args.device.startswith("cuda"): torch.cuda.reset_peak_memory_stats(args.device)
                    synchronize(args.device); start = time.perf_counter()
                    values = function(model, dataset, optimizer, scaler, runtime, 21, args.device)
                    synchronize(args.device); training_seconds = time.perf_counter()-start
                    peak = torch.cuda.max_memory_allocated(args.device) if args.device.startswith("cuda") else 0
                    evaluator = original_evaluate if mode == "original" else evaluate_neural
                    evaluator(model, parts, "validation", runtime["item_batch_size"], 128)
                    synchronize(args.device); start = time.perf_counter()
                    metrics, predictions = evaluator(model, parts, "validation", runtime["item_batch_size"], args.users)
                    synchronize(args.device); validation_seconds = time.perf_counter()-start
                    if repetition == 0:
                        result_predictions[mode] = predictions
                    full_validation_seconds = None
                    if args.full_validation and mode == "optimized" and repetition == 0:
                        synchronize(args.device); start = time.perf_counter()
                        evaluate_neural(model, parts, "validation", runtime["item_batch_size"])
                        synchronize(args.device); full_validation_seconds = time.perf_counter()-start
                    synchronize(args.device); start = time.perf_counter()
                    atomic_torch_save(args.output / "checkpoint_timing.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict()})
                    checkpoint_seconds = time.perf_counter()-start
                    batches = values["batches"]
                    examples = values.get("examples", batches*runtime["batch_size"])
                    rows.append({"model": name, "mode": mode, "repetition": repetition, "batch_size": runtime["batch_size"], "amp": runtime["amp"],
                        "batches": batches, "measured_training_seconds": training_seconds, "seconds_per_batch": training_seconds/batches,
                        "examples_per_second": examples/training_seconds, "training_samples": len(dataset), "dataset_construction_seconds": construction_seconds,
                        "measured_validation_users": args.users, "validation_seconds": validation_seconds, "full_validation_seconds": full_validation_seconds,
                        "cluster_seconds": cluster_seconds, "checkpoint_seconds": checkpoint_seconds, "peak_gpu_memory_mib": peak/1048576,
                        "train_loss": values["train_loss"]})
                    pd.DataFrame(rows).to_csv(args.output / "measurements.csv", index=False)
                    print(f"{name} {mode} repeat={repetition} batch={training_seconds/batches:.4f}s validation={validation_seconds:.3f}s", flush=True)
                    del model, optimizer, scaler
                del dataset
            a, b = result_predictions["original"], result_predictions["optimized"]
            rank_mismatches = int((a.target_rank.to_numpy() != b.target_rank.to_numpy()).sum())
            top_mismatches = sum(list(x) != list(y) for x, y in zip(a.top_k, b.top_k))
            correctness.append({"model": name, "rank_mismatches_after_training": rank_mismatches, "top_k_mismatches_after_training": top_mismatches,
                                "note": "AMP or differing stochastic arithmetic may change trained weights; exact ranking equivalence is separately tested with identical weights"})
            write_json(args.output / "correctness.json", correctness)
    finally:
        sampler.stop()
        write_json(args.output / "gpu_samples.json", sampler.samples)
    print(args.output)


if __name__ == "__main__":
    main()
