"""Validation-selected, deterministic epoch-boundary resumable training."""
import json
import platform
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
from sklearn.cluster import KMeans
from src.data.neural_dataset import TrainingDataset, verify_training_boundary
from src.data.splitting import validate_splits, evaluation_cases
from src.data.sparsity import subsample_train
from src.models.factory import build_model
from src.training.checkpoint import seed_everything, atomic_torch_save, rng_state, restore_rng
from src.training.losses import objective_loss
from src.evaluation.neural_ranking import evaluate_neural
from src.evaluation.efficiency import synchronize, inference_profile
from src.utils.artifacts import verify_bundle, fingerprint, write_json, file_hash
from src.utils.runs import verify_run, REQUIRED


def common_warm_analysis(original_train, parts, predictions, seed, ratios=(.2, .4, .6, .8, 1.)):
    from src.evaluation.metrics import contributions
    common = set(map(int, parts["test"].user_id))
    for ratio in ratios:
        retained, _ = subsample_train(original_train, ratio, seed)
        warm = set(retained.item_id)
        common &= set(parts["test"].loc[parts["test"].item_id.isin(warm), "user_id"])
    subset = predictions.loc[predictions.user_id.isin(common)]
    metrics = {"users": len(subset), **{key: float(subset[key].mean()) if len(subset) else None for key in contributions(1)}}
    return sorted(map(int, common)), metrics


def training_epoch(model, dataset, optimizer, scaler, config, epoch, device):
    from itertools import islice
    from src.training.loading import make_loader
    loader = make_loader(dataset, config, device)
    model.train()
    losses, components, examples, optimizer_steps = [], [], 0, 0
    accumulation = config.get("gradient_accumulation_steps", 1)
    if model.config["model"] == "fenrec" and model.contrastive != "none" and accumulation != 1:
        raise ValueError("Cannot split FENRec contrastive negative pools with accumulation")
    iterator = iter(loader)
    cap = config["max_train_batches"]
    if cap is not None:
        iterator = islice(iterator, cap)
    while True:
        group = list(islice(iterator, accumulation))
        if not group:
            break
        counts = [int(batch["targets"].ne(0).sum()) if "targets" in batch else len(batch["tokens"]) for batch in group]
        total = sum(counts)
        optimizer.zero_grad(set_to_none=True)
        for batch, count in zip(group, counts):
            examples += len(batch["tokens"])
            batch = {key: value.to(device, non_blocking=config.get("pin_memory", True)) for key, value in batch.items()}
            with torch.autocast(device_type=torch.device(device).type, enabled=config["amp"], dtype=torch.float16):
                loss, values = objective_loss(model, batch, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss * (count/total)).backward()
            losses.append(float(loss.detach()))
            components.append(values)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        optimizer_steps += 1
        with torch.no_grad():
            model.embedding.weight[0].zero_()
    if not losses:
        raise ValueError("Empty training epoch")
    return {"train_loss": float(np.mean(losses)), "batches": len(losses), "examples": examples, "optimizer_steps": optimizer_steps,
            **{key: float(np.mean([row[key] for row in components])) for key in components[0]}}


@torch.no_grad()
def fit_training_clusters(model, dataset, config, device):
    if model.config["model"] != "fenrec" or model.contrastive == "none" or not config["beta_0"]:
        return
    model.eval()
    import time
    start_time = time.perf_counter()
    vectors = []
    use_gpu_buffer = str(device).startswith("cuda") and len(dataset.samples)*model.dim*4 <= config.get("cluster_gpu_buffer_mb", 128)*1048576
    buffer = torch.empty(len(dataset.samples), model.dim, device=device) if use_gpu_buffer else None
    for start in range(0, len(dataset.samples), config["eval_batch_size"]):
        from src.data.neural_dataset import left_pad
        end = min(len(dataset.samples), start+config["eval_batch_size"])
        cached = getattr(dataset, "cached_tokens", None)
        tokens = cached[start:end] if cached is not None else torch.stack([left_pad(sample["history"], dataset.length) for sample in dataset.samples[start:end]])
        encoded = model(tokens.to(device))[:, -1].float()
        if use_gpu_buffer:
            buffer[start:end] = encoded
        else:
            vectors.append(encoded.cpu().numpy())
    vectors = buffer.cpu().numpy() if use_gpu_buffer else np.concatenate(vectors)
    encoding_seconds = time.perf_counter()-start_time
    count = config["intent_num"]
    if len(vectors) < count:
        raise ValueError("intent_num exceeds number of training prefixes; use an explicit smaller smoke configuration")
    cluster = KMeans(n_clusters=count, random_state=config["cluster_seed"], n_init=config["cluster_restarts"],
                     max_iter=config["cluster_iterations"], algorithm="lloyd").fit(vectors)
    model.set_clusters(torch.from_numpy(cluster.cluster_centers_))
    return {"cluster_encoding_seconds": encoding_seconds, "kmeans_seconds": time.perf_counter()-start_time-encoding_seconds}


def run_neural(bundle, config, results_root=Path("results/runs"), experiment="E1", device="cpu", resume=False, stop_after_epochs=None):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; choose --device cpu")
    if config["amp"] and device.type != "cuda":
        raise ValueError("AMP is enabled only on CUDA")
    if config["model"] == "fenrec" and config.get("enduring", True) and not config["pilot"] and config["max_epochs"] < config["warm_up_epochs"] + config.get("min_post_warmup_epochs", 1):
        raise ValueError("FENRec budget must include configured training after warm-up")
    torch.set_num_threads(config["cpu_threads"])
    seed_everything(config["seed"])
    bundle = Path(bundle)
    manifest = verify_bundle(bundle)
    parts = {key: pd.read_parquet(bundle / f"{key}.parquet") for key in ("train", "validation", "test")}
    validate_splits(parts)
    original_train = parts["train"]
    retention_manifest = None
    if config["retention"] != 1.:
        retained, retention_manifest = subsample_train(parts["train"], config["retention"], config["seed"])
        parts["train"] = retained
    config = {**config, "dataset": manifest["policy"]["dataset"], "split_version": manifest["split_version"],
              "retention_fingerprint": fingerprint(retention_manifest) if retention_manifest else "full",
              "device_type": device.type}
    identity = fingerprint(config)
    output = Path(results_root) / experiment / config["dataset"] / config["model"] / identity / str(config["seed"])
    # Reuse only byte-for-byte identical resolved settings and split provenance.
    if not output.exists() and experiment != "E1":
        candidate = Path(results_root) / "E1" / config["dataset"] / config["model"] / identity / str(config["seed"])
        if candidate.exists():
            verify_run(candidate)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(candidate, output)
            write_json(output / "reuse.json", {"source_run": str(candidate), "config_hash": identity})
            status = json.loads((output / "status.json").read_text())
            status["files"]["reuse.json"] = file_hash(output / "reuse.json")
            if experiment == "E3":
                predictions = pd.read_parquet(output / "predictions.parquet")
                users, common_metrics = common_warm_analysis(original_train, parts, predictions, config["seed"], config.get("common_warm_ratios", [.2, .4, .6, .8, 1.]))
                metrics = json.loads((output / "test_metrics.json").read_text())
                metrics["common_warm"] = common_metrics
                write_json(output / "test_metrics.json", metrics)
                write_json(output / "common_warm_users.json", users)
                for name in ("test_metrics.json", "common_warm_users.json"):
                    status["files"][name] = file_hash(output / name)
            write_json(output / "status.json", status)
            verify_run(output)
            print(f"REUSED verified exact-configuration E1 artifact: {output}", flush=True)
            return output
    if output.exists():
        status = json.loads((output / "status.json").read_text())
        if status["state"] == "VALIDATED":
            from src.utils.validation_runs import verify_validation_run
            verify_validation_run(output)
            return output
        if status["state"] == "SUCCESS":
            verify_run(output)
            print(f"SKIPPED verified completed run: {output}", flush=True)
            return output
        if not resume:
            raise ValueError(f"Existing {status['state']} run preserved; use --resume for epoch-boundary recovery")
        if not (output / "latest.pt").exists():
            raise ValueError("No resumable epoch checkpoint; choose a new results root")
        # Preserve previous state records before beginning a new attempt.
        attempt = sum(1 for _ in output.glob("attempt_*.json")) + 1
        shutil.copy2(output / "status.json", output / f"attempt_{attempt}.json")
    else:
        output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    write_json(output / "status.json", {"state": "RUNNING", "config_hash": identity})
    if retention_manifest:
        write_json(output / "retention_manifest.json", retention_manifest)
    try:
        mappings = json.loads((bundle / "mappings.json").read_text())
        num_items = len(mappings["items"])
        dataset = TrainingDataset(parts["train"], num_items, config)
        verify_training_boundary(dataset, parts)
        model = build_model(config["model"], num_items, config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(.9, config["adam_beta2"]), weight_decay=config["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
        history, validations, start_epoch, best, bad_epochs = [], [], 0, -1., 0
        if resume and (output / "latest.pt").exists():
            state = torch.load(output / "latest.pt", map_location=device, weights_only=True)
            if state["config_hash"] != identity:
                raise ValueError("Resume configuration mismatch")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            history, validations = state["history"], state["validations"]
            start_epoch, best, bad_epochs = state["epoch"]+1, state["best"], state["bad_epochs"]
            if "best_checkpoint" in state:
                atomic_torch_save(output / "checkpoint.pt", state["best_checkpoint"])
            restore_rng(state["rng"], device)
        environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
                       "pandas": pd.__version__, "device": str(device), "cuda_runtime": torch.version.cuda,
                       "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None}
        if not (output / "environment.json").exists():
            write_json(output / "environment.json", environment)
        elif resume:
            number = len(list(output.glob("environment_resume_*.json"))) + 1
            write_json(output / f"environment_resume_{number}.json", environment)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        completed_here = 0
        minimum_epochs = config.get("min_epochs", 1)
        if config["model"] == "fenrec" and model.enduring:
            minimum_epochs = max(minimum_epochs, config["warm_up_epochs"] + config.get("min_post_warmup_epochs", 1))
        minimum_epochs = min(minimum_epochs, config["max_epochs"])
        for epoch in range(start_epoch, config["max_epochs"]):
            if bad_epochs >= config["early_stopping_patience"] and epoch >= minimum_epochs:
                break
            synchronize(device)
            start = time.perf_counter()
            cluster_detail = fit_training_clusters(model, dataset, config, device) or {}
            synchronize(device)
            cluster_seconds = time.perf_counter()-start
            optimization_start = time.perf_counter()
            values = training_epoch(model, dataset, optimizer, scaler, config, epoch, device)
            synchronize(device)
            optimization_seconds = time.perf_counter()-optimization_start
            epoch_seconds = time.perf_counter()-start
            validation_seconds = 0.
            validate_now = epoch == 0 or (epoch+1) % config.get("validation_every", 1) == 0 or epoch+1 == config["max_epochs"]
            score = validations[-1]["ndcg@10"] if validations else 0.
            if validate_now:
                val_start = time.perf_counter()
                validation, _ = evaluate_neural(model, parts, "validation", config["item_batch_size"], config["validation_user_limit"])
                score = validation["full"]["ndcg@10"]
                validation_seconds = time.perf_counter()-val_start
                if not np.isfinite(score):
                    raise FloatingPointError("Nonfinite validation score")
                if score > best:
                    best, bad_epochs = score, 0
                    model.save(output / "checkpoint.pt")
                else:
                    bad_epochs += 1
                validations.append({"epoch": epoch, **validation["full"]})
            if epoch+1 < minimum_epochs:
                bad_epochs = 0
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            row = {"epoch": epoch, **values, "epoch_seconds": epoch_seconds, "validation_seconds": validation_seconds,
                   "cluster_seconds": cluster_seconds, "optimization_seconds": optimization_seconds, **cluster_detail, "validation_performed": validate_now,
                   "validation_ndcg@10": score, "peak_gpu_memory_bytes": peak, "checkpoint_seconds": 0.}
            history.append(row)
            pd.DataFrame(history).to_csv(output / "train_history.csv", index=False)
            pd.DataFrame(validations).to_csv(output / "validation_metrics.csv", index=False)
            checkpoint_start = time.perf_counter()
            checkpoint_now = epoch == 0 or (epoch+1) % config.get("checkpoint_every", 1) == 0 or epoch+1 == config["max_epochs"] or (bad_epochs >= config["early_stopping_patience"] and epoch+1 >= minimum_epochs) or (stop_after_epochs is not None and completed_here+1 >= stop_after_epochs)
            if checkpoint_now:
                atomic_torch_save(output / "latest.pt", {"config_hash": identity, "epoch": epoch, "model": model.state_dict(),
                                  "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(device),
                                  "history": history, "validations": validations, "best": best, "bad_epochs": bad_epochs, "best_checkpoint": torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)})
            row["checkpoint_seconds"] = time.perf_counter()-checkpoint_start
            pd.DataFrame(history).to_csv(output / "train_history.csv", index=False)
            with open(output / "run.log", "a", encoding="utf-8") as log:
                log.write(json.dumps(row, allow_nan=False) + "\n")
            print(f"{config['model']} epoch={epoch} loss={values['train_loss']:.5f} validation_ndcg@10={score:.6f} validated={validate_now} seconds={epoch_seconds:.2f} peak_gpu_mib={peak/1048576:.1f}", flush=True)
            completed_here += 1
            if stop_after_epochs is not None and completed_here >= stop_after_epochs:
                write_json(output / "status.json", {"state": "INCOMPLETE", "config_hash": identity, "last_epoch": epoch, "reason": "requested_epoch_boundary_stop"})
                return output
        if not history:
            raise ValueError("No completed training epoch")
        # Test only once with validation-selected best weights, after training stops.
        from src.models.base import SequentialModel
        best_model = SequentialModel.load(output / "checkpoint.pt", device)
        if not config.get("evaluate_test", True):
            # Strict branch: tuning never calls the test evaluator or creates test artifacts.
            validation_metrics, validation_predictions = evaluate_neural(best_model, parts, "validation", config["item_batch_size"], config["validation_user_limit"])
            write_json(output / "validation_selection.json", validation_metrics)
            validation_predictions.to_parquet(output / "validation_predictions.parquet", index=False)
            write_json(output / "efficiency.json", {"epochs_completed": len(history), "total_training_seconds": sum(row["epoch_seconds"] for row in history),
                       "total_validation_seconds": sum(row["validation_seconds"] for row in history), "best_validation_ndcg@10": best,
                       "stopped_for_patience": bad_epochs >= config["early_stopping_patience"], "training_limit_reached": len(history) >= config["max_epochs"],
                       "peak_gpu_memory_bytes": max(row["peak_gpu_memory_bytes"] for row in history), "pilot": config["pilot"], "evaluate_test": False})
            names = [name for name in REQUIRED if name not in ("test_metrics.json", "predictions.parquet")] + ["latest.pt", "validation_selection.json", "validation_predictions.parquet"]
            write_json(output / "status.json", {"state": "VALIDATED", "config_hash": identity, "files": {name: file_hash(output / name) for name in names}})
            from src.utils.validation_runs import verify_validation_run
            verify_validation_run(output)
            return output
        test_start = time.perf_counter()
        metrics, predictions = evaluate_neural(best_model, parts, "test", config["item_batch_size"])
        test_seconds = time.perf_counter()-test_start
        if experiment == "E3":
            common, common_metrics = common_warm_analysis(original_train, parts, predictions, config["seed"], config.get("common_warm_ratios", [.2, .4, .6, .8, 1.]))
            metrics["common_warm"] = common_metrics
            write_json(output / "common_warm_users.json", common)
        expected_users = set(map(int, parts["test"].user_id))
        if set(predictions.user_id) != expected_users:
            raise ValueError("Incomplete user coverage")
        write_json(output / "test_metrics.json", metrics)
        predictions.to_parquet(output / "predictions.parquet", index=False)
        predictions[["user_id", "target"]].to_csv(output / "evaluation_users.csv", index=False)
        profile = inference_profile(best_model, evaluation_cases(parts)[0]["history"])
        efficiency = {"total_parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "total_training_seconds": sum(row["epoch_seconds"] for row in history), "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
                      "total_validation_seconds": sum(row["validation_seconds"] for row in history), "test_evaluation_seconds": test_seconds,
                      "test_users_per_second": len(predictions)/test_seconds, "peak_gpu_memory_bytes": max(row["peak_gpu_memory_bytes"] for row in history),
                      "checkpoint_bytes": (output / "checkpoint.pt").stat().st_size, "inference_latency": profile,
                      "epochs_completed": len(history), "best_validation_ndcg@10": best, "training_samples": len(dataset),
                      "realized_retention": len(parts["train"])/len(original_train), "converged": bad_epochs >= config["early_stopping_patience"],
                      "training_limit_reached": len(history) >= config["max_epochs"], "pilot": config["pilot"]}
        write_json(output / "efficiency.json", efficiency)
        required = REQUIRED + ["latest.pt", "evaluation_users.csv"] + (["retention_manifest.json"] if retention_manifest else []) + (["common_warm_users.json"] if experiment == "E3" else [])
        write_json(output / "status.json", {"state": "SUCCESS", "config_hash": identity, "pilot": config["pilot"], "files": {name: file_hash(output / name) for name in required}})
        verify_run(output)
    except BaseException as error:
        write_json(output / "status.json", {"state": "INCOMPLETE" if isinstance(error, KeyboardInterrupt) else "FAILED", "config_hash": identity, "error": repr(error)})
        raise
    return output
