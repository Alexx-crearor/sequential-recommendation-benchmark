"""Audit the original Phase 5C ZIP and derive publication summaries on CPU only."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

import numpy as np
import pandas as pd
import torch
import yaml

from src.utils.artifacts import file_hash, fingerprint, write_json

ARCHIVE_NAME = "seqrec-phase5c-final-20261126_132556_321898.zip"
ARCHIVE_SHA256 = "7861c626561fc1d7914abb8ee5a7eb4d552a932aca5dfbc54419648efb0176fa"
SPLIT = "2228499f23f01d419863d0f09b3671f098672471887c4fb37b85bae26d42f91e"
MODELS = ("sasrec_ce", "sasrec_bce", "gru4rec_ce", "bert4rec", "fenrec")
LABELS = {"sasrec_ce": "SASRec-CE", "sasrec_bce": "SASRec-BCE", "gru4rec_ce": "GRU4Rec-CE", "bert4rec": "BERT4Rec", "fenrec": "FENRec"}
REQUIRED = {"checkpoint.pt", "latest.pt", "config.yaml", "efficiency.json", "environment.json", "run.log",
            "train_history.csv", "validation_metrics.csv", "validation_predictions.parquet", "validation_selection.json"}


def resolve_archive_path(root, explicit=None):
    """Use a portable repository path unless explicitly overridden; never scan Downloads."""
    root = Path(root).resolve()
    override = explicit or os.environ.get("SEQREC_PHASE5C_ARCHIVE")
    path = Path(override).expanduser() if override else root / "artifacts/phase5c" / ARCHIVE_NAME
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_verified(archive, destination, expected_sha256=ARCHIVE_SHA256):
    """Preflight all ZIP entries/conflicts, then extract missing files exclusively."""
    archive, destination = Path(archive), Path(destination).resolve()
    digest = file_hash(archive)
    require(expected_sha256 is None or digest == expected_sha256, "Unexpected source archive SHA256")
    inventory, names = [], set()
    with zipfile.ZipFile(archive) as stream:
        require(sum(i.file_size for i in stream.infolist()) <= 1024 ** 3, "Archive exceeds 1 GiB extraction limit")
        require(stream.testzip() is None, "ZIP CRC failure")
        for info in stream.infolist():
            member = PurePosixPath(info.filename)
            require(not member.is_absolute() and ".." not in member.parts and "\\" not in info.orig_filename
                    and ":" not in info.filename and info.filename not in names, "Unsafe or duplicate ZIP member")
            names.add(info.filename)
            require(not stat.S_ISLNK(info.external_attr >> 16), "ZIP symlink is forbidden")
            path = (destination / info.filename).resolve()
            require(path.is_relative_to(destination), "ZIP path escaped destination")
            if info.is_dir():
                continue
            with stream.open(info) as source:
                hasher = hashlib.sha256()
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    hasher.update(block)
            item = {"member": info.filename, "bytes": info.file_size, "sha256": hasher.hexdigest()}
            if path.exists():
                require(path.is_file() and file_hash(path) == item["sha256"], f"Existing artifact differs: {info.filename}")
            inventory.append(item)
        # No file is changed until every member and existing artifact has passed.
        for item in inventory:
            path = destination / item["member"]
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with stream.open(item["member"]) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target)
    return digest, inventory


def validate_predictions(predictions, metrics, num_items, expected_users):
    require(len(predictions) == expected_users and not predictions.user_id.duplicated().any(), "Incomplete/duplicate validation users")
    ranks = predictions.target_rank.to_numpy()
    require(np.isfinite(ranks).all() and np.equal(ranks, ranks.astype(int)).all()
            and ((ranks >= 1) & (ranks <= num_items)).all(), "Invalid full-catalogue ranks")
    warm = predictions.warm_target.to_numpy(dtype=bool)
    expected = {}
    for k in (5, 10, 20):
        expected[f"recall@{k}"] = (ranks <= k).astype(float)
        expected[f"ndcg@{k}"] = np.where(ranks <= k, 1 / np.log2(ranks + 1), 0.)
    expected["mrr@10"] = np.where(ranks <= 10, 1 / ranks, 0.)
    for key, values in expected.items():
        require(np.allclose(predictions[key], values, atol=1e-12, rtol=1e-10), f"Incorrect per-user {key}")
        for group, mask in (("full", np.ones(len(ranks), dtype=bool)), ("warm", warm)):
            require(metrics[group]["users"] == int(mask.sum()), "Validation group user count mismatch")
            require(np.isclose(values[mask].mean(), metrics[group][key], atol=1e-12), f"Incorrect aggregate {group}/{key}")
    require(np.isclose(warm.mean(), metrics["warm_coverage"], atol=1e-12), "Incorrect warm coverage")
    # Includes complete history/target mapping but excludes predictions from the signature.
    signature = hashlib.sha256()
    for row in predictions.sort_values("user_id").itertuples(index=False):
        history, top = list(row.history), list(row.top_k)
        seen = set(history) - {row.target}
        require(1 <= row.target <= num_items and all(1 <= i <= num_items for i in history), "Invalid input item ID")
        require(len(top) == min(20, num_items-len(seen)) and len(set(top)) == len(top)
                and all(1 <= i <= num_items and i not in seen for i in top), "Invalid/history-masked top-K")
        require(row.target_rank <= num_items-len(seen), "Rank exceeds eligible catalogue")
        require((row.target_rank <= len(top) and top[row.target_rank-1] == row.target)
                or (row.target_rank > len(top) and row.target not in top), "Top-K and target rank disagree")
        signature.update(np.asarray([row.user_id, row.target, int(row.warm_target), len(history), *history], dtype="<i8").tobytes())
    return signature.hexdigest()


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def validate_run(run, plan, index, parameters, expected_users, num_items):
    config = yaml.safe_load((run / "config.yaml").read_text())
    status = read_json(run / "status.json")
    identity = fingerprint(config)
    require(status["state"] == "VALIDATED" and status["config_hash"] == identity == run.parent.name, "Run state/config hash mismatch")
    require(REQUIRED <= set(status["files"]), "Missing required hashed artifacts")
    for name, digest in status["files"].items():
        require(PurePosixPath(name).name == name and file_hash(run / name) == digest, f"Artifact SHA256 mismatch: {name}")
    require({p.name for p in run.iterdir() if p.is_file()} == REQUIRED | {"status.json"}, "Unexpected/missing run files")
    require(config["model"] == plan["model"] and config["dataset"] == "beauty"
            and config["split_version"] == plan["split_version"] == SPLIT and config["seed"] == 7
            and config["tuning_plan_hash"] == fingerprint(plan) and config["tuning_trial"] == index, "Trial provenance mismatch")
    require(config["evaluate_test"] is False and not config["pilot"] and config["max_train_batches"] is None
            and config["validation_user_limit"] is None and config["retention"] == 1.0, "Tuning is test-enabled, pilot or truncated")
    require(config["ranking"] == "full_catalogue_history_mask_target_exempt_ties_id", "Changed ranking protocol")
    require(all(config[k] == v for k, v in {**plan["runtime_overrides"], **parameters}.items()), "Planned/resolved hyperparameter mismatch")
    history = pd.read_csv(run / "train_history.csv")
    validation = pd.read_csv(run / "validation_metrics.csv")
    efficiency = read_json(run / "efficiency.json")
    metrics = read_json(run / "validation_selection.json")
    require(list(history.epoch) == list(range(len(history))) and len(history) == efficiency["epochs_completed"], "Non-contiguous/inconsistent epochs")
    for key in ("train_loss", "recommendation_loss", "coarse_loss", "fine_loss", "epoch_seconds", "validation_seconds", "peak_gpu_memory_bytes"):
        require(np.isfinite(history[key]).all(), f"Nonfinite history {key}")
    evaluated = history[history.validation_performed]
    require(list(evaluated.epoch) == list(validation.epoch)
            and np.allclose(evaluated["validation_ndcg@10"], validation["ndcg@10"], atol=1e-12), "Validation history mismatch")
    expected_schedule = [e for e in history.epoch if e == 0 or (e+1) % config["validation_every"] == 0 or e+1 == config["max_epochs"]]
    require(list(validation.epoch) == expected_schedule and (validation.users == expected_users).all(), "Incorrect validation schedule/coverage")
    best_row = validation.loc[validation["ndcg@10"].idxmax()]
    score = metrics["full"]["ndcg@10"]
    require(np.isclose(score, best_row["ndcg@10"], atol=1e-12)
            and np.isclose(score, efficiency["best_validation_ndcg@10"], atol=1e-12), "Best checkpoint metric mismatch")
    for hist_key, efficiency_key in (("epoch_seconds", "total_training_seconds"), ("validation_seconds", "total_validation_seconds")):
        require(np.isclose(history[hist_key].sum(), efficiency[efficiency_key], atol=1e-6), "Timing accounting mismatch")
    require(int(history.peak_gpu_memory_bytes.max()) == efficiency["peak_gpu_memory_bytes"], "Memory accounting mismatch")
    require(efficiency["evaluate_test"] is False and not efficiency["pilot"], "Invalid efficiency provenance")
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    latest = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    require(checkpoint["config"] == config and checkpoint["num_items"] == num_items
            and latest["config_hash"] == identity and latest["epoch"]+1 == len(history), "Checkpoint metadata mismatch")
    require(finite_tree(checkpoint["model"]) and finite_tree(latest["model"]) and finite_tree(latest["optimizer"]), "Nonfinite model/optimizer")
    require({"optimizer", "scaler", "rng", "best_checkpoint", "history", "validations"} <= set(latest), "Missing resume state")
    require(latest["best_checkpoint"]["config"] == config
            and all(torch.equal(v, latest["best_checkpoint"]["model"][k]) for k, v in checkpoint["model"].items()), "Latest/best weights disagree")
    require(np.isclose(latest["best"], score, atol=1e-12)
            and len(latest["history"]) == len(history) and len(latest["validations"]) == len(validation), "Resume history/best mismatch")
    from src.models.factory import build_model
    model = build_model(config["model"], num_items, config)  # CPU load/shape check only; no forward or training.
    model.load_state_dict(checkpoint["model"], strict=True)
    minimum = max(config["min_epochs"], config.get("warm_up_epochs", 0)+config["min_post_warmup_epochs"] if config["model"] == "fenrec" else 1)
    require(len(history) >= minimum and len(history) <= config["max_epochs"], "Invalid minimum/maximum training budget")
    limited = len(history) == config["max_epochs"]
    patience = latest["bad_epochs"] >= config["early_stopping_patience"]
    running_best, expected_bad = float("-inf"), 0
    for epoch in history.epoch:
        current = validation[validation.epoch == epoch]
        if len(current):
            value = float(current["ndcg@10"].iloc[0])
            if value > running_best:
                running_best, expected_bad = value, 0
            else:
                expected_bad += 1
        if epoch+1 < minimum:
            expected_bad = 0
    require(expected_bad == latest["bad_epochs"], "Patience counter differs from recorded validation trajectory")
    require(efficiency["training_limit_reached"] == limited and efficiency["stopped_for_patience"] == patience
            and (limited or patience), "Stopping reason inconsistent")
    signature = validate_predictions(pd.read_parquet(run / "validation_predictions.parquet"), metrics, num_items, expected_users)
    environment = read_json(run / "environment.json")
    row = {"model": config["model"], "trial": index, "seed": 7, "config_hash": identity,
           "tuning_plan_hash": config["tuning_plan_hash"], "split_version": SPLIT, **parameters,
           "validation_ndcg@10": score, "best_epoch": int(best_row.epoch)+1, "epochs_completed": len(history),
           "max_epochs": config["max_epochs"], "validation_checks": len(validation), "stopped_for_patience": patience,
           "training_limit_reached": limited, "stopping_reason": "max_epochs" if limited else "patience",
           "initial_train_loss": float(history.train_loss.iloc[0]), "final_train_loss": float(history.train_loss.iloc[-1]),
           "final_validation_ndcg@10": float(validation["ndcg@10"].iloc[-1]),
           "training_seconds": efficiency["total_training_seconds"], "validation_seconds": efficiency["total_validation_seconds"],
           "clustering_seconds": float(history.cluster_seconds.sum()), "checkpoint_seconds": float(history.checkpoint_seconds.sum()),
           "peak_gpu_memory_mib": efficiency["peak_gpu_memory_bytes"]/1024**2,
           "gpu_name": environment["gpu_name"], "gpu_device": environment["device"],
           "validation_users": expected_users, "parameter_count": sum(p.numel() for p in model.parameters())}
    return row, history, validation, signature


def winner_index(records):
    require(bool(records), "No completed trials")
    return max(records, key=lambda row: (row["validation_ndcg@10"], -row["trial"]))["trial"]


def analyze_archive(archive, root):
    root = Path(root).resolve()
    raw = root / "results/raw/phase5c" / ARCHIVE_SHA256
    digest, inventory = extract_verified(archive, raw)
    paths = {item["member"] for item in inventory}
    require(not any(PurePosixPath(p).name in {"test_metrics.json", "predictions.parquet"} for p in paths), "Test result artifact in tuning archive")
    data_audit = read_json(raw / "results/reports/kaggle_data_audit.json")
    require(data_audit["state"] == "PASSED" and data_audit["split_version"] == SPLIT, "Dataset audit/split mismatch")
    users, num_items = data_audit["statistics"]["users"], data_audit["statistics"]["items"]
    require((users, num_items, data_audit["statistics"]["interactions"]) == (22363, 12101, 198502), "Unexpected Beauty population")
    plans = sorted((raw / "results/tuning/beauty").glob("*/*/tuning_plan.json"))
    require(len(plans) == 5 and {read_json(p)["model"] for p in plans} == set(MODELS), "Expected exactly five model plans")
    require(len([p for p in paths if p.endswith('/status.json')]) == 15, "Expected exactly fifteen runs")
    rows, histories, validations, selections, signatures, plan_hashes = [], [], [], {}, set(), {}
    for path in plans:
        plan, directory = read_json(path), path.parent
        model, plan_hash = plan["model"], fingerprint(plan)
        require(plan_hash == directory.name and plan["seed"] == 7 and plan["evaluate_test"] is False
                and plan["selection_metric"] == "validation_ndcg@10" and plan["split_version"] == SPLIT
                and len(plan["trials"]) == 3, "Invalid plan hash/selection protocol")
        records = read_json(directory / "trial_results.json")
        selection = yaml.safe_load((directory / "selected_config.yaml").read_text())
        require(sorted(r["trial"] for r in records) == [0, 1, 2], "Missing or duplicate trial records")
        require(len(list(directory.glob("runs/TUNING/beauty/*/*/7/status.json"))) == 3, "Extra/missing trial directories")
        by_index = {r["trial"]: r for r in records}
        for index, parameters in enumerate(plan["trials"]):
            record = by_index[index]
            member = PurePosixPath(record["run_path"])
            require(not member.is_absolute() and ".." not in member.parts and "\\" not in str(member)
                    and str(member).startswith(str(directory.relative_to(raw)).replace("\\", "/")+"/runs/"), "Run path outside plan")
            run = raw / str(member)
            require(record["hyperparameters"] == parameters, "Trial parameter record mismatch")
            row, history, validation, signature = validate_run(run, plan, index, parameters, users, num_items)
            require(np.isclose(row["validation_ndcg@10"], record["validation_ndcg@10"], atol=1e-12), "Trial score record mismatch")
            row["source_member"] = str(member)
            rows.append(row)
            histories.append(history.assign(model=model, trial=index, epoch_display=history.epoch+1))
            validations.append(validation.assign(model=model, trial=index, epoch_display=validation.epoch+1))
            signatures.add(signature)
            print(f"Verified {model} trial={index}: {row['epochs_completed']} epochs, validation NDCG@10={row['validation_ndcg@10']:.8f}", flush=True)
        winning = winner_index(records)
        require(selection["trial"] == winning and all(selection[k] == by_index[winning][k] for k in by_index[winning]), "Winner is not validation argmax/earliest tie")
        require(selection["selection_split"] == "validation" and selection["seed"] == 7
                and selection["trials_completed"] == 3 and selection["pilot_only"] is False
                and selection["model"] == model and selection["dataset"] == "beauty"
                and selection["split_version"] == SPLIT and selection["tuning_plan_hash"] == plan_hash, "Invalid selected configuration provenance")
        selections[model] = {"path": directory / "selected_config.yaml", "trial": winning}
        plan_hashes[model] = plan_hash
    require(len(signatures) == 1, "Trials use inconsistent validation user/history/target populations")
    frame = pd.DataFrame(rows).sort_values(["model", "trial"]).reset_index(drop=True)
    frame["selected"] = [int(r.trial) == selections[r.model]["trial"] for r in frame.itertuples()]
    history_frame, validation_frame = pd.concat(histories, ignore_index=True), pd.concat(validations, ignore_index=True)
    tables = root / "results/tables/phase5c"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / "tuning_trials.csv", index=False)
    frame[frame.selected].to_csv(tables / "selected_hyperparameters.csv", index=False)
    history_frame.to_csv(tables / "epoch_history.csv", index=False)
    validation_frame.to_csv(tables / "validation_curves.csv", index=False)
    pd.DataFrame(inventory).to_csv(tables / "archive_inventory.csv", index=False)
    selection_dir = root / "configs/selected/phase5c/beauty"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_manifest = {}
    for model, value in selections.items():
        target = selection_dir / f"{model}.yaml"
        content = value["path"].read_bytes()
        require(not target.exists() or target.read_bytes() == content, "Existing selected configuration differs")
        if not target.exists():
            target.write_bytes(content)
        selection_manifest[model] = {"selected_trial": value["trial"], "sha256": file_hash(target),
                                     "source_member": value["path"].relative_to(raw).as_posix(), "tuning_plan_hash": plan_hashes[model]}
    write_json(selection_dir / "selection_manifest.json", {"archive_sha256": digest, "selection_split": "validation", "models": selection_manifest})
    report = {"state": "VERIFIED", "phase": "5C", "source_archive": Path(archive).name, "archive_sha256": digest,
              "archive_files": len(inventory), "archive_uncompressed_bytes": sum(i["bytes"] for i in inventory),
              "models": list(MODELS), "trials": len(frame), "trials_per_model": 3, "seed": 7,
              "checkpoint_files_verified": 30, "hashed_run_artifacts_verified": len(frame)*len(REQUIRED),
              "split_version": SPLIT, "validation_users_per_trial": users, "catalogue_items": num_items,
              "validation_population_signature": next(iter(signatures)), "selection_split": "validation",
              "test_result_artifacts": 0, "experimental_training_runs_launched_locally": 0,
              "limit_reached_trials": int(frame.training_limit_reached.sum()), "patience_stopped_trials": int(frame.stopped_for_patience.sum()),
              "total_training_seconds": float(frame.training_seconds.sum()), "total_validation_seconds": float(frame.validation_seconds.sum()),
              "dataset_verification_scope": "Archived PASSED data-audit report and matching split IDs across all plans/configs/checkpoints; dataset bytes are not in this ZIP and were not independently rehashed in this analysis.",
              "execution_provenance_limit": "ZIP has no training source snapshot or Git commit attestation; hash/selection consistency verifies supplied artifacts, not unseen execution history.",
              "timing_scope": "Logged epoch training includes FENRec clustering; validation excludes final selected-checkpoint validation; preparation/session wall time is not recorded.",
              "convergence_scope": "Patience is a stopping criterion, not proof of global convergence. Epoch-capped trials are budget-limited.",
              "selection_manifest": selection_manifest}
    write_json(tables / "artifact_audit.json", report)
    derived = root / "results/derived/phase5c" / digest
    write_json(derived / "analysis_provenance.json", {"archive_sha256": digest, "analysis_source_sha256": file_hash(Path(__file__)),
                                                    "summary_files": {p.name: file_hash(p) for p in tables.iterdir() if p.is_file()},
                                                    "python_packages": {"numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__}})
    return frame, report
