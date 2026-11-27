"""Read-only audit/statistics for the first official Kaggle master session."""
import itertools
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
import yaml
from threadpoolctl import threadpool_limits

from src.analysis.phase5c import extract_verified, require, read_json, finite_tree, validate_predictions, SPLIT, LABELS
from src.utils.artifacts import file_hash, fingerprint, verify_bundle, write_json
from src.data.splitting import validate_splits, evaluation_cases

ARCHIVE_NAME = "seqrec-master-results-20261126_234725_576013.zip"
ARCHIVE_SHA256 = "9f6b30dd3d29750b21ef64ddd7c8fea88c66d1c3ca52ea44a9809b3cc323cfab"
SEEDS = (7, 42, 2026)
MODELS = ("mostpopular", "gru4rec_ce", "sasrec_bce", "sasrec_ce", "bert4rec", "fenrec")
LABELS = {**LABELS, "mostpopular": "MostPopular", "attention_ablation": "Attention ablation"}
METRICS = ("recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20", "mrr@10")
BUNDLE_HASH = "298d55546282277e0e2ba77a752e8be10e9067b1a05fee87ab09b6a5bffed6da"


def paired_metric_bootstrap(left, right, resamples=10000, seed=42):
    """Exact empirical paired-user bootstrap via multinomial multiplicities.

    Compress identical per-user difference vectors, not individual users' seeds.
    This has exactly the same distribution as sampling N user rows with replacement.
    All metric columns use the same replicate multiplicities. No normal approximation.
    """
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    require(left.ndim == 2 and left.shape == right.shape and len(left) > 0 and resamples > 0, "Paired nonempty matrices required")
    require(np.isfinite(left).all() and np.isfinite(right).all(), "Nonfinite bootstrap contribution")
    delta = left-right
    values, counts = np.unique(delta, axis=0, return_counts=True)
    probabilities = counts/counts.sum()
    rng = np.random.default_rng(seed)
    means = np.empty((resamples, delta.shape[1]))
    with threadpool_limits(limits=1):
        for start in range(0, resamples, 128):
            multiplicity = rng.multinomial(len(delta), probabilities, size=min(128, resamples-start))
            means[start:start+len(multiplicity)] = multiplicity @ values / len(delta)
    return delta.mean(axis=0), np.quantile(means, .025, axis=0), np.quantile(means, .975, axis=0)


def seed_average(frames, model, seeds=SEEDS):
    """Pair sorted user rows, average repeated seeds within each user first."""
    use = (7,) if model == "mostpopular" else seeds
    selected = [frames[(model, s)].sort_values("user_id") for s in use]
    users = selected[0].user_id.to_numpy()
    for frame in selected[1:]:
        require(np.array_equal(users, frame.user_id.to_numpy()), "Bootstrap user alignment mismatch")
    return np.mean([frame[list(METRICS)].to_numpy() for frame in selected], axis=0)


def course_jobs(jobs, successful_paths):
    """Resource-defined scope, independent of all test scores; do not invent SUCCESS."""
    output = []
    for job in jobs:
        exp = job["experiment"]
        if exp not in ("E1", "E2") and not (exp == "E3" and job["model"] in ("sasrec_ce", "bert4rec", "fenrec")
                                              and job["seed"] == 7 and job["retention"] in (.4, 1.)):
            continue
        record = dict(job)
        record["archive_job_id"] = record.pop("job")
        record["evidence_state"] = "ARCHIVED_SUCCESS" if job["path"] in successful_paths else "PENDING"
        if exp == "E3" and job["retention"] == 1.:
            source = job["path"].replace("/E3/", "/E1/")
            if source in successful_paths:
                record.update(evidence_state="REUSE_AVAILABLE", reuse_source=source)
        output.append(record)
    return output


def audit_run(run, job, parts, expected_cases, expected_users=22363, num_items=12101):
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    status, efficiency = read_json(run / "status.json"), read_json(run / "efficiency.json")
    require(status["state"] == "SUCCESS" and status["config_hash"] == fingerprint(config) == job["config_hash"]
            and config == job["config"] and run.parent.name == fingerprint(config), "Run/config/job fingerprint mismatch")
    required = {"config.yaml", "checkpoint.pt", "environment.json", "efficiency.json", "predictions.parquet", "test_metrics.json", "train_history.csv", "validation_metrics.csv", "run.log"}
    if config["model"] != "mostpopular":
        required |= {"latest.pt", "evaluation_users.csv"}
    require(required <= set(status["files"]), "Missing status-hashed artifacts")
    for name, checksum in status["files"].items():
        require(Path(name).name == name and file_hash(run / name) == checksum, f"Artifact hash mismatch: {name}")
    require({p.name for p in run.iterdir() if p.is_file()} == set(status["files"]) | {"status.json"}, "Unexpected run files")
    require(config["split_version"] == SPLIT and config["dataset"] == "beauty" and not config.get("pilot", False), "Pilot/dataset/split mismatch")
    require(config["ranking"] == "full_catalogue_history_mask_target_exempt_ties_id", "Ranking protocol mismatch")
    frame = pd.read_parquet(run / "predictions.parquet").sort_values("user_id").reset_index(drop=True)
    metrics = read_json(run / "test_metrics.json")
    signature = validate_predictions(frame, metrics, num_items, expected_users)
    train_items = set(parts["train"].item_id)
    require(np.array_equal(frame.warm_target.to_numpy(), frame.target.isin(train_items).to_numpy()), "Warm-target membership mismatch")
    for prediction, case in zip(frame.itertuples(index=False), expected_cases):
        require(prediction.user_id == case["user_id"] and prediction.target == case["target"]
                and list(prediction.history) == case["history"], "Prediction history/target differs from hashed held-out dataset")
    history = pd.read_csv(run / "train_history.csv")
    validation = pd.read_csv(run / "validation_metrics.csv")
    environment = read_json(run / "environment.json")
    row = {k: job[k] for k in ("experiment", "model", "seed", "variant", "retention", "max_seq_length", "config_hash", "path")}
    row.update(split_version=SPLIT, validation_users=expected_users, test_users=expected_users,
               best_validation_ndcg10=float(validation["ndcg@10"].max()), checkpoint_bytes=(run / "checkpoint.pt").stat().st_size,
               hashed_artifacts=len(status["files"]), measurement_source="archived_Kaggle_efficiency_json",
               measurement_device=environment.get("device", "cpu"), gpu_name=environment.get("gpu_name"),
               inference_mean_ms=efficiency["inference_latency"]["batch1_mean_seconds"]*1000,
               inference_p95_ms=efficiency["inference_latency"]["batch1_p95_seconds"]*1000,
               inference_scope=efficiency["inference_latency"]["scope"], inference_repeats=efficiency["inference_latency"]["repeats"],
               inference_warmup=efficiency["inference_latency"]["warmup"])
    require(np.isfinite([row["inference_mean_ms"], row["inference_p95_ms"]]).all(), "Nonfinite inference measurement")
    if config["model"] == "mostpopular":
        from src.models.popularity import MostPopular
        model = MostPopular.load(run / "checkpoint.pt")
        counts = np.bincount(parts["train"].item_id.to_numpy(), minlength=num_items+1)
        require(model.num_items == num_items and np.array_equal(model.counts, counts), "Popularity weights differ from training split")
        row.update(epochs_completed=1, best_epoch=1, stopped_for_patience=False, training_limit_reached=False,
                   training_seconds=efficiency["fit_seconds"], validation_seconds=np.nan, clustering_seconds=0.,
                   checkpoint_seconds=np.nan, peak_gpu_memory_mib=np.nan, total_parameters=0, test_evaluation_seconds=np.nan,
                   test_users_per_second=np.nan, timing_scope="baseline_fit_only", initial_train_loss=np.nan, final_train_loss=np.nan)
        return row, frame, history, validation, signature
    require(config["evaluate_test"] is True and config["max_train_batches"] is None
            and config["validation_user_limit"] is None and config["retention"] == 1., "Truncated/non-final evaluation")
    require(list(history.epoch) == list(range(len(history))) and len(history) == efficiency["epochs_completed"], "Epoch sequence mismatch")
    for key in ("train_loss", "recommendation_loss", "coarse_loss", "fine_loss", "epoch_seconds", "validation_seconds", "peak_gpu_memory_bytes"):
        require(np.isfinite(history[key]).all(), f"Nonfinite {key}")
    checks = history[history.validation_performed]
    require(list(checks.epoch) == list(validation.epoch) and np.allclose(checks["validation_ndcg@10"], validation["ndcg@10"], atol=1e-12), "Validation log mismatch")
    require(list(validation.epoch) == [e for e in history.epoch if e == 0 or (e+1) % config["validation_every"] == 0 or e+1 == config["max_epochs"]], "Validation schedule mismatch")
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    latest = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    require(checkpoint["config"] == config and checkpoint["num_items"] == num_items and latest["config_hash"] == fingerprint(config)
            and latest["epoch"]+1 == len(history), "Checkpoint/resume metadata mismatch")
    require(finite_tree(checkpoint["model"]) and finite_tree(latest["model"]) and finite_tree(latest["optimizer"]), "Nonfinite checkpoint state")
    require({"rng", "scaler", "optimizer", "history", "validations", "best_checkpoint"} <= set(latest), "Incomplete resume state")
    require(latest["best_checkpoint"]["config"] == config and all(torch.equal(value, latest["best_checkpoint"]["model"][key]) for key, value in checkpoint["model"].items()), "Best/latest best weights differ")
    best = validation.loc[validation["ndcg@10"].idxmax()]
    require(np.isclose(latest["best"], best["ndcg@10"], atol=1e-12) and np.isclose(efficiency["best_validation_ndcg@10"], best["ndcg@10"], atol=1e-12), "Best selection mismatch")
    minimum = max(config["min_epochs"], config.get("warm_up_epochs", 0)+config["min_post_warmup_epochs"] if config["model"] == "fenrec" else 1)
    best_so_far, bad = -np.inf, 0
    for epoch in history.epoch:
        current = validation[validation.epoch == epoch]
        if len(current):
            score = float(current["ndcg@10"].iloc[0])
            if score > best_so_far:
                best_so_far, bad = score, 0
            else:
                bad += 1
        if epoch+1 < minimum:
            bad = 0
    limited, patience = len(history) >= config["max_epochs"], bad >= config["early_stopping_patience"]
    require(minimum <= len(history) <= config["max_epochs"] and bad == latest["bad_epochs"]
            and limited == efficiency["training_limit_reached"] and patience == efficiency["converged"] and (limited or patience), "Stopping-state mismatch")
    for column, key in (("epoch_seconds", "total_training_seconds"), ("validation_seconds", "total_validation_seconds")):
        require(np.isclose(history[column].sum(), efficiency[key], atol=1e-6), "Timing accounting mismatch")
    require(int(history.peak_gpu_memory_bytes.max()) == efficiency["peak_gpu_memory_bytes"], "GPU peak mismatch")
    from src.models.factory import build_model
    model = build_model(config["model"], num_items, config)
    model.load_state_dict(checkpoint["model"], strict=True)  # CPU shape/load check only.
    require(sum(p.numel() for p in model.parameters()) == efficiency["total_parameters"], "Parameter count mismatch")
    evaluated = pd.read_csv(run / "evaluation_users.csv").sort_values("user_id")
    require(np.array_equal(evaluated.user_id, frame.user_id) and np.array_equal(evaluated.target, frame.target), "Evaluation user manifest mismatch")
    row.update(epochs_completed=len(history), best_epoch=int(best.epoch)+1, stopped_for_patience=patience,
               training_limit_reached=limited, training_seconds=efficiency["total_training_seconds"], validation_seconds=efficiency["total_validation_seconds"],
               clustering_seconds=float(history.cluster_seconds.sum()), checkpoint_seconds=float(history.checkpoint_seconds.sum()),
               peak_gpu_memory_mib=efficiency["peak_gpu_memory_bytes"]/1048576, total_parameters=efficiency["total_parameters"],
               test_evaluation_seconds=efficiency["test_evaluation_seconds"], test_users_per_second=efficiency["test_users_per_second"],
               timing_scope="epoch_training_including_clustering", initial_train_loss=float(history.train_loss.iloc[0]), final_train_loss=float(history.train_loss.iloc[-1]))
    return row, frame, history, validation, signature


def error_analysis(frames, parts, metadata, private):
    """Aggregate every E1 user's saved predictions; never run model inference."""
    first = frames[("sasrec_ce", 7)].sort_values("user_id").reset_index(drop=True)
    lengths = first.history.map(len).to_numpy()
    frequency = parts["train"].item_id.value_counts()
    popularity = first.target.map(frequency).fillna(0).to_numpy(dtype=int)
    length_bin = pd.cut(lengths, [0, 5, 10, 20, 50, np.inf], labels=["1-5", "6-10", "11-20", "21-50", "51+"])
    popularity_bin = pd.cut(popularity, [-1, 0, 5, 10, 20, np.inf], labels=["cold_0", "1-5", "6-10", "11-20", "21+"])
    groups, segments, pairs, examples = [], [], [], []
    item_meta = {int(row["item_id"]): row for row in metadata}
    for seed in SEEDS:
        aligned = {model: frames[(model, 7 if model == "mostpopular" else seed)].sort_values("user_id").reset_index(drop=True) for model in MODELS}
        hit = {model: frame.target_rank.to_numpy() <= 10 for model, frame in aligned.items()}
        categories = {"sasrec_only_vs_bert": hit["sasrec_ce"] & ~hit["bert4rec"],
                      "bert_only_vs_sasrec": hit["bert4rec"] & ~hit["sasrec_ce"],
                      "fenrec_only_vs_sasrec": hit["fenrec"] & ~hit["sasrec_ce"],
                      "all_six_incorrect": ~np.logical_or.reduce(list(hit.values()))}
        for name, mask in categories.items():
            groups.append({"seed": seed, "category": name, "users": int(mask.sum()), "fraction": float(mask.mean()),
                           "mean_history_length": float(lengths[mask].mean()) if mask.any() else np.nan,
                           "median_target_train_frequency": float(np.median(popularity[mask])) if mask.any() else np.nan})
            # Fixed smallest-user-ID examples, not greatest test-score differences.
            for index in np.flatnonzero(mask)[:3]:
                record = first.iloc[index]
                target_meta = item_meta.get(int(record.target), {})
                examples.append({"seed": seed, "category": name, "user_id": int(record.user_id), "target": int(record.target),
                                 "history": list(map(int, record.history)), "target_title": target_meta.get("title"),
                                 "target_categories": target_meta.get("categories"), "target_train_frequency": int(popularity[index]),
                                 "models": {m: {"target_rank": int(f.iloc[index].target_rank), "top_k": list(map(int, f.iloc[index].top_k))} for m, f in aligned.items()}})
        for left, right in (("sasrec_ce", "bert4rec"), ("fenrec", "sasrec_ce")):
            for left_hit, right_hit in itertools.product((False, True), repeat=2):
                mask = (hit[left] == left_hit) & (hit[right] == right_hit)
                pairs.append({"seed": seed, "left": left, "right": right, "left_hit": left_hit, "right_hit": right_hit, "users": int(mask.sum()), "fraction": float(mask.mean())})
        for model, frame in aligned.items():
            if model == "mostpopular" and seed != 7:
                continue
            for dimension, bins in (("history_length", length_bin), ("target_training_popularity", popularity_bin)):
                for label in bins.categories:
                    mask = np.asarray(bins == label)
                    for metric in METRICS:
                        segments.append({"model": model, "seed": seed, "dimension": dimension, "bin": str(label),
                                         "users": int(mask.sum()), "metric": metric, "value": float(frame.loc[mask, metric].mean()) if mask.any() else np.nan})
    write_json(private / "e8_private_examples.json", {"selection_rule": "first three sorted user IDs per predefined category/seed", "cases": examples})
    return pd.DataFrame(groups), pd.DataFrame(segments), pd.DataFrame(pairs)


def analyze(archive, root, bundle, resamples=10000):
    root, archive, bundle = Path(root).resolve(), Path(archive).resolve(), Path(bundle).resolve()
    require(resamples >= 10000, "Research report requires at least 10,000 bootstrap resamples")
    torch.set_num_threads(1)
    destination = root / "artifacts/master_session1" / ARCHIVE_NAME
    require(file_hash(archive) == ARCHIVE_SHA256, "Wrong official source ZIP")
    if destination.exists():
        require(file_hash(destination) == ARCHIVE_SHA256, "Local archive differs; refusing overwrite")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
    raw = root / "results/raw/master_session1" / ARCHIVE_SHA256
    _, inventory = extract_verified(destination, raw, ARCHIVE_SHA256)
    master = raw / "results/master"
    manifest = read_json(master / "dataset_manifest.json")
    require(fingerprint({k: v for k, v in manifest.items() if k != "split_version"}) == manifest["split_version"] == SPLIT, "Archived dataset manifest fingerprint mismatch")
    require(verify_bundle(bundle) == manifest, "Local bundle bytes/manifest differ from archived split")
    parts = {key: pd.read_parquet(bundle / f"{key}.parquet") for key in ("train", "validation", "test")}
    validate_splits(parts)
    cases = evaluation_cases(parts, "test")
    audit = read_json(raw / "results/reports/kaggle_data_audit.json")
    require(audit["state"] == "PASSED" and audit["split_version"] == SPLIT, "Archived dataset audit mismatch")
    jobs, plan = read_json(master / "jobs.json"), read_json(master / "master_plan.json")
    for model, selection in plan["selections"].items():
        require(selection["selection_split"] == "validation" and selection["split_version"] == SPLIT and not selection["pilot_only"], "Non-validation preselection")
        snapshot = yaml.safe_load((master / "selections" / f"{model}.yaml").read_text(encoding="utf-8"))
        require(snapshot == selection, "Selection snapshot differs from master plan")
    progress = pd.read_csv(master / "progress.csv")
    require(len(jobs) == len(progress) == 127 and len(set(j["path"] for j in jobs)) == 127, "Unexpected/duplicate original jobs")
    require(read_json(master / "completion.json")["training_state"] == "BUDGET_STOP"
            and read_json(master / "completion.json")["gpu_e7_state"] == "DEFERRED", "Session state mismatch")
    for job in jobs:
        require(fingerprint(job["config"]) == job["config_hash"] and job["config"]["split_version"] == SPLIT, "Planned job hash/split mismatch")
        record = progress.loc[progress.job == job["job"]]
        require(len(record) == 1 and record.iloc[0].path == job["path"] and record.iloc[0].config_hash == job["config_hash"], "Progress/job manifest mismatch")
    successful = [j for j in jobs if (raw / j["path"] / "status.json").exists()]
    require(len(successful) == 22 and sum(j["experiment"] == "E1" for j in successful) == 16
            and sum(j["experiment"] == "E2" for j in successful) == 6, "Unexpected completed job coverage")
    require(set(progress.loc[progress.state == "SUCCESS", "path"]) == {j["path"] for j in successful}, "SUCCESS coverage mismatch")
    require(set(progress.loc[~progress.path.isin([j["path"] for j in successful]), "state"]) == {"PENDING"}, "Unexpected non-complete state")
    rows, long_metrics, histories, validations, signatures, frames, e2_frames = [], [], [], [], set(), {}, {}
    for job in successful:
        if job["model"] != "mostpopular":
            expected = plan["e2_settings"] if job["experiment"] == "E2" else plan["selections"][job["model"]]["hyperparameters"]
            require(all(job["config"][k] == v for k, v in expected.items()), "Archived preselected hyperparameters not applied")
        row, frame, history, validation, signature = audit_run(raw / job["path"], job, parts, cases)
        rows.append(row)
        signatures.add(signature)
        metric_record = read_json(raw / job["path"] / "test_metrics.json")
        for group in ("full", "warm"):
            for metric in METRICS:
                long_metrics.append({k: row[k] for k in ("experiment", "model", "variant", "seed")} | {"group": group, "metric": metric, "value": metric_record[group][metric], "users": metric_record[group]["users"]})
        histories.append(history.assign(experiment=job["experiment"], model=job["model"], seed=job["seed"], variant=job["variant"]))
        validations.append(validation.assign(experiment=job["experiment"], model=job["model"], seed=job["seed"], variant=job["variant"]))
        if job["experiment"] == "E1":
            frames[(job["model"], job["seed"])] = frame
        else:
            e2_frames[(job["variant"], job["seed"])] = frame
        print(f"VERIFIED {job['experiment']} {job['model']} {job['variant']} seed={job['seed']}", flush=True)
    require(len(signatures) == 1, "Runs have inconsistent test populations")
    for model in MODELS:
        require({s for m, s in frames if m == model} == ({7} if model == "mostpopular" else set(SEEDS)), "E1 missing/extra seeds")
    run_frame, metric_frame = pd.DataFrame(rows), pd.DataFrame(long_metrics)
    require(run_frame.loc[run_frame.experiment == "E2", "total_parameters"].nunique() == 1, "E2 capacities do not match")
    summary = metric_frame.groupby(["experiment", "model", "variant", "group", "metric"], as_index=False).agg(n_runs=("value", "size"), mean=("value", "mean"), sd=("value", "std"))
    summary["design_complete"] = (summary.experiment == "E1") | (summary.n_runs == 3)
    bootstrap_rows = []
    averaged = {m: seed_average(frames, m) for m in MODELS}
    for left, right in itertools.combinations(MODELS, 2):
        point, low, high = paired_metric_bootstrap(averaged[left], averaged[right], resamples)
        for i, metric in enumerate(METRICS):
            bootstrap_rows.append({"experiment": "E1", "left": left, "right": right, "metric": metric, "difference": point[i], "ci_low": low[i], "ci_high": high[i], "users": len(cases), "resamples": resamples, "bootstrap_seed": 42, "seed_policy": "user contributions averaged across 7,42,2026; deterministic baseline once", "ci_adjustment": "none; pointwise exploratory intervals"})
        print(f"BOOTSTRAP E1 {left} - {right}", flush=True)
    for left, right in (("A2", "A1"), ("A4", "A3")):
        available = sorted({s for v, s in e2_frames if v == left} & {s for v, s in e2_frames if v == right})
        lvalues = np.mean([e2_frames[(left, s)][list(METRICS)].to_numpy() for s in available], axis=0)
        rvalues = np.mean([e2_frames[(right, s)][list(METRICS)].to_numpy() for s in available], axis=0)
        point, low, high = paired_metric_bootstrap(lvalues, rvalues, resamples)
        for i, metric in enumerate(METRICS):
            bootstrap_rows.append({"experiment": "E2", "left": left, "right": right, "metric": metric, "difference": point[i], "ci_low": low[i], "ci_high": high[i], "users": len(cases), "resamples": resamples, "bootstrap_seed": 42, "seed_policy": "matched completed seeds " + ",".join(map(str, available)), "ci_adjustment": "none; incomplete design"})
    private = root / "results/derived/master_session1" / ARCHIVE_SHA256
    e8_groups, e8_segments, e8_pairs = error_analysis(frames, parts, read_json(bundle / "metadata.json"), private)
    tables = root / "results/tables/master_session1"
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {"runs": run_frame, "per_seed_metrics": metric_frame, "metric_summary": summary, "paired_bootstrap": pd.DataFrame(bootstrap_rows),
               "epoch_history": pd.concat(histories, ignore_index=True), "validation_curves": pd.concat(validations, ignore_index=True),
               "e8_error_groups": e8_groups, "e8_segments": e8_segments, "e8_pair_outcomes": e8_pairs, "original_progress": progress,
               "archive_inventory": pd.DataFrame(inventory)}
    successes = {j["path"] for j in successful}
    revised = course_jobs(jobs, successes)
    require(len(revised) == 34 and sum(j["evidence_state"] == "REUSE_AVAILABLE" for j in revised) == 3, "Reduced course design mismatch")
    outputs["course_manifest"] = pd.DataFrame([{k: v for k, v in j.items() if k != "config"} for j in revised])
    for name, data in outputs.items():
        data.to_csv(tables / f"{name}.csv", index=False)
    write_json(root / "configs/experiments/course_jobs.json", {"source_archive_sha256": ARCHIVE_SHA256, "reason": "GPU time budget and course-project scope; not test-score-driven", "jobs": revised})
    report = {"state": "VERIFIED", "source_archive": ARCHIVE_NAME, "archive_sha256": ARCHIVE_SHA256, "archive_members": len(inventory),
              "archive_uncompressed_bytes": sum(i["bytes"] for i in inventory), "successful_runs": 22, "E1_success": 16, "E2_success": 6,
              "original_total": 127, "original_session_state": "BUDGET_STOP", "E7_gpu_state": "DEFERRED", "E8_local_state": "COMPLETED",
              "checkpoints_verified": 43, "hashed_artifacts": int(run_frame.hashed_artifacts.sum()), "test_users_per_run": len(cases),
              "split_version": SPLIT, "dataset_bytes_rehashed_against_archived_manifest": True, "chronological_split_checks": "PASSED",
              "validation_selected_hyperparameters_match_plan": True, "bootstrap_resamples": resamples,
              "bootstrap_method": "exact empirical paired user bootstrap, multinomial over identical seed-averaged contribution vectors; percentile 95%, seed 42",
              "bootstrap_limit": "conditional on trained seeds; not full retraining uncertainty; no multiplicity adjustment or confirmatory p-values",
              "experiment_training_or_tuning_launched_locally": False, "new_inference_or_gpu_measurements_locally": False,
              "pending_original": {"E2": 6, "E3": 27, "E4": 36, "E5": 36},
              "course_scope": {"requested_artifact_jobs": 34, "archived_success": 22, "E3_reuse_available": 3, "new_training_jobs_remaining": 9,
                               "E2_pending": 6, "E3_new_retention_0_4": 3, "E4": "DEFERRED", "E5": "DEFERRED", "TIGER": "THEORY_ONLY", "CRAMER": "THEORY_ONLY"},
              "dataset_manifest": manifest, "master_plan_sha256": file_hash(master / "master_plan.json"),
              "archived_driver_sha256": file_hash(master / "master_driver.py"), "training_source_limit": "Archive has the driver, not a full training-source snapshot or commit attestation"}
    write_json(tables / "artifact_audit.json", report)
    write_json(private / "analysis_provenance.json", {"archive_sha256": ARCHIVE_SHA256, "local_dataset_manifest": str(bundle / "manifest.json"),
                                                    "analysis_source_sha256": file_hash(Path(__file__)), "source_files": {p.name: file_hash(p) for p in tables.iterdir() if p.is_file()}})
    return outputs, report
