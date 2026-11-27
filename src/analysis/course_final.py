"""Final closed-course audit; immutable delta import and prediction-only statistics."""
import hashlib
import itertools
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile
import numpy as np
import pandas as pd
import torch
import yaml

from src.analysis.phase5c import require, read_json, SPLIT, extract_verified
from src.analysis import master_session1 as prior
from src.analysis import phase5c as tuning
from src.utils.artifacts import file_hash, fingerprint, verify_bundle, write_json
from src.data.splitting import validate_splits, evaluation_cases

ARCHIVE_NAME = "seqrec-e2-final-20261127_122610_032968.zip"
ARCHIVE_SHA256 = "4bc3da79817ad1463f1c4ff16b2e9aa13efbad8e9caad20f9368cfb690acdc39"
CONTRASTS = (("A2", "A1"), ("A3", "A2"), ("A4", "A3"))
VARIANT_DESIGN = {"A1": ("causal", "next_bce"), "A2": ("causal", "next_ce"),
                  "A3": ("causal", "masked_ce"), "A4": ("bidirectional", "masked_ce")}


def zipped_digest(stream, name):
    digest = hashlib.sha256()
    with stream.open(name) as source:
        for block in iter(lambda: source.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_prior_runs(new_zip, old_zip):
    """Compare every byte's SHA256 for all old completed-run files, including logs."""
    names = [n for n in old_zip.namelist() if n.startswith("results/core/") and not n.endswith("/")]
    checks = []
    for name in names:
        require(name in new_zip.namelist(), f"Prior completed artifact disappeared: {name}")
        old, new = zipped_digest(old_zip, name), zipped_digest(new_zip, name)
        require(old == new, f"Prior completed artifact changed: {name}")
        checks.append({"member": name, "old_sha256": old, "new_sha256": new, "unchanged": True})
    return checks


def import_delta(archive, destination, old_archive, old_raw, old_inventory, expected_sha=ARCHIVE_SHA256, old_sha=prior.ARCHIVE_SHA256):
    """Preflight ZIP/conflicts; hard-link identical immutable bytes on the same volume."""
    archive, destination, old_raw = Path(archive), Path(destination).resolve(), Path(old_raw).resolve()
    require(file_hash(archive) == expected_sha and file_hash(old_archive) == old_sha, "Source ZIP identity differs")
    existing_by_digest = {item["sha256"]: old_raw / item["member"] for item in old_inventory}
    inventory, names, planned = [], set(), []
    with zipfile.ZipFile(archive) as stream, zipfile.ZipFile(old_archive) as old_zip:
        require(stream.testzip() is None and old_zip.testzip() is None, "ZIP CRC failure")
        require(sum(i.file_size for i in stream.infolist()) <= 1024**3, "Archive exceeds extraction size limit")
        unchanged = compare_prior_runs(stream, old_zip)
        for info in stream.infolist():
            name = PurePosixPath(info.filename)
            require(not name.is_absolute() and ".." not in name.parts and "\\" not in info.orig_filename
                    and ":" not in info.filename and info.filename not in names and not stat.S_ISLNK(info.external_attr >> 16), "Unsafe/duplicate ZIP path")
            names.add(info.filename)
            target = (destination / info.filename).resolve()
            require(target.is_relative_to(destination), "ZIP target escaped destination")
            if info.is_dir():
                continue
            digest = zipped_digest(stream, info.filename)
            source = existing_by_digest.get(digest)
            if source is not None:
                require(source.is_file() and file_hash(source) == digest, "Prior raw artifact was altered")
            if target.exists():
                require(target.is_file() and file_hash(target) == digest, "Existing imported artifact differs")
            inventory.append({"member": info.filename, "bytes": info.file_size, "sha256": digest,
                              "deduplicated_prior_content": source is not None})
            planned.append((info.filename, target, source))
        # Every comparison/conflict passed before any write. Existing bytes are never modified.
        for name, target, source in planned:
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if source is not None:
                try:
                    os.link(source, target)
                    continue
                except OSError:
                    pass  # Cross-volume/platform fallback; copy only where linking is impossible.
            with stream.open(name) as origin, target.open("xb") as output:
                shutil.copyfileobj(origin, output)
    return inventory, unchanged


def closed_jobs(jobs, completed_paths):
    """Final scope has completed E1/E2 only; score-free and no remaining training."""
    selected = []
    for job in jobs:
        if job["experiment"] not in ("E1", "E2"):
            continue
        require(job["path"] in completed_paths, "Cannot close an incomplete E1/E2 design")
        record = dict(job)
        record["archive_job_id"] = record.pop("job")
        record["evidence_state"] = "ARCHIVED_SUCCESS"
        selected.append(record)
    return selected


def analyze(archive, root, bundle, resamples=10000):
    root, archive, bundle = Path(root).resolve(), Path(archive).resolve(), Path(bundle).resolve()
    require(resamples >= 10000, "At least 10,000 research bootstrap resamples required")
    torch.set_num_threads(1)
    target_zip = root / "artifacts/e2_final" / ARCHIVE_NAME
    require(file_hash(archive) == ARCHIVE_SHA256, "Wrong final archive")
    if target_zip.exists():
        require(file_hash(target_zip) == ARCHIVE_SHA256, "Existing final ZIP differs")
    else:
        target_zip.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("rb") as source, target_zip.open("xb") as target:
            shutil.copyfileobj(source, target)
    old_archive = root / "artifacts/master_session1" / prior.ARCHIVE_NAME
    old_raw = root / "results/raw/master_session1" / prior.ARCHIVE_SHA256
    old_inventory = pd.read_csv(root / "results/tables/master_session1/archive_inventory.csv").to_dict("records")
    if not (old_raw / "results/master/jobs.json").exists():
        extract_verified(old_archive, old_raw, prior.ARCHIVE_SHA256)
    raw = root / "results/raw/e2_final" / ARCHIVE_SHA256
    inventory, unchanged = import_delta(target_zip, raw, old_archive, old_raw, old_inventory)
    require(len(unchanged) == 262, "Expected all 262 files of the 22 prior runs")
    completion = read_json(raw / "results/e2_resume/completion.json")
    require(completion["state"] == "SUCCESS" and completion["e2_complete"] is True and completion["e2_completed"] == completion["e2_total"] == 12
            and completion["completed"] == 28 and completion["total"] == 127
            and completion["recovery_archive_sha256"] == prior.ARCHIVE_SHA256 and completion["split_version"] == SPLIT, "Final completion record mismatch")
    manifest = read_json(raw / "results/master/dataset_manifest.json")
    require(verify_bundle(bundle) == manifest and manifest["split_version"] == SPLIT, "Dataset bytes/split differ from archive")
    parts = {k: pd.read_parquet(bundle / f"{k}.parquet") for k in ("train", "validation", "test")}
    validate_splits(parts)
    cases = evaluation_cases(parts, "test")
    jobs, plan = read_json(raw / "results/master/jobs.json"), read_json(raw / "results/master/master_plan.json")
    progress = pd.read_csv(raw / "results/master/progress.csv")
    require(len(jobs) == len(progress) == 127, "Original grid differs")
    completed = [j for j in jobs if (raw / j["path"] / "status.json").is_file()]
    require(len(completed) == 28 and {j["job"] for j in completed} == set(range(28)), "Unexpected completed coverage")
    require(set(progress.loc[progress.state == "SUCCESS", "path"]) == {j["path"] for j in completed}, "Progress/SUCCESS mismatch")
    for job in jobs:
        require(fingerprint(job["config"]) == job["config_hash"], "Planned config hash mismatch")
    for model, selection in plan["selections"].items():
        frozen = yaml.safe_load((root / f"configs/selected/phase5c/beauty/{model}.yaml").read_text(encoding="utf-8"))
        require(selection == frozen and selection["selection_split"] == "validation", "Phase 5C preselection changed")
    require(file_hash(root / "artifacts/phase5c" / tuning.ARCHIVE_NAME) == tuning.ARCHIVE_SHA256, "Phase 5C original ZIP changed")
    tuning_raw = root / "results/raw/phase5c" / tuning.ARCHIVE_SHA256
    if not any(tuning_raw.glob("results/tuning/beauty/*/*/tuning_plan.json")):
        extract_verified(root / "artifacts/phase5c" / tuning.ARCHIVE_NAME, tuning_raw, tuning.ARCHIVE_SHA256)
    tuning_inventory = pd.read_csv(root / "results/tables/phase5c/archive_inventory.csv")
    for item in tuning_inventory.itertuples(index=False):
        require(file_hash(root / "results/raw/phase5c" / tuning.ARCHIVE_SHA256 / item.member) == item.sha256, "Prior tuning raw artifact changed")
    rows, metrics, histories, validations, signatures, e1, e2 = [], [], [], [], set(), {}, {}
    e2_configs = []
    for job in completed:
        config = job["config"]
        if job["model"] != "mostpopular":
            expected = plan["e2_settings"] if job["experiment"] == "E2" else plan["selections"][job["model"]]["hyperparameters"]
            require(all(config[k] == v for k, v in expected.items()), "Frozen hyperparameters changed")
        row, frame, history, validation, signature = prior.audit_run(raw / job["path"], job, parts, cases)
        row["new_in_final_session"] = job["job"] >= 22
        rows.append(row)
        signatures.add(signature)
        test_metrics = read_json(raw / job["path"] / "test_metrics.json")
        for group in ("full", "warm"):
            for metric in prior.METRICS:
                metrics.append({k: row[k] for k in ("experiment", "model", "variant", "seed")} | {"group": group, "metric": metric, "value": test_metrics[group][metric], "users": test_metrics[group]["users"]})
        histories.append(history.assign(experiment=job["experiment"], model=job["model"], variant=job["variant"], seed=job["seed"]))
        validations.append(validation.assign(experiment=job["experiment"], model=job["model"], variant=job["variant"], seed=job["seed"]))
        if job["experiment"] == "E1":
            e1[(job["model"], job["seed"])] = frame
        else:
            require((config["attention"], config["objective"]) == VARIANT_DESIGN[job["variant"]], "E2 variant semantics differ")
            e2_configs.append({k: v for k, v in config.items() if k not in ("seed", "variant", "attention", "objective")})
            e2[(job["variant"], job["seed"])] = frame
        print(f"VERIFIED {job['experiment']} {job['variant']} {job['model']} seed={job['seed']}", flush=True)
    require(len(signatures) == 1 and len(e1) == 16 and len(e2) == 12, "User/population/design mismatch")
    require(len({fingerprint(c) for c in e2_configs}) == 1, "E2 non-factor settings do not match")
    for variant in VARIANT_DESIGN:
        require({s for v, s in e2 if v == variant} == set(prior.SEEDS), "Missing E2 seed")
    runs, per_seed = pd.DataFrame(rows), pd.DataFrame(metrics)
    require(runs.loc[runs.experiment == "E2", "total_parameters"].nunique() == 1, "E2 parameter counts differ")
    summary = per_seed.groupby(["experiment", "model", "variant", "group", "metric"], as_index=False).agg(n_runs=("value", "size"), mean=("value", "mean"), sd=("value", "std"))
    summary["design_complete"] = True
    per_seed["evaluation_split"] = "test"
    summary["evaluation_split"] = "test"
    intervals = []
    averaged = {m: prior.seed_average(e1, m) for m in prior.MODELS}
    for left, right in itertools.combinations(prior.MODELS, 2):
        point, low, high = prior.paired_metric_bootstrap(averaged[left], averaged[right], resamples)
        for index, metric in enumerate(prior.METRICS):
            intervals.append({"experiment": "E1", "group": "full", "left": left, "right": right, "metric": metric,
                              "difference": point[index], "ci_low": low[index], "ci_high": high[index], "users": len(cases),
                              "resamples": resamples, "bootstrap_seed": 42, "seeds": "7,42,2026; deterministic baseline once", "adjustment": "none"})
    for group in ("full", "warm"):
        mask = np.ones(len(cases), dtype=bool) if group == "full" else next(iter(e2.values())).warm_target.to_numpy(dtype=bool)
        for left, right in CONTRASTS:
            lvalues = np.mean([e2[(left, s)][list(prior.METRICS)].to_numpy() for s in prior.SEEDS], axis=0)[mask]
            rvalues = np.mean([e2[(right, s)][list(prior.METRICS)].to_numpy() for s in prior.SEEDS], axis=0)[mask]
            point, low, high = prior.paired_metric_bootstrap(lvalues, rvalues, resamples)
            for index, metric in enumerate(prior.METRICS):
                intervals.append({"experiment": "E2", "group": group, "left": left, "right": right, "metric": metric,
                                  "difference": point[index], "ci_low": low[index], "ci_high": high[index], "users": int(mask.sum()),
                                  "resamples": resamples, "bootstrap_seed": 42, "seeds": "7,42,2026", "adjustment": "none"})
            print("BOOTSTRAP", group, left, right, flush=True)
    private = root / "results/derived/course_final" / ARCHIVE_SHA256
    groups, segments, pairs = prior.error_analysis(e1, parts, read_json(bundle / "metadata.json"), private)
    tables = root / "results/tables/course_final"
    tables.mkdir(parents=True, exist_ok=True)
    outputs = {"runs": runs, "per_seed_metrics": per_seed, "metric_summary": summary, "paired_bootstrap": pd.DataFrame(intervals),
               "epoch_history": pd.concat(histories, ignore_index=True), "validation_curves": pd.concat(validations, ignore_index=True),
               "e8_error_groups": groups, "e8_segments": segments, "e8_pair_outcomes": pairs, "original_progress": progress,
               "archive_inventory": pd.DataFrame(inventory), "prior_run_comparison": pd.DataFrame(unchanged)}
    for name, data in outputs.items():
        data.to_csv(tables / f"{name}.csv", index=False)
    closed = {"training_closed": True, "source_archive_sha256": ARCHIVE_SHA256,
              "reason": "Final Data Mining course scope closed for resource/course limits, not test scores",
              "jobs": closed_jobs(jobs, {j["path"] for j in completed})}
    history_dir = root / "configs/experiments/history"
    history_dir.mkdir(parents=True, exist_ok=True)
    current_jobs = root / "configs/experiments/course_jobs.json"
    historical_jobs = history_dir / "course_jobs_session1.json"
    if not historical_jobs.exists() and current_jobs.exists() and not read_json(current_jobs).get("training_closed"):
        historical_jobs.write_bytes(current_jobs.read_bytes())
    scope = root / "configs/experiments/course_scope.yaml"
    history_scope = history_dir / "course_scope_session1.yaml"
    if not history_scope.exists() and scope.exists():
        history_scope.write_bytes(scope.read_bytes())
    write_json(current_jobs, closed)
    report = {"state": "VERIFIED", "source_archive": ARCHIVE_NAME, "archive_sha256": ARCHIVE_SHA256,
              "previous_archive_sha256": prior.ARCHIVE_SHA256, "archive_files": len(inventory), "prior_runs_unchanged": 22,
              "prior_run_files_compared": len(unchanged), "prior_changed_files": 0,
              "phase5c_raw_files_hash_reverified": len(tuning_inventory), "phase5c_selection_continuity": True,
              "deduplicated_prior_members": sum(i["deduplicated_prior_content"] for i in inventory),
              "reused_prior_logical_bytes": sum(i["bytes"] for i in inventory if i["deduplicated_prior_content"]),
              "runs_verified": len(runs), "new_E2_runs": 6, "checkpoint_files_verified": 55,
              "hashed_artifacts_verified": int(runs.hashed_artifacts.sum()), "split_version": SPLIT,
              "dataset_bytes_rehashed_against_archived_manifest": True, "all_user_predictions_verified": True,
              "test_users": len(cases), "E1": "COMPLETE_16_OF_16", "E2": "COMPLETE_12_OF_12",
              "training_closed": True, "remaining_requested_training_runs": 0,
              "E3": "DEFERRED", "E4": "DEFERRED", "E5": "DEFERRED", "TIGER": "THEORY_ONLY", "CRAMER": "THEORY_ONLY",
              "E7_recorded_analysis": "COMPLETE", "E7_dedicated_GPU_profiling": "DEFERRED", "E8": "COMPLETE",
              "original_grid_reference": "28/127; full grid not requested by closed course scope",
              "bootstrap_resamples": resamples, "bootstrap_seed": 42, "bootstrap_scope": "seed-averaged user contributions; percentile 95%; pointwise/unadjusted; conditional on trained seeds",
              "new_experimental_training_tuning_inference_or_GPU_profiling_locally": False,
              "session_duration_source": "user-reported approximately 5667 seconds, not an independently saved wall-clock field",
              "training_source_limit": "archived driver/plan, no complete training-source snapshot attestation"}
    write_json(tables / "artifact_audit.json", report)
    write_json(private / "analysis_provenance.json", {"archive_sha256": ARCHIVE_SHA256, "source_sha256": file_hash(Path(__file__)),
                                                    "table_hashes": {p.name: file_hash(p) for p in tables.iterdir() if p.is_file()}})
    return outputs, report
