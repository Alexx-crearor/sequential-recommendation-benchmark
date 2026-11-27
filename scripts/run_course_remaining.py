"""Plan/status locally; explicitly run archived unchanged jobs on Kaggle only."""
import argparse
import json
import os
from pathlib import Path
import _bootstrap
from src.utils.artifacts import verify_bundle, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_bootstrap.ROOT / "configs/experiments/course_jobs.json")
    parser.add_argument("--results-root", type=Path, default=Path("results/core"))
    parser.add_argument("--experiment", choices=["E1", "E2", "E3"])
    parser.add_argument("--archive-job-id", type=int)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run", action="store_true", help="Execute selected jobs; permitted only in a Kaggle runtime")
    args = parser.parse_args()
    current = _bootstrap.ROOT / "configs/experiments/course_jobs.json"
    if args.run and current.exists() and json.loads(current.read_text(encoding="utf-8")).get("training_closed"):
        parser.error("The Data Mining course experimental scope is CLOSED: no further training, tuning or GPU profiling is requested")
    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))["jobs"]
    selected = [j for j in jobs if (not args.experiment or j["experiment"] == args.experiment)
                and (args.archive_job_id is None or j["archive_job_id"] == args.archive_job_id)]
    if not selected:
        parser.error("No matching jobs")
    if args.run and not (Path("/kaggle/working").is_dir() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")):
        parser.error("Training is restricted to Kaggle; local mode only prints the plan")
    if args.run and args.bundle is None:
        parser.error("--bundle required for execution")
    for job in selected:
        config = job["config"]
        if fingerprint(config) != job["config_hash"]:
            raise ValueError("Frozen job configuration changed")
        relative = Path(job["path"]).relative_to("results/core")
        path = args.results_root / relative
        state = "LOCAL_ARTIFACTS_NOT_RESTORED"
        if (path / "status.json").exists():
            state = json.loads((path / "status.json").read_text())["state"]
            if state == "SUCCESS":
                from src.utils.runs import verify_run
                verify_run(path)
        print(job["archive_job_id"], job["experiment"], job["model"], job["variant"], job["seed"], job["retention"],
              "local_state="+state, "archived_evidence="+job["evidence_state"], flush=True)
        if not args.run or state == "SUCCESS":
            continue
        if job["evidence_state"] == "ARCHIVED_SUCCESS":
            raise ValueError("Restore the completed original run first; this workflow will not retrain archived SUCCESS jobs")
        if job["evidence_state"] == "REUSE_AVAILABLE":
            source = args.results_root / Path(job["reuse_source"]).relative_to("results/core")
            if not (source / "status.json").is_file():
                raise ValueError("Restore the matching E1 source before E3 retention-1.0 reuse; do not retrain it")
        if verify_bundle(args.bundle)["split_version"] != config["split_version"]:
            raise ValueError("Dataset split differs from frozen plan")
        if job["model"] == "mostpopular":
            from run_train import run
            run(args.bundle, args.results_root, job["experiment"])
        else:
            from src.training.trainer import run_neural
            base = {k: v for k, v in config.items() if k not in ("dataset", "split_version", "retention_fingerprint", "device_type")}
            actual = run_neural(args.bundle, base, args.results_root, job["experiment"], args.device, True)
            if Path(actual) != path:
                raise ValueError("Execution path differs from frozen archived identity")


if __name__ == "__main__":
    main()
