"""Bounded seed-7 validation-only search. Test evaluation is structurally disabled."""
import argparse
import itertools
import json
import random
from pathlib import Path
import yaml
import _bootstrap
from src.utils.profiles import load_profile, profile_overrides
from src.utils.configuration import resolve_config, MODELS
from src.utils.artifacts import verify_bundle, write_json, fingerprint
from src.training.trainer import run_neural
from src.utils.validation_runs import verify_validation_run


def candidates(profile, model, limit):
    tuning = profile.get("tuning", {})
    if not 1 <= limit <= tuning.get("max_allowed_trials", 12):
        raise ValueError("Tuning budget must be between 1 and the configured maximum (at most 12)")
    search = {**tuning.get("search", {}), **tuning.get("model_search", {}).get(model, {})}
    keys = sorted(search)
    values = [dict(zip(keys, combination)) for combination in itertools.product(*(search[key] for key in keys))]
    random.Random(7).shuffle(values)
    # Always measure the model's documented defaults first, then distinct bounded trials.
    baseline = resolve_config(model, profile_overrides(profile, model))
    default = {key: baseline[key] for key in keys}
    unique = {fingerprint(default): default}
    for value in values:
        unique.setdefault(fingerprint(value), value)
    return list(unique.values())[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--profile", choices=["core_research", "full_research"], default="core_research")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-root", type=Path, default=Path("results/tuning"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, help="Explicit bounded-budget overrides (pilot limits only with pilot=true)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    _, profile = load_profile(args.profile)
    manifest = verify_bundle(args.bundle)
    dataset = manifest["policy"]["dataset"]
    limit = args.trials if args.trials is not None else profile["tuning"]["max_trials_per_model_dataset"]
    trials = candidates(profile, args.model, limit)
    overrides = yaml.safe_load(args.config.read_text()) if args.config else {}
    tuning_budget = {**profile["tuning"].get("overrides", {}), **profile["tuning"].get("model_overrides", {}).get(args.model, {})}
    runtime_overrides = {**profile_overrides(profile, args.model), **tuning_budget, **overrides}
    plan = {"model": args.model, "dataset": dataset, "split_version": manifest["split_version"], "profile": args.profile,
            "seed": 7, "selection_metric": "validation_ndcg@10", "trials": trials, "evaluate_test": False, "runtime_overrides": runtime_overrides}
    identity = fingerprint(plan)
    directory = args.results_root / dataset / args.model / identity
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "tuning_plan.json", plan)
    if args.dry_run:
        print(directory / "tuning_plan.json")
        return
    results = []
    for index, parameters in enumerate(trials):
        config = resolve_config(args.model, {**runtime_overrides, **parameters, **overrides,
                                "seed": 7, "evaluate_test": False, "tuning_trial": index, "tuning_plan_hash": identity})
        run = run_neural(args.bundle, config, directory / "runs", "TUNING", args.device, args.resume)
        verify_validation_run(run)
        metrics = json.loads((run / "validation_selection.json").read_text())
        results.append({"trial": index, "hyperparameters": parameters, "validation_ndcg@10": metrics["full"]["ndcg@10"], "run_path": str(run)})
        write_json(directory / "trial_results.json", results)
    # Stable earliest-trial tie break; no test metric is loaded anywhere in this script.
    winner = max(results, key=lambda row: (row["validation_ndcg@10"], -row["trial"]))
    selection = {"model": args.model, "dataset": dataset, "split_version": manifest["split_version"], "profile": args.profile,
                 "selection_split": "validation", "seed": 7, **winner, "tuning_plan_hash": identity,
                 "pilot_only": bool(overrides.get("pilot", False)), "trials_completed": len(results)}
    (directory / "selected_config.yaml").write_text(yaml.safe_dump(selection), encoding="utf-8")
    print(directory / "selected_config.yaml")


if __name__ == "__main__":
    main()
