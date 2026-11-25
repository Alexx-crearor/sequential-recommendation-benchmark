import argparse
from pathlib import Path
import pandas as pd
import yaml
import _bootstrap
from src.utils.grid import expand_grid
from run_train import run


def main():
    parser = argparse.ArgumentParser(description="Plan or execute selected E1-E5 configurations")
    parser.add_argument("--experiment", required=True, choices=["all"] + [f"E{i}" for i in range(1, 9)])
    parser.add_argument("--profile", default="full_research", choices=["pilot", "core_research", "full_research"])
    parser.add_argument("--final-seeds", action="store_true", help="Use final three-seed design after validation-only selection")
    parser.add_argument("--selection", type=Path, help="Validation-selected hyperparameter YAML")
    parser.add_argument("--dataset", choices=["beauty", "sports", "toys"])
    parser.add_argument("--model")
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int, choices=[7, 42, 2026])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    parser.add_argument("--retention", type=float)
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/tables/experiment_manifest.csv"))
    args = parser.parse_args()
    if args.results_root is None:
        args.results_root = Path("results/pilots_phase5a" if args.profile == "pilot" else "results/core" if args.profile == "core_research" else "results/runs")
    from src.utils.profiles import load_profile, profile_overrides
    grid, profile = load_profile(args.profile, args.final_seeds)
    rows = expand_grid(grid, None if args.experiment == "all" else args.experiment)
    for key in ("dataset", "model", "variant", "seed"):
        value = getattr(args, key)
        if value is not None:
            rows = [row for row in rows if row.get(key) == value or "task" in row]
    if args.retention is not None:
        rows = [row for row in rows if row.get("retention") == args.retention]
    if args.max_seq_length is not None:
        rows = [row for row in rows if row.get("max_seq_length") == args.max_seq_length]
    if not rows:
        parser.error("No configurations match the selectors")
    if args.dry_run:
        for row in rows:
            row["profile"] = args.profile
            row["seed_stage"] = "final" if args.final_seeds else "initial"
            if row.get("model") and row["model"] != "mostpopular" and row["model"] != "tiger":
                from src.utils.configuration import resolve_config
                runtime = resolve_config(row["model"], {**profile_overrides(profile, row["model"]), "variant": row["variant"], **grid["experiments"][row["experiment"]].get("variants", {}).get(row["variant"], {})})
                row["max_epochs"] = runtime["max_epochs"]
                row["validation_every"] = runtime["validation_every"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"Planned {len(rows)} entries: {args.output}. Pilot estimates are in results/reports; full-grid estimates remain unresolved. Reuse candidates require resolved split/config equality.")
        return
    if args.experiment in ("E7", "E8"):
        from run_analysis import run_analysis
        print(run_analysis(args.experiment, args.bundle, args.results_root, Path("results/analysis") / args.profile, args.device))
        return
    if args.experiment not in ("E1", "E2", "E3", "E4", "E5") or not args.bundle:
        parser.error("Execution requires one of E1-E5 and --bundle; use --dry-run for other phases")
    from src.utils.artifacts import verify_bundle
    actual_dataset = verify_bundle(args.bundle)["policy"]["dataset"]
    if args.dataset and args.dataset != actual_dataset:
        parser.error("Bundle dataset does not match --dataset")
    rows = [row for row in rows if row.get("dataset") == actual_dataset]
    if any(row["model"] == "tiger" for row in rows):
        parser.error("TIGER remains Phase 6; select an implemented model explicitly")
    from src.utils.configuration import resolve_config
    from src.training.trainer import run_neural
    user_overrides = yaml.safe_load(args.config.read_text()) if args.config else {}
    for setting in args.set:
        key, value = setting.split("=", 1)
        user_overrides[key] = yaml.safe_load(value)
    for row in rows:
        if row["model"] == "mostpopular":
            print(run(args.bundle, args.results_root, args.experiment))
            continue
        spec = grid["experiments"][args.experiment]
        variant_settings = spec.get("variants", {}).get(row["variant"], {})
        selected = {}
        if args.selection:
            selection = yaml.safe_load(args.selection.read_text())
            if selection.get("pilot_only") and not profile.get("overrides", {}).get("pilot", False):
                parser.error("Pilot-only tuning cannot select a research configuration")
            if selection["model"] != row["model"]:
                parser.error("Selected hyperparameters belong to another model")
            if selection.get("dataset") != actual_dataset or selection.get("split_version") != verify_bundle(args.bundle)["split_version"]:
                parser.error("Selection dataset/split provenance differs from bundle")
            selected = selection["hyperparameters"]
        overrides = {**profile_overrides(profile, row["model"]), **selected, **variant_settings, "variant": row["variant"], "seed": row["seed"],
                     "retention": row["retention"], "max_seq_length": row["max_seq_length"], **user_overrides}
        config = resolve_config(row["model"], overrides)
        print(run_neural(args.bundle, config, args.results_root, args.experiment, args.device, args.resume))


if __name__ == "__main__":
    main()
