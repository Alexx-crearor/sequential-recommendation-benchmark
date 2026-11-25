"""E7/E8 reuse verified E1 artifacts; never retrain or choose configs on test scores."""
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
import _bootstrap
from src.utils.artifacts import verify_bundle, write_json, fingerprint
from src.utils.runs import verify_run


def run_analysis(experiment, bundle, results_root, output_root, device="cpu"):
    if bundle is None:
        raise ValueError("Analysis requires --bundle to resolve dataset/split provenance")
    manifest = verify_bundle(bundle)
    dataset = manifest["policy"]["dataset"]
    sources = []
    for status in (Path(results_root) / "E1" / dataset).rglob("status.json"):
        config = yaml.safe_load((status.parent / "config.yaml").read_text())
        if config.get("pilot") or config["split_version"] != manifest["split_version"]:
            continue
        verify_run(status.parent)
        sources.append((status.parent, config))
    if not sources:
        raise ValueError("No verified non-pilot E1 sources for this dataset/split")
    identity = fingerprint({"experiment": experiment, "sources": [str(path) for path, config in sources]})
    output = Path(output_root) / experiment / dataset / identity
    output.mkdir(parents=True, exist_ok=True)
    if experiment == "E7":
        from src.models.base import SequentialModel
        from src.models.popularity import MostPopular
        from src.evaluation.efficiency import inference_profile
        rows = []
        for path, config in sources:
            model = MostPopular.load(path / "checkpoint.pt") if config["model"] == "mostpopular" else SequentialModel.load(path / "checkpoint.pt", device)
            predictions = pd.read_parquet(path / "predictions.parquet")
            profile = inference_profile(model, predictions.iloc[0].history)
            rows.append({"model": config["model"], "seed": config["seed"], "run_path": str(path), "measurement_device": "cpu" if config["model"] == "mostpopular" else device,
                         **json.loads((path / "efficiency.json").read_text()), **profile})
        pd.json_normalize(rows).to_csv(output / "table_08_efficiency.csv", index=False)
    elif experiment == "E8":
        mapping, seen = {}, set()
        for path, config in sources:
            key = (config["model"], config["seed"])
            if key in seen:
                raise ValueError("Multiple E1 configurations per model/seed: supply a results root containing only preselected final runs")
            seen.add(key)
            frame = pd.read_parquet(path / "predictions.parquet")
            mapping[key] = frame.set_index("user_id")
        all_rows = []
        for seed in sorted({key[1] for key in mapping}):
            models = {name: frame for (name, s), frame in mapping.items() if s == seed or name == "mostpopular"}
            user_sets = [set(frame.index) for frame in models.values()]
            if any(users != user_sets[0] for users in user_sets):
                raise ValueError("Inconsistent E8 user sets")
            for user in sorted(user_sets[0]):
                records = {name: frame.loc[user] for name, frame in models.items()}
                if len({int(row.target) for row in records.values()}) != 1:
                    raise ValueError("Inconsistent E8 targets")
                correct = {name: row.target_rank <= 10 for name, row in records.items()}
                categories = []
                if "sasrec_ce" in correct and "bert4rec" in correct:
                    if correct["sasrec_ce"] and not correct["bert4rec"]: categories.append("sasrec_only_vs_bert")
                    if correct["bert4rec"] and not correct["sasrec_ce"]: categories.append("bert_only_vs_sasrec")
                if "fenrec" in correct and "sasrec_ce" in correct and correct["fenrec"] and not correct["sasrec_ce"]:
                    categories.append("fenrec_only_vs_sasrec")
                if not any(correct.values()): categories.append("all_available_models_incorrect")
                first = next(iter(records.values()))
                all_rows.append({"user_id": int(user), "seed": int(seed), "target": int(first.target), "history": list(map(int, first.history)),
                                 "history_length": len(first.history), "categories": categories,
                                 "models": {name: {"top_k": list(map(int, row.top_k)), "target_rank": int(row.target_rank)} for name, row in records.items()}})
        write_json(output / "error_cases.json", all_rows)
        metadata = json.loads((Path(bundle) / "metadata.json").read_text())
        write_json(output / "item_metadata.json", metadata)
        train = pd.read_parquet(Path(bundle) / "train.parquet")
        write_json(output / "training_popularity.json", {str(k): int(v) for k, v in train.item_id.value_counts().items()})
        pd.DataFrame([{key: row[key] for key in ("user_id", "seed", "target", "history_length", "categories")} for row in all_rows]).to_csv(output / "table_12_error_cases.csv", index=False)
    else:
        raise ValueError("Only E7/E8 are supported")
    write_json(output / "sources.json", {"split_version": manifest["split_version"], "sources": [str(path) for path, config in sources], "interpretation": "Descriptive errors only; no causal claims"})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["E7", "E8"], required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--output", type=Path, default=Path("results/analysis"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(run_analysis(args.experiment, args.bundle, args.results_root, args.output, args.device))
