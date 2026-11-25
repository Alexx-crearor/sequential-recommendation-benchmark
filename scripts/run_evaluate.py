"""Evaluate a Phase 2 checkpoint against a verified bundle without changing its run."""
import argparse
import json
from pathlib import Path
import pandas as pd
import _bootstrap
from src.utils.artifacts import verify_bundle, write_json
from src.models.popularity import MostPopular
from src.evaluation.evaluate import evaluate
from src.data.splitting import validate_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_bundle(args.bundle)
    if args.output.exists():
        parser.error("Output exists; choose a new directory to preserve artifacts")
    parts = {key: pd.read_parquet(args.bundle / f"{key}.parquet") for key in ("train", "validation", "test")}
    validate_splits(parts)
    # A standalone array checkpoint has no split provenance; require the owning verified run.
    from src.utils.runs import verify_run
    import yaml
    verify_run(args.checkpoint.parent)
    config = yaml.safe_load((args.checkpoint.parent / "config.yaml").read_text())
    if config["model"] == "mostpopular":
        model = MostPopular.load(args.checkpoint)
    else:
        from src.models.base import SequentialModel
        model = SequentialModel.load(args.checkpoint)
    mappings = json.loads((args.bundle / "mappings.json").read_text())
    if model.num_items != len(mappings["items"]):
        parser.error("Checkpoint catalogue size differs from bundle")
    manifest = verify_bundle(args.bundle)
    if config["split_version"] != manifest["split_version"]:
        parser.error("Checkpoint split provenance differs from bundle")
    metrics, predictions = evaluate(model, parts, args.split)
    args.output.mkdir(parents=True)
    write_json(args.output / f"{args.split}_metrics.json", metrics)
    predictions.to_parquet(args.output / "predictions.parquet", index=False)
    print(args.output)


if __name__ == "__main__":
    main()
