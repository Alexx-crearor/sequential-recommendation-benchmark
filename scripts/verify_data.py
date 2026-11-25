"""Verify immutable hashes, chronological splits and all neural training boundaries."""
import argparse
import json
from pathlib import Path
import pandas as pd
import _bootstrap
from src.utils.artifacts import verify_bundle, write_json
from src.data.splitting import validate_splits
from src.data.neural_dataset import TrainingDataset, verify_training_boundary
from src.utils.configuration import resolve_config


def audit(bundle):
    manifest = verify_bundle(bundle)
    bundle = Path(bundle)
    parts = {key: pd.read_parquet(bundle / f"{key}.parquet") for key in ("train", "validation", "test")}
    validate_splits(parts)
    full = pd.read_parquet(bundle / "interactions.parquet")
    if set(full.event_id) != set(pd.concat(parts.values()).event_id):
        raise ValueError("Split event union differs from processed interactions")
    if full.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Repeated user/item events violate policy")
    core = manifest["policy"]["core"]
    if full.groupby("user_id").size().min() < core or full.groupby("item_id").size().min() < core:
        raise ValueError("Core integrity failure")
    mappings = json.loads((bundle / "mappings.json").read_text())
    statistics = json.loads((bundle / "statistics.json").read_text())
    if statistics["users"] != full.user_id.nunique() or statistics["items"] != full.item_id.nunique() or statistics["interactions"] != len(full):
        raise ValueError("Dataset statistics mismatch")
    samples = {}
    for name in ["gru4rec_ce", "sasrec_bce", "sasrec_ce", "bert4rec", "attention_ablation", "fenrec"]:
        dataset = TrainingDataset(parts["train"], len(mappings["items"]), resolve_config(name))
        verify_training_boundary(dataset, parts)
        samples[name] = len(dataset)
    return {"state": "PASSED", "split_version": manifest["split_version"], "statistics": statistics,
            "training_sample_counts": samples, "checks": ["sha256_artifacts", "chronology", "event_target_separation", "iterative_core", "repeat_policy", "train_only_neural_samples", "train_only_fenrec_future_windows"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.bundle)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
