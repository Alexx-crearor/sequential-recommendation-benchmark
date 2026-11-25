"""Write immutable, content-addressed dataset bundles."""
import argparse
import json
from pathlib import Path
import yaml
import _bootstrap
from src.data.preprocessing import load_interactions, preprocess, load_metadata
from src.data.splitting import leave_one_out
from src.utils.artifacts import file_hash, fingerprint, write_json, verify_bundle


def prepare(dataset, interactions, metadata, output_root, core=5):
    policy = {"schema": 1, "dataset": dataset, "core": core, "rating": "all_1_to_5_implicit",
              "repeats": "earliest_user_item", "ties": "timestamp_then_lexicographic_asin", "split": "per_user_leave_two_out"}
    inputs = {"interactions_sha256": file_hash(interactions), "metadata_sha256": file_hash(metadata)}
    identity = fingerprint({**policy, **inputs})
    directory = Path(output_root) / dataset / identity
    if directory.exists():
        verify_bundle(directory)
        return directory
    frame, users, items = preprocess(load_interactions(interactions), core)
    parts = leave_one_out(frame)
    directory.mkdir(parents=True)
    for name, part in parts.items():
        part.to_parquet(directory / f"{name}.parquet", index=False)
    frame.to_parquet(directory / "interactions.parquet", index=False)
    write_json(directory / "mappings.json", {"users": users, "items": items, "padding_id": 0})
    item_metadata = load_metadata(metadata, items)
    write_json(directory / "metadata.json", item_metadata)
    warm = set(parts["train"].item_id)
    stats = {"users": len(users), "items": len(items), "interactions": len(frame),
             "train_interactions": len(parts["train"]), "warm_items": len(warm),
             "metadata_missing": sum(row["metadata_missing"] for row in item_metadata),
             "validation_warm_coverage": float(parts["validation"].item_id.isin(warm).mean()),
             "test_warm_coverage": float(parts["test"].item_id.isin(warm).mean())}
    write_json(directory / "statistics.json", stats)
    files = {p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file()}
    manifest = {"policy": policy, "inputs": inputs, "files": files}
    manifest["split_version"] = fingerprint(manifest)
    write_json(directory / "manifest.json", manifest)
    verify_bundle(directory)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["beauty", "sports", "toys"])
    parser.add_argument("--interactions", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    config = yaml.safe_load((_bootstrap.ROOT / "configs/datasets" / f"{args.dataset}.yaml").read_text())
    interactions = args.interactions or Path(config["interactions"])
    metadata = args.metadata or Path(config["metadata"])
    print(prepare(args.dataset, interactions, metadata, args.output_root, config["core"]))


if __name__ == "__main__":
    main()
