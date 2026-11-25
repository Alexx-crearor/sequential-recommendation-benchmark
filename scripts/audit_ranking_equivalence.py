"""Compare ranking implementations using the same real checkpoints and validation users."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
import _bootstrap
from src.models.base import SequentialModel
from src.evaluation.neural_ranking import evaluate_neural
from benchmarks.phase34.ranking import evaluate_neural as reference
from src.utils.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--pilots-root", type=Path, default=Path("results/pilots/PILOT/beauty"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    parts = {key: pd.read_parquet(args.bundle/f"{key}.parquet") for key in ("train", "validation", "test")}
    rows = []
    for name in ["gru4rec_ce", "sasrec_bce", "sasrec_ce", "bert4rec", "fenrec"]:
        checkpoint = next((args.pilots_root/name).glob("*/7/checkpoint.pt"))
        model = SequentialModel.load(checkpoint, args.device)
        expected_metrics, expected = reference(model, parts, "validation", 4096, 1024)
        metrics, actual = evaluate_neural(model, parts, "validation", 4096, 1024)
        rank_mismatches = int((expected.target_rank.to_numpy()!=actual.target_rank.to_numpy()).sum())
        top_mismatches = sum(list(a)!=list(b) for a,b in zip(expected.top_k,actual.top_k))
        rows.append({"model": name, "checkpoint": str(checkpoint), "users": len(actual), "rank_mismatches": rank_mismatches,
                     "top_k_mismatches": top_mismatches, "metrics_equal": metrics == expected_metrics})
        del model
    write_json(args.output, rows)
    print(json.dumps(rows, indent=2))
    if any(row["rank_mismatches"] or row["top_k_mismatches"] or not row["metrics_equal"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__": main()
