"""Exact static-score ranking: sort once and subtract excluded items' ranks."""
import numpy as np
import pandas as pd
from src.data.splitting import evaluation_cases
from src.evaluation.metrics import contributions


def evaluate_popularity(model, parts, split="test"):
    ids = np.arange(1, model.num_items+1)
    order = ids[np.lexsort((ids, -model.counts[1:model.num_items+1]))]
    position = np.zeros(model.num_items+1, dtype=np.int64)
    position[order] = np.arange(1, model.num_items+1)
    warm, rows = set(parts["train"].item_id), []
    for case in evaluation_cases(parts, split):
        excluded = set(case["history"]) - {case["target"]}
        target_position = position[case["target"]]
        rank = int(target_position - sum(position[item] < target_position for item in excluded))
        # Only the first K + excluded_count positions can be needed.
        top = [int(item) for item in order[:20+len(excluded)] if item not in excluded][:20]
        rows.append({**case, "top_k": top, "target_rank": rank, "warm_target": case["target"] in warm, **contributions(rank)})
    predictions = pd.DataFrame(rows)
    metrics = {}
    for name, subset in (("full", predictions), ("warm", predictions.loc[predictions.warm_target])):
        metrics[name] = {"users": len(subset), **{key: float(subset[key].mean()) if len(subset) else None for key in contributions(1)}}
    metrics["warm_coverage"] = float(predictions.warm_target.mean())
    return metrics, predictions
