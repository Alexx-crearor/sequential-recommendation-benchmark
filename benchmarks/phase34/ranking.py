"""Encode each history once, then score the complete catalogue in batches."""
import numpy as np
import pandas as pd
import torch
def evaluation_cases(parts, split="test"):
    history = parts["train"]
    if split == "test":
        history = pd.concat([history, parts["validation"]])
    histories = history.sort_values(["user_id", "timestamp", "item_id"]).groupby("user_id").item_id.apply(list).to_dict()
    return [{"user_id": int(row.user_id), "history": histories[row.user_id], "target": int(row.item_id)}
            for row in parts[split].sort_values("user_id").itertuples()]
from src.evaluation.metrics import contributions


@torch.no_grad()
def evaluate_neural(model, parts, split="test", item_batch_size=4096, user_limit=None):
    model.eval()
    cases = evaluation_cases(parts, split)
    if user_limit is not None:
        cases = cases[:user_limit]  # deterministic, explicitly pilot-only subset
    warm, rows = set(parts["train"].item_id), []
    batch_size = model.config["eval_batch_size"]
    for offset in range(0, len(cases), batch_size):
        batch = cases[offset:offset+batch_size]
        reps = model.representations([case["history"] for case in batch])
        targets = np.array([case["target"] for case in batch])
        # Gather targets with the same scoring head, without exposing them to the encoder.
        unique, inverse = np.unique(targets, return_inverse=True)
        target_logits = model.logits(reps, torch.as_tensor(unique, device=model.device))
        target_scores = target_logits[torch.arange(len(batch), device=model.device), torch.as_tensor(inverse, device=model.device)].float().cpu().numpy()
        if not np.isfinite(target_scores).all():
            raise ValueError("Nonfinite target scores")
        ranks = np.ones(len(batch), dtype=np.int64)
        best = [[] for _ in batch]
        excluded = [set(case["history"]) - {case["target"]} for case in batch]
        for start in range(1, model.num_items+1, item_batch_size):
            ids = np.arange(start, min(start+item_batch_size, model.num_items+1))
            scores = model.logits(reps, torch.as_tensor(ids, device=model.device)).float().cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite catalogue scores")
            for index, case in enumerate(batch):
                values = scores[index].copy()
                if start <= targets[index] < start+len(ids):
                    values[targets[index]-start] = target_scores[index]
                valid = ~np.isin(ids, list(excluded[index]))
                candidates, values = ids[valid], values[valid]
                ranks[index] += ((values > target_scores[index]) | ((values == target_scores[index]) & (candidates < targets[index]))).sum()
                candidates = np.concatenate([np.array([item for item, score in best[index]], dtype=np.int64), candidates])
                values = np.concatenate([np.array([score for item, score in best[index]]), values])
                order = np.lexsort((candidates, -values))[:20]
                best[index] = list(zip(candidates[order].tolist(), values[order].tolist()))
        for index, case in enumerate(batch):
            rank = int(ranks[index])
            rows.append({**case, "top_k": [item for item, score in best[index]], "target_rank": rank,
                         "warm_target": case["target"] in warm, **contributions(rank)})
    predictions = pd.DataFrame(rows)
    metrics = {}
    for name, subset in (("full", predictions), ("warm", predictions.loc[predictions.warm_target])):
        metrics[name] = {"users": len(subset), **{key: float(subset[key].mean()) if len(subset) else None for key in contributions(1)}}
    metrics["warm_coverage"] = float(predictions.warm_target.mean())
    return metrics, predictions
