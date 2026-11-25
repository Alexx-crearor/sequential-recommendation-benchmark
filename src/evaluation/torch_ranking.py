"""Exact streaming full-catalogue ranking, device resident until final results."""
import numpy as np
import pandas as pd
import torch
from src.data.splitting import evaluation_cases
from src.evaluation.metrics import contributions


@torch.no_grad()
def evaluate_torch(model, parts, split="test", item_batch_size=4096, user_limit=None):
    model.eval()
    cases = evaluation_cases(parts, split)
    if user_limit is not None:
        cases = cases[:user_limit]
    if not cases:
        raise ValueError("Empty evaluation population")
    warm, rows = set(parts["train"].item_id), []
    device = model.device
    for offset in range(0, len(cases), model.config["eval_batch_size"]):
        batch = cases[offset:offset+model.config["eval_batch_size"]]
        reps = model.representations([case["history"] for case in batch])
        targets = torch.tensor([case["target"] for case in batch], device=device)
        unique, inverse = targets.unique(sorted=True, return_inverse=True)
        target_scores = model.logits(reps, unique)[torch.arange(len(batch), device=device), inverse].float()
        ranks = torch.ones(len(batch), dtype=torch.long, device=device)
        best_ids = torch.empty(len(batch), 0, dtype=torch.long, device=device)
        best_scores = torch.empty(len(batch), 0, device=device)
        excluded_rows, excluded_items = [], []
        for index, case in enumerate(batch):
            items = sorted(set(case["history"]) - {case["target"]})
            excluded_rows.extend([index]*len(items))
            excluded_items.extend(items)
        excluded_rows = torch.tensor(excluded_rows, dtype=torch.long, device=device)
        excluded_items = torch.tensor(excluded_items, dtype=torch.long, device=device)
        finite = torch.isfinite(target_scores).all()
        for start in range(1, model.num_items+1, item_batch_size):
            end = min(start+item_batch_size, model.num_items+1)
            ids = torch.arange(start, end, device=device)
            scores = model.logits(reps, ids).float()
            finite = finite & torch.isfinite(scores).all()
            target_in_chunk = (targets >= start) & (targets < end)
            target_rows = target_in_chunk.nonzero(as_tuple=True)[0]
            scores[target_rows, targets[target_rows]-start] = target_scores[target_rows]
            seen = (excluded_items >= start) & (excluded_items < end)
            scores[excluded_rows[seen], excluded_items[seen]-start] = -torch.inf
            ranks += ((scores > target_scores[:, None]) | ((scores == target_scores[:, None]) & (ids[None] < targets[:, None]))).sum(-1)
            merged_ids = torch.cat([best_ids, ids[None].expand(len(batch), -1)], -1)
            merged_scores = torch.cat([best_scores, scores], -1)
            # Chunk IDs ascend; previous ties already ascend and have lower IDs.
            # Stable score sorting therefore implements exact score-desc/ID-asc order.
            order = torch.argsort(merged_scores, dim=-1, descending=True, stable=True)[:, :20]
            best_ids = merged_ids.gather(1, order)
            best_scores = merged_scores.gather(1, order)
        if not finite.item():
            raise ValueError("Nonfinite catalogue or target scores")
        top = best_ids.cpu().tolist()
        valid = torch.isfinite(best_scores).cpu().tolist()
        rank_values = ranks.cpu().tolist()
        for index, case in enumerate(batch):
            rank = int(rank_values[index])
            rows.append({**case, "top_k": [item for item, keep in zip(top[index], valid[index]) if keep],
                         "target_rank": rank, "warm_target": case["target"] in warm, **contributions(rank)})
    predictions = pd.DataFrame(rows)
    metrics = {}
    for name, subset in (("full", predictions), ("warm", predictions.loc[predictions.warm_target])):
        metrics[name] = {"users": len(subset), **{key: float(subset[key].mean()) if len(subset) else None for key in contributions(1)}}
    metrics["warm_coverage"] = float(predictions.warm_target.mean())
    return metrics, predictions
