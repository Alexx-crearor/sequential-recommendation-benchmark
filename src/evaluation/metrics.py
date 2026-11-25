import math


def contributions(rank, ks=(5, 10, 20)):
    if rank is not None and (not isinstance(rank, int) or rank < 1):
        raise ValueError("Rank must be a positive integer or None")
    result = {}
    for k in ks:
        if k < 1:
            raise ValueError("K must be positive")
        hit = rank is not None and rank <= k
        result[f"recall@{k}"] = float(hit)
        result[f"ndcg@{k}"] = 1 / math.log2(rank + 1) if hit else 0.0
    result["mrr@10"] = 1 / rank if rank is not None and rank <= 10 else 0.0
    return result
