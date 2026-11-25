import numpy as np


def rank_catalogue(model, history, target, num_items, k=20, item_batch_size=4096):
    """Exact rank with O(item_batch_size + K) working storage; ties by ID."""
    if target < 1 or target > num_items or k < 1 or item_batch_size < 1:
        raise ValueError("Invalid ranking arguments")
    excluded = set(history) - {target}
    target_score = float(model.score_items(history, np.array([target]))[0])
    if not np.isfinite(target_score):
        raise ValueError("Nonfinite target score")
    better, best_ids, best_scores = 0, np.array([], dtype=np.int64), np.array([], dtype=float)
    for start in range(1, num_items + 1, item_batch_size):
        ids = np.arange(start, min(start + item_batch_size, num_items + 1))
        scores = np.asarray(model.score_items(history, ids), dtype=float)
        if scores.shape != ids.shape or not np.isfinite(scores).all():
            raise ValueError("Invalid catalogue scores")
        keep = ~np.isin(ids, list(excluded))
        ids, scores = ids[keep], scores[keep]
        better += int(((scores > target_score) | ((scores == target_score) & (ids < target))).sum())
        all_ids, all_scores = np.concatenate([best_ids, ids]), np.concatenate([best_scores, scores])
        order = np.lexsort((all_ids, -all_scores))[:k]
        best_ids, best_scores = all_ids[order], all_scores[order]
    return best_ids.tolist(), better + 1


def deduplicate(items, eligible):
    seen, output = set(), []
    for item in items:
        if item in eligible and item not in seen:
            output.append(item)
            seen.add(item)
    return output
