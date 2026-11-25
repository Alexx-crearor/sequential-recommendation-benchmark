import numpy as np
from src.utils.artifacts import fingerprint


def subsample_train(train, ratio, seed):
    if not 0 < ratio <= 1:
        raise ValueError("retention ratio must be in (0, 1]")
    rng = np.random.default_rng(seed)
    selected = []
    # Keep at least one event per user so the held-out user set remains fixed.
    for _, group in train.sort_values(["user_id", "timestamp", "item_id"]).groupby("user_id"):
        count = max(1, int(np.floor(len(group) * ratio)))
        selected.extend(rng.choice(group.event_id.to_numpy(), count, replace=False).tolist())
    retained = train.loc[train.event_id.isin(selected)].copy()
    manifest = {"seed": seed, "requested_ratio": ratio, "realized_ratio": len(retained) / len(train), "event_ids": sorted(selected)}
    manifest["fingerprint"] = fingerprint(manifest)
    return retained, manifest
