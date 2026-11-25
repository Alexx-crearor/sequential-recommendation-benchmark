import numpy as np


def sample_negatives(num_items, excluded, count, rng):
    if num_items < 1 or count < 0:
        raise ValueError("Invalid catalogue size or sample count")
    candidates = np.setdiff1d(np.arange(1, num_items + 1), list(excluded))
    if len(candidates) < count:
        raise ValueError("Insufficient eligible negative items")
    return rng.choice(candidates, size=count, replace=False)
