import numpy as np


def paired_bootstrap(left, right, resamples=10000, seed=42):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 1 or left.shape != right.shape or not len(left) or resamples < 1:
        raise ValueError("Paired nonempty arrays and positive resamples required")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Nonfinite contributions")
    delta, rng = left - right, np.random.default_rng(seed)
    means = np.array([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(resamples)])
    low, high = np.quantile(means, [0.025, 0.975])
    return {"difference": float(delta.mean()), "ci_low": float(low), "ci_high": float(high), "resamples": resamples, "seed": seed}
