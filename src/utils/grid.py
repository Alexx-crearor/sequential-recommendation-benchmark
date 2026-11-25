import itertools
from src.utils.artifacts import fingerprint


def expand_grid(grid, selected=None):
    rows, seen = [], {}
    for experiment, spec in grid["experiments"].items():
        if "reuse" in spec:
            rows.append({"experiment": experiment, "task": spec["task"], "reuse_experiment": spec["reuse"], "state": "PLANNED_ANALYSIS"})
            continue
        variants = spec.get("variants", {"default": {}})
        for dataset, variant, ratio, length in itertools.product(spec["datasets"], variants, spec.get("retention", [1.0]), spec.get("max_seq_length", [grid["defaults"]["max_seq_length"]])):
            settings = variants[variant]
            for model in spec.get("models", [settings.get("model")]):
                for seed in ([7] if model == "mostpopular" else grid["seeds"]):
                    config = {**grid["defaults"], **settings, "dataset": dataset, "model": model, "seed": seed, "retention": ratio, "max_seq_length": length, "variant": variant}
                    # Only exact training settings can reuse. Normalize documented identical aliases.
                    training = dict(config)
                    key = fingerprint(training)
                    row = {"experiment": experiment, **config, "planning_fingerprint": key,
                           "split_version": "UNRESOLVED", "reuse_candidate": seen.get(key, ""),
                           "state": "PLANNED", "estimated_seconds": None, "estimated_peak_gpu_bytes": None}
                    seen.setdefault(key, f"{experiment}/{dataset}/{model}/{variant}/{seed}")
                    rows.append(row)
    return [row for row in rows if selected is None or row["experiment"] == selected]
