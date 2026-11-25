import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.models.popularity import MostPopular
from src.utils.artifacts import file_hash, fingerprint
from src.evaluation.metrics import contributions

REQUIRED = ["config.yaml", "environment.json", "train_history.csv", "validation_metrics.csv", "test_metrics.json", "efficiency.json", "predictions.parquet", "checkpoint.pt", "run.log"]


def verify_run(directory):
    directory = Path(directory)
    status = json.loads((directory / "status.json").read_text())
    if status["state"] != "SUCCESS":
        raise ValueError(f"Run is {status['state']}")
    for name in REQUIRED:
        if not (directory / name).is_file() or file_hash(directory / name) != status["files"].get(name):
            raise ValueError(f"Missing or altered artifact: {name}")
    config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    if fingerprint(config) != status["config_hash"]:
        raise ValueError("Run configuration fingerprint mismatch")
    if config["model"] == "mostpopular":
        model = MostPopular.load(directory / "checkpoint.pt")
    else:
        from src.models.base import SequentialModel
        model = SequentialModel.load(directory / "checkpoint.pt")
        import torch
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Nonfinite checkpoint parameters")
        if model.config != config:
            raise ValueError("Checkpoint configuration mismatch")
        for name, checksum in status["files"].items():
            if file_hash(directory / name) != checksum:
                raise ValueError(f"Missing or altered artifact: {name}")
    predictions = pd.read_parquet(directory / "predictions.parquet")
    metrics = json.loads((directory / "test_metrics.json").read_text())
    for name in ("train_history.csv", "validation_metrics.csv"):
        table = pd.read_csv(directory / name)
        numeric = table.select_dtypes(include="number")
        if not len(table) or not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"Nonfinite or empty history: {name}")
    if len(predictions) != metrics["full"]["users"] or predictions.user_id.duplicated().any() or not len(predictions):
        raise ValueError("Incomplete predictions")
    if (directory / "evaluation_users.csv").exists():
        expected = pd.read_csv(directory / "evaluation_users.csv")
        if set(zip(expected.user_id, expected.target)) != set(zip(predictions.user_id, predictions.target)):
            raise ValueError("Evaluation user or target coverage mismatch")
    for _, row in predictions.iterrows():
        if not 1 <= row.target_rank <= model.num_items:
            raise ValueError("Invalid rank")
        top = list(row.top_k)
        if len(top) != len(set(top)) or any(not 1 <= item <= model.num_items for item in top):
            raise ValueError("Invalid recommendations")
        if len(top) != min(20, model.num_items - len(set(row.history) - {row.target})):
            raise ValueError("Missing recommendations")
        if set(top) & (set(row.history) - {row.target}):
            raise ValueError("Previously seen recommendation")
        if row.target_rank <= len(top) and top[int(row.target_rank) - 1] != row.target:
            raise ValueError("Target rank disagrees with recommendations")
        for key, expected in contributions(int(row.target_rank)).items():
            if not np.isfinite(row[key]) or not np.isclose(row[key], expected):
                raise ValueError("Incorrect metric contribution")
    for group, subset in (("full", predictions), ("warm", predictions.loc[predictions.warm_target])):
        if len(subset) != metrics[group]["users"]:
            raise ValueError("Metric user count mismatch")
        for key in contributions(1):
            expected = float(subset[key].mean()) if len(subset) else None
            actual = metrics[group][key]
            if (expected is None and actual is not None) or (expected is not None and (actual is None or not np.isclose(expected, actual))):
                raise ValueError("Aggregate metric mismatch")
    if not np.isclose(metrics["warm_coverage"], predictions.warm_target.mean()):
        raise ValueError("Warm coverage mismatch")
    if "common_warm" in metrics:
        users = json.loads((directory / "common_warm_users.json").read_text())
        subset = predictions.loc[predictions.user_id.isin(users)]
        if len(subset) != len(users) or metrics["common_warm"]["users"] != len(subset):
            raise ValueError("Common warm user coverage mismatch")
        for key in contributions(1):
            expected = float(subset[key].mean()) if len(subset) else None
            actual = metrics["common_warm"][key]
            if (expected is None and actual is not None) or (expected is not None and (actual is None or not np.isclose(expected, actual))):
                raise ValueError("Common warm metric mismatch")
    return status
