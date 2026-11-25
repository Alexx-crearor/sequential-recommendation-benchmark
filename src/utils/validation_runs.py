"""Validation-only artifacts have a separate state and never require test labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.utils.artifacts import file_hash, fingerprint


def verify_validation_run(directory):
    directory = Path(directory)
    status = json.loads((directory / "status.json").read_text())
    if status["state"] != "VALIDATED":
        raise ValueError("Not a completed validation-only run")
    config = yaml.safe_load((directory / "config.yaml").read_text())
    if config.get("evaluate_test", True) or fingerprint(config) != status["config_hash"]:
        raise ValueError("Invalid validation-only configuration")
    for name, digest in status["files"].items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"Altered validation artifact: {name}")
    if (directory / "test_metrics.json").exists() or (directory / "predictions.parquet").exists():
        raise ValueError("Tuning run contains test artifacts")
    from src.models.base import SequentialModel
    model = SequentialModel.load(directory / "checkpoint.pt")
    if model.config != config:
        raise ValueError("Validation checkpoint configuration mismatch")
    import torch
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Nonfinite validation checkpoint")
    predictions = pd.read_parquet(directory / "validation_predictions.parquet")
    metrics = json.loads((directory / "validation_selection.json").read_text())
    if len(predictions) != metrics["full"]["users"] or predictions.user_id.duplicated().any():
        raise ValueError("Validation user coverage mismatch")
    for key in ("recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20", "mrr@10"):
        if not np.isfinite(predictions[key]).all() or not np.isclose(predictions[key].mean(), metrics["full"][key]):
            raise ValueError("Validation metric mismatch")
    from src.evaluation.metrics import contributions
    for _, row in predictions.iterrows():
        if not 1 <= row.target_rank <= model.num_items:
            raise ValueError("Invalid validation target rank")
        for key, value in contributions(int(row.target_rank)).items():
            if not np.isclose(row[key], value):
                raise ValueError("Incorrect validation contribution")
    return status
