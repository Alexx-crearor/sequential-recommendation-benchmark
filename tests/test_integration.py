import json
import sys
from pathlib import Path
import pandas as pd
import pytest
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from preprocess_data import prepare
from run_train import run
from src.utils.artifacts import verify_bundle, file_hash
from src.utils.runs import verify_run
from src.utils.grid import expand_grid


def make_bundle(tmp_path, frame):
    raw = tmp_path / "raw.json"
    lines = [{"reviewerID": row.user, "asin": row.item, "unixReviewTime": int(row.timestamp), "overall": int(row.rating)} for row in frame.itertuples()]
    raw.write_text("\n".join(json.dumps(row) for row in lines), encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text("\n".join(json.dumps({"asin": key, "title": key}) for key in frame.item.unique()), encoding="utf-8")
    return prepare("fixture_only", raw, metadata, tmp_path / "processed")


def test_end_to_end_artifacts_and_skip(tmp_path, frame):
    bundle = make_bundle(tmp_path, frame)
    verify_bundle(bundle)
    output = run(bundle, tmp_path / "runs")
    assert verify_run(output)["state"] == "SUCCESS"
    checksum, mtime = file_hash(output / "checkpoint.pt"), (output / "checkpoint.pt").stat().st_mtime_ns
    assert run(bundle, tmp_path / "runs") == output
    assert file_hash(output / "checkpoint.pt") == checksum
    assert (output / "checkpoint.pt").stat().st_mtime_ns == mtime
    (output / "predictions.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="altered"):
        run(bundle, tmp_path / "runs")


def test_dataset_tamper_rejected(tmp_path, frame):
    bundle = make_bundle(tmp_path, frame)
    (bundle / "train.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(bundle)


def test_incomplete_run_preserved(tmp_path, frame):
    bundle = make_bundle(tmp_path, frame)
    output = run(bundle, tmp_path / "runs")
    (output / "status.json").write_text(json.dumps({"state": "INCOMPLETE"}))
    with pytest.raises(ValueError, match="INCOMPLETE"):
        run(bundle, tmp_path / "runs")


def test_grid():
    root = Path(__file__).resolve().parents[1]
    grid = yaml.safe_load((root / "configs/experiments/experiment_grid.yaml").read_text())
    rows = expand_grid(grid)
    overall = [row for row in rows if row["experiment"] == "E1"]
    assert len(overall) == 57
    assert len([row for row in overall if row["model"] == "mostpopular"]) == 3
    assert all(row["estimated_seconds"] is None for row in overall)
    assert any(row.get("reuse_candidate") for row in rows if row["experiment"] == "E3")
