import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import pytest

from src.analysis.phase5c import extract_verified, validate_predictions, winner_index


def test_extraction_preserves_identical_files_and_preflights_all_conflicts(tmp_path):
    archive, output = tmp_path / "source.zip", tmp_path / "raw"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("first.json", "{}")
        stream.writestr("last.json", '{"a":1}')
    digest, inventory = extract_verified(archive, output, expected_sha256=None)
    assert len(inventory) == 2
    first_mtime = (output / "first.json").stat().st_mtime_ns
    extract_verified(archive, output, expected_sha256=digest)
    assert (output / "first.json").stat().st_mtime_ns == first_mtime
    (output / "first.json").unlink()
    (output / "last.json").write_text("preserve")
    with pytest.raises(ValueError, match="differs"):
        extract_verified(archive, output, expected_sha256=digest)
    assert not (output / "first.json").exists()
    assert (output / "last.json").read_text() == "preserve"


@pytest.mark.parametrize("member", ["../escape", "/absolute", "C:/escape", "folder\\escape"])
def test_archive_rejects_unsafe_paths_before_extraction(tmp_path, member):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("safe.json", "{}")
        info = zipfile.ZipInfo("placeholder")
        info.filename = member  # Retain malicious raw separators even on Windows writers.
        stream.writestr(info, "unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_verified(archive, tmp_path / "raw", expected_sha256=None)
    assert not (tmp_path / "raw/safe.json").exists()


def test_archive_identity_is_enforced(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("safe", "ok")
    with pytest.raises(ValueError, match="SHA256"):
        extract_verified(archive, tmp_path / "raw", expected_sha256="0"*64)


def prediction_fixture():
    gain = 1/np.log2(3)
    frame = pd.DataFrame({"user_id": [1, 2], "history": [[1], [2]], "target": [2, 4],
                          "top_k": [list(range(2, 11)), [3, 4, 1, 5, 6, 7, 8, 9, 10]],
                          "target_rank": [1, 2], "warm_target": [True, True], "mrr@10": [1., .5]})
    full = {"users": 2, "mrr@10": .75}
    for k in (5, 10, 20):
        frame[f"recall@{k}"] = [1., 1.]
        frame[f"ndcg@{k}"] = [1., gain]
        full[f"recall@{k}"] = 1.
        full[f"ndcg@{k}"] = (1+gain)/2
    return frame, {"full": full, "warm": dict(full), "warm_coverage": 1.}


def test_prediction_contributions_masks_and_population_signature():
    frame, metrics = prediction_fixture()
    signature = validate_predictions(frame, metrics, 10, 2)
    assert signature == validate_predictions(frame.iloc[::-1], metrics, 10, 2)
    frame.loc[0, "ndcg@10"] = .5
    with pytest.raises(ValueError, match="per-user"):
        validate_predictions(frame, metrics, 10, 2)


def test_prediction_top_k_cannot_contain_seen_items():
    frame, metrics = prediction_fixture()
    frame.at[0, "top_k"] = [1, *range(3, 11)]
    with pytest.raises(ValueError, match="history-masked"):
        validate_predictions(frame, metrics, 10, 2)


def test_selection_uses_validation_argmax_and_earliest_trial_tie():
    records = [{"trial": 2, "validation_ndcg@10": .4, "test_ndcg@10": .99},
               {"trial": 1, "validation_ndcg@10": .5, "test_ndcg@10": .01},
               {"trial": 0, "validation_ndcg@10": .5, "test_ndcg@10": .02}]
    assert winner_index(records) == 0
