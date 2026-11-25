import numpy as np
import pandas as pd
import pytest
from src.data.preprocessing import preprocess, records
from src.data.splitting import leave_one_out, validate_splits, evaluation_cases
from src.data.sequence_dataset import SequenceDataset
from src.data.negative_sampling import sample_negatives
from src.data.sparsity import subsample_train


def test_deterministic_preprocessing(frame):
    raw = frame[["user", "item", "timestamp", "rating"]]
    first, users, items = preprocess(raw)
    second, users2, items2 = preprocess(raw.sample(frac=1, random_state=42))
    pd.testing.assert_frame_equal(first, second)
    assert users == users2 and items == items2


def test_repeated_interaction_earliest(frame):
    raw = frame[["user", "item", "timestamp", "rating"]]
    repeated = raw.iloc[[0]].assign(timestamp=100)
    actual, _, _ = preprocess(pd.concat([raw, repeated]))
    assert len(actual) == len(frame)
    assert actual.timestamp.max() < 100


def test_ties_sorted_by_asin(frame):
    raw = frame[["user", "item", "timestamp", "rating"]].assign(timestamp=10)
    actual, _, _ = preprocess(raw.sample(frac=1, random_state=7))
    for _, group in actual.groupby("user_id"):
        assert group.item.tolist() == sorted(group.item)


def test_iterative_core_cascade():
    # u0 has 3 items; all other users have 2. Removing them collapses u0's items.
    rows = [("a", i, n, 5) for n, i in enumerate("xyz")]
    rows += [(u, i, n, 5) for u in ("b", "c") for n, i in enumerate("xy")]
    with pytest.raises(ValueError, match="No interactions"):
        preprocess(pd.DataFrame(rows, columns=["user", "item", "timestamp", "rating"]), core=3)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 1.5])
def test_invalid_timestamp(frame, value):
    raw = frame[["user", "item", "timestamp", "rating"]].astype({"timestamp": float})
    raw.loc[0, "timestamp"] = value
    with pytest.raises(ValueError):
        preprocess(raw)


def test_split_and_training_target_separation(parts):
    validate_splits(parts)
    train, validation, test = (parts[key] for key in ("train", "validation", "test"))
    assert len(train) == 24 and len(validation) == len(test) == 6
    assert train.timestamp.max() < validation.timestamp.min() < test.timestamp.min()
    dataset = SequenceDataset(train, max_seq_length=5)
    assert len(dataset) == 18
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["target_event_id"] in set(train.event_id)
        assert sample["target_event_id"] not in set(validation.event_id) | set(test.event_id)
        user_targets = set(validation.loc[validation.user_id == sample["user_id"], "item_id"]) | set(test.loc[test.user_id == sample["user_id"], "item_id"])
        assert sample["target"] not in user_targets
        assert not set(sample["sequence"]) & user_targets
        assert sample["sequence"].shape == (5,)
        assert sample["sequence"][0] == 0


def test_leakage_rejected(parts):
    bad = {**parts, "train": pd.concat([parts["train"], parts["test"]])}
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(bad)


def test_chronology_rejected(parts):
    bad = {**parts, "validation": parts["test"], "test": parts["validation"]}
    with pytest.raises(ValueError, match="Chronological"):
        validate_splits(bad)


def test_evaluation_histories(parts):
    val, test = evaluation_cases(parts, "validation"), evaluation_cases(parts, "test")
    assert all(len(case["history"]) == 4 for case in val)
    assert all(len(case["history"]) == 5 for case in test)
    assert all(case["target"] not in case["history"] for case in test)


def test_negatives():
    actual = sample_negatives(5, {0, 2, 4}, 3, np.random.default_rng(7))
    assert set(actual) == {1, 3, 5}
    with pytest.raises(ValueError, match="Insufficient"):
        sample_negatives(2, {1, 2}, 1, np.random.default_rng(7))


def test_sparsity_fixed_targets(parts):
    original_targets = parts["test"].copy(deep=True)
    a, manifest_a = subsample_train(parts["train"], .4, 7)
    b, manifest_b = subsample_train(parts["train"], .4, 7)
    pd.testing.assert_frame_equal(a, b)
    assert manifest_a == manifest_b
    assert set(a.event_id) <= set(parts["train"].event_id)
    assert a.groupby("user_id").size().min() >= 1
    assert list(a.event_id) == sorted(a.event_id)
    pd.testing.assert_frame_equal(parts["test"], original_targets)


def test_safe_legacy_metadata(tmp_path):
    path = tmp_path / "meta.txt"
    path.write_text("{'asin': 'a', 'title': 'A'}\n")
    assert list(records(path))[0]["asin"] == "a"
    path.write_text("__import__('os').getcwd()\n")
    with pytest.raises(ValueError):
        list(records(path))
