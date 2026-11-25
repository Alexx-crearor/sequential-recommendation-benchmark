import math
import numpy as np
import pytest
from src.models.popularity import MostPopular
from src.evaluation.ranking import rank_catalogue, deduplicate
from src.evaluation.metrics import contributions
from src.evaluation.evaluate import evaluate
from src.evaluation.bootstrap import paired_bootstrap


def test_metrics_manual():
    assert contributions(1)["ndcg@5"] == 1
    assert contributions(2)["ndcg@5"] == pytest.approx(1 / math.log2(3))
    assert contributions(6)["recall@5"] == 0
    assert contributions(6)["recall@10"] == 1
    assert contributions(10)["mrr@10"] == .1
    assert contributions(11)["mrr@10"] == 0
    assert contributions(None)["ndcg@20"] == 0
    with pytest.raises(ValueError):
        contributions(0)


def test_popularity_train_only(parts):
    model = MostPopular(6).fit(parts["train"])
    assert model.counts.sum() == len(parts["train"])
    np.testing.assert_array_equal(model.counts, np.bincount(parts["train"].item_id, minlength=7))


@pytest.mark.parametrize("batch", [1, 2, 100])
def test_target_exemption_ties_and_padding(batch):
    model = MostPopular(6)
    top, rank = rank_catalogue(model, [1, 2, 3], 2, 6, item_batch_size=batch)
    assert top == [2, 4, 5, 6] and rank == 1
    assert 0 not in top


def test_batched_ranking_matches_bruteforce():
    model = MostPopular(100)
    model.counts = np.random.default_rng(7).integers(0, 10, 101)
    history, target = [2, 4, 6, 8], 33
    eligible = [i for i in range(1, 101) if i not in history]
    expected = sorted(eligible, key=lambda i: (-model.counts[i], i))
    top, rank = rank_catalogue(model, history, target, 100, item_batch_size=7)
    assert top == expected[:20]
    assert rank == expected.index(target) + 1


def test_nan_rejected():
    class Bad:
        def score_items(self, history, ids):
            return np.full(len(ids), np.nan)
    with pytest.raises(ValueError, match="Nonfinite"):
        rank_catalogue(Bad(), [], 1, 2)


def test_cold_target_coverage(parts):
    # All items are warm in the dense fixture; zero out target's training occurrences.
    cold = int(parts["test"].iloc[0].item_id)
    modified = {**parts, "train": parts["train"].loc[parts["train"].item_id != cold]}
    metrics, predictions = evaluate(MostPopular(6).fit(modified["train"]), modified)
    assert len(predictions) == len(parts["test"])
    assert metrics["warm_coverage"] < 1
    assert not predictions.loc[predictions.target == cold, "warm_target"].any()


def test_checkpoint_equivalence(parts, tmp_path):
    model = MostPopular(6).fit(parts["train"])
    path = tmp_path / "checkpoint.pt"
    model.save(path)
    loaded = MostPopular.load(path)
    np.testing.assert_array_equal(model.counts, loaded.counts)
    assert rank_catalogue(model, [1], 2, 6) == rank_catalogue(loaded, [1], 2, 6)


def test_deduplication():
    assert deduplicate([2, 2, 99, 1, 0, 1], {1, 2}) == [2, 1]


def test_paired_bootstrap():
    result = paired_bootstrap([1, 1, 1], [0, 0, 0], 100)
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 1
    assert result == paired_bootstrap([1, 1, 1], [0, 0, 0], 100)
