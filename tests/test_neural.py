import copy
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from src.models.factory import build_model
from src.models.base import SequentialModel, pad_histories
from src.utils.configuration import resolve_config
from src.data.neural_dataset import TrainingDataset, verify_training_boundary
from src.training.checkpoint import seed_everything, rng_state, restore_rng
from src.training.losses import objective_loss
from src.evaluation.neural_ranking import evaluate_neural
from src.evaluation.ranking import rank_catalogue


NAMES = ["gru4rec_ce", "sasrec_bce", "sasrec_ce", "bert4rec", "attention_ablation", "fenrec"]


def tiny_config(name, **overrides):
    return resolve_config(name, {"hidden_dim": 8, "embedding_dim": 8, "ffn_dim": 32,
                        "num_layers": 1, "num_heads": 2, "max_seq_length": 5,
                        "batch_size": 4, "eval_batch_size": 3, "dropout": 0.,
                        "attention_dropout": 0., "intent_num": 2, "max_epochs": 2,
                        **overrides})


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize("name", NAMES)
def test_models_finite_losses_gradients_and_padding(name, parts):
    seed_everything(7)
    config = tiny_config(name)
    dataset = TrainingDataset(parts["train"], 6, config)
    verify_training_boundary(dataset, parts)
    model = build_model(name, 6, config)
    if name == "fenrec":
        model.set_clusters(torch.randn(2, 8))
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    loss, components = objective_loss(model, batch, 21)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert torch.equal(model.embedding.weight.grad[0], torch.zeros(8))
    model.eval()
    tokens = torch.tensor([[0, 0, 1, 2, 3], [0, 0, 0, 0, 0]])
    output = model(tokens)
    assert torch.isfinite(output).all()
    assert (output[tokens == 0] == 0).all()


@pytest.mark.parametrize("name", ["sasrec_bce", "sasrec_ce", "attention_ablation", "fenrec", "gru4rec_ce"])
def test_future_tokens_cannot_change_past(name):
    seed_everything(42)
    config = tiny_config(name)
    model = build_model(name, 10, config).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    changed = tokens.clone()
    changed[0, -1] = 9
    torch.testing.assert_close(model(tokens)[:, :-1], model(changed)[:, :-1], rtol=0, atol=0)


def test_bidirectional_sees_right_context():
    model = build_model("bert4rec", 10, tiny_config("bert4rec")).eval()
    first, second = torch.tensor([[1, 2, 11, 4, 5]]), torch.tensor([[1, 2, 11, 4, 9]])
    assert not torch.equal(model(first)[:, 2], model(second)[:, 2])


@pytest.mark.parametrize("name", NAMES)
def test_pad_embedding_has_no_effect_on_real_states(name):
    model = build_model(name, 10, tiny_config(name)).eval()
    tokens = torch.tensor([[0, 0, 1, 2, 3]])
    expected = model(tokens).detach().clone()
    with torch.no_grad():
        model.embedding.weight[0].fill_(100.)
    torch.testing.assert_close(model(tokens), expected)


def test_bert_mask_training_and_inference(parts):
    config = tiny_config("bert4rec", mask_ratio=.5, last_mask_probability=0.)
    dataset = TrainingDataset(parts["train"], 6, config)
    sample = dataset[0]
    selected = sample["targets"].ne(0)
    assert selected.sum() == 2
    assert sample["tokens"][selected].eq(7).all()
    assert sample["targets"][sample["tokens"].eq(0)].eq(0).all()
    tokens = pad_histories([[1, 2, 3]], 5, "cpu", 7)
    assert tokens.tolist() == [[0, 1, 2, 3, 7]]
    model = build_model("bert4rec", 6, config).eval()
    torch.testing.assert_close(model.representations([[1, 2, 3]]), model(tokens)[:, -1])


def test_shifted_targets_and_negative_sampling(parts):
    config = tiny_config("sasrec_bce")
    dataset = TrainingDataset(parts["train"], 6, config)
    for index in range(len(dataset)):
        item = dataset[index]
        known = set(dataset.sequences[index][1])
        assert not set(item["negatives"].tolist()) & known
        real_x, real_y = item["tokens"][item["tokens"] != 0], item["targets"][item["targets"] != 0]
        assert torch.equal(real_x[1:], real_y[:-1])
        assert (item["negatives"][item["targets"].eq(0)] == 0).all()


@pytest.mark.parametrize("name", NAMES)
def test_checkpoint_and_batched_ranking(name, parts, tmp_path):
    config = tiny_config(name)
    model = build_model(name, 6, config).eval()
    path = tmp_path / "checkpoint.pt"
    model.save(path)
    restored = SequentialModel.load(path)
    np.testing.assert_array_equal(model.score_items([1, 2], np.arange(1, 7)), restored.score_items([1, 2], np.arange(1, 7)))
    metrics, predictions = evaluate_neural(model, parts, "test", 2)
    assert len(predictions) == len(parts["test"])
    for row in predictions.itertuples():
        top, rank = rank_catalogue(model, row.history, row.target, 6, item_batch_size=2)
        assert row.top_k == top
        assert row.target_rank == rank


def test_rng_restoration():
    import random
    seed_everything(7)
    state = rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(2))
    restore_rng(state)
    actual = (random.random(), np.random.rand(), torch.rand(2))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])


@pytest.mark.parametrize("name", NAMES)
def test_neural_ranking_with_multiple_eligible_items(name):
    import pandas as pd
    model = build_model(name, 10, tiny_config(name)).eval()
    parts = {"train": pd.DataFrame({"user_id": [1, 1], "item_id": [1, 2], "timestamp": [1, 2]}),
             "validation": pd.DataFrame({"user_id": [1], "item_id": [3], "timestamp": [3]}),
             "test": pd.DataFrame({"user_id": [1], "item_id": [8], "timestamp": [4]})}
    metrics, rows = evaluate_neural(model, parts, "test", 3)
    expected, rank = rank_catalogue(model, [1, 2, 3], 8, 10, item_batch_size=3)
    assert rows.iloc[0].top_k == expected
    assert rows.iloc[0].target_rank == rank
    assert len(expected) == 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.cuda
@pytest.mark.parametrize("name", NAMES)
def test_cuda_forward_backward_and_amp(name, parts):
    config = tiny_config(name)
    model = build_model(name, 6, config).cuda()
    if name == "fenrec":
        model.set_clusters(torch.randn(2, 8, device="cuda"))
    batch = {key: value.cuda() for key, value in next(iter(DataLoader(TrainingDataset(parts["train"], 6, config), batch_size=4))).items()}
    with torch.autocast("cuda", enabled=name != "fenrec"):
        loss, _ = objective_loss(model, batch, 21)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
