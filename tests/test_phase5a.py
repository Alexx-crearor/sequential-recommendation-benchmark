import copy
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import pytest
import torch
from src.utils.configuration import resolve_config
from src.utils.profiles import load_profile
from src.utils.grid import expand_grid
from src.data.neural_dataset import TrainingDataset
from benchmarks.phase34.dataset import TrainingDataset as OriginalDataset
from benchmarks.phase34.ranking import evaluate_neural as original_ranking
from benchmarks.phase34.gru4rec import GRU4RecCE as OriginalGRU
from src.evaluation.neural_ranking import evaluate_neural
from src.models.factory import build_model
from src.training.checkpoint import seed_everything, rng_state, restore_rng
from src.training.trainer import training_epoch, run_neural
from src.training.losses import objective_loss
from src.utils.validation_runs import verify_validation_run
from test_neural import tiny_config, NAMES
from test_integration import make_bundle


@pytest.mark.parametrize("name", NAMES)
def test_cached_preprocessing_preserves_samples_and_random_transforms(name, parts):
    config = tiny_config(name)
    original = OriginalDataset(parts["train"], 6, config)
    cached = TrainingDataset(parts["train"], 6, config)
    seed_everything(7)
    for index in range(len(original)):
        state = rng_state()
        expected = original[index]
        restore_rng(state)
        actual = cached[index]
        assert actual.keys() == expected.keys()
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("chunk", [1, 7, 67])
def test_streamed_ranking_ties_masks_cold_targets_and_short_catalogue(device, chunk):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    class StaticModel:
        def __init__(self):
            self.device, self.num_items = torch.device(device), 31
            self.config = {"eval_batch_size": 3, "ranking_backend": "torch"}
            # Deliberate ties, including zero-frequency/cold-like scores.
            self.scores = torch.tensor(np.random.default_rng(7).integers(0, 4, (6, 32)), device=device, dtype=torch.float32)
        def eval(self): return self
        def representations(self, histories):
            return torch.tensor([history[0] for history in histories], device=device)[:, None]
        def logits(self, states, ids):
            return self.scores[states[:, 0].long()][:, ids]
    model = StaticModel()
    train = pd.DataFrame({"user_id": np.repeat(np.arange(1, 6), 2), "item_id": [i for u in range(1, 6) for i in [u, u+6]], "timestamp": [1, 2]*5})
    parts = {"train": train, "validation": pd.DataFrame({"user_id": range(1, 6), "item_id": [30, 2, 31, 4, 28], "timestamp": [3]*5}),
             "test": pd.DataFrame({"user_id": range(1, 6), "item_id": [29, 30, 31, 28, 27], "timestamp": [4]*5})}
    for split in ("validation", "test"):
        expected_metrics, expected = original_ranking(model, parts, split, chunk)
        metrics, actual = evaluate_neural(model, parts, split, chunk)
        assert metrics == expected_metrics
        assert actual.target_rank.tolist() == expected.target_rank.tolist()
        assert actual.top_k.tolist() == expected.top_k.tolist()


def test_vectorized_gru_matches_original_states_and_gradients():
    config = tiny_config("gru4rec_ce")
    original = OriginalGRU(10, config)
    optimized = build_model("gru4rec_ce", 10, config)
    optimized.load_state_dict(original.state_dict())
    tokens = torch.tensor([[0, 0, 1, 2, 3], [1, 0, 2, 0, 3], [0, 0, 0, 0, 0]])
    expected, actual = original(tokens), optimized(tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.sum().backward(); actual.sum().backward()
    for left, right in zip(original.parameters(), optimized.parameters()):
        torch.testing.assert_close(left.grad, right.grad)


@pytest.mark.parametrize("name", ["gru4rec_ce", "sasrec_ce", "sasrec_bce", "bert4rec"])
def test_accumulation_matches_large_batch_update(name, parts):
    config = tiny_config(name, batch_size=4, gradient_clip=1e9, cpu_threads=1)
    # Keep corruption/negative draws out of the comparison: freeze objective batches.
    seed_everything(7)
    dataset = TrainingDataset(parts["train"], 6, config)
    frozen = [dataset[index] for index in range(len(dataset))]
    a = build_model(name, 6, config)
    smaller = {**config, "batch_size": 2, "gradient_accumulation_steps": 2}
    b = build_model(name, 6, smaller)
    b.load_state_dict(a.state_dict())
    for model, cfg in [(a, config), (b, smaller)]:
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        seed_everything(42)
        training_epoch(model, frozen, optimizer, torch.amp.GradScaler("cuda", enabled=False), cfg, 0, "cpu")
    for key in a.state_dict():
        torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("name", ["sasrec_ce", "bert4rec"])
def test_checkpointed_catalogue_ce_preserves_loss_and_gradients(name, parts):
    config = tiny_config(name, loss_position_chunk=2)
    batch = next(iter(torch.utils.data.DataLoader(TrainingDataset(parts["train"], 6, config), batch_size=4)))
    a = build_model(name, 6, config)
    b = build_model(name, 6, {**config, "activation_checkpoint_ce": True})
    b.load_state_dict(a.state_dict())
    left, _ = objective_loss(a, batch, 0)
    right, _ = objective_loss(b, batch, 0)
    torch.testing.assert_close(left, right)
    left.backward(); right.backward()
    for first, second in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(first.grad, second.grad)


def test_fenrec_accumulation_rejected():
    with pytest.raises(ValueError, match="change the objective"):
        tiny_config("fenrec", gradient_accumulation_steps=2)


def test_profiles_preserve_full_grid_and_core_axes():
    full, _ = load_profile("full_research")
    core, _ = load_profile("core_research")
    final, _ = load_profile("core_research", True)
    assert len(expand_grid(full)) == 209
    assert len(expand_grid(core)) == 45
    assert len(expand_grid(final)) == 129
    assert core["experiments"]["E3"]["retention"] == [.4, .7, 1.]
    assert core["experiments"]["E5"]["max_seq_length"] == [10, 20, 50]
    assert full["experiments"]["E1"]["datasets"] == ["beauty", "sports", "toys"]
    assert "tiger" in full["experiments"]["E1"]["models"]


def test_validation_only_training_never_evaluates_test(tmp_path, frame, monkeypatch):
    from src.training import trainer
    bundle = make_bundle(tmp_path, frame)
    original = trainer.evaluate_neural
    def guard(model, parts, split="test", *args):
        assert split == "validation", "Test evaluation entered during tuning"
        return original(model, parts, split, *args)
    monkeypatch.setattr(trainer, "evaluate_neural", guard)
    config = tiny_config("sasrec_ce", evaluate_test=False, max_epochs=3, validation_every=2, checkpoint_every=2, cpu_threads=1)
    output = run_neural(bundle, config, tmp_path / "tuning")
    assert verify_validation_run(output)["state"] == "VALIDATED"
    assert not (output / "test_metrics.json").exists()
    history = pd.read_csv(output / "train_history.csv")
    assert history.validation_performed.tolist() == [True, True, True]


def test_sparse_validation_resume_and_checkpoint_cadence(tmp_path, frame):
    bundle = make_bundle(tmp_path, frame)
    config = tiny_config("sasrec_ce", max_epochs=4, validation_every=3, checkpoint_every=3, cpu_threads=1)
    a = run_neural(bundle, config, tmp_path / "continuous")
    b = run_neural(bundle, config, tmp_path / "interrupted", stop_after_epochs=2)
    b = run_neural(bundle, config, tmp_path / "interrupted", resume=True)
    assert pd.read_csv(a / "train_history.csv").validation_performed.tolist() == [True, False, True, True]
    left = torch.load(a / "latest.pt", weights_only=True)
    right = torch.load(b / "latest.pt", weights_only=True)
    for key in left["model"]:
        torch.testing.assert_close(left["model"][key], right["model"][key], rtol=0, atol=0)
