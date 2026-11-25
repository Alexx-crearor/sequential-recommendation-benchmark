import json
from pathlib import Path
import sys
import zipfile
import pytest
import torch
import pandas as pd
from test_neural import tiny_config
from test_integration import make_bundle
from src.training.checkpoint import seed_everything
from src.training.trainer import training_epoch, run_neural
from src.models.factory import build_model
from src.data.neural_dataset import TrainingDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from kaggle_session import restore, package
from run_tuning import candidates
from src.utils.profiles import load_profile


def test_result_restore_is_idempotent_preserves_artifacts_and_rejects_traversal(tmp_path):
    archive = tmp_path / "results.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("results/run/checkpoint.pt", b"checkpoint")
    destination = tmp_path / "working"
    restore(archive, destination)
    restore(archive, destination)
    target = destination / "results/run/checkpoint.pt"
    target.write_bytes(b"different_completed_artifact")
    with pytest.raises(ValueError, match="differs"):
        restore(archive, destination)
    assert target.read_bytes() == b"different_completed_artifact"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape", b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        restore(archive, destination)


def test_bounded_tuning_candidates_are_seeded_and_capped():
    _, profile = load_profile("core_research")
    first = candidates(profile, "fenrec", 6)
    assert len(first) == 6 and first == candidates(profile, "fenrec", 6)
    assert len({json.dumps(row, sort_keys=True) for row in first}) == 6
    assert any("soft_label_gamma" in row for row in first)
    with pytest.raises(ValueError, match="budget"):
        candidates(profile, "fenrec", 13)


def test_worker_rng_training_is_reproducible(parts):
    config = tiny_config("bert4rec", num_workers=2, max_train_batches=2, pilot=True, cpu_threads=1)
    dataset = TrainingDataset(parts["train"], 6, config)
    a = build_model("bert4rec", 6, config)
    b = build_model("bert4rec", 6, config)
    b.load_state_dict(a.state_dict())
    losses = []
    for model in (a, b):
        seed_everything(7)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        losses.append(training_epoch(model, dataset, optimizer, torch.amp.GradScaler("cuda", enabled=False), config, 0, "cpu")["train_loss"])
    assert losses[0] == losses[1]
    for key in a.state_dict():
        torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)


def test_fenrec_patience_does_not_stop_before_postwarmup_budget(tmp_path, frame, monkeypatch):
    import src.training.trainer as trainer
    original = trainer.evaluate_neural
    def constant_validation(model, parts, split="test", *args):
        metrics, predictions = original(model, parts, split, *args)
        if split == "validation": metrics["full"]["ndcg@10"] = .1
        return metrics, predictions
    monkeypatch.setattr(trainer, "evaluate_neural", constant_validation)
    config = tiny_config("fenrec", warm_up_epochs=2, min_post_warmup_epochs=2, max_epochs=6,
                         early_stopping_patience=1, cpu_threads=1, cluster_iterations=2, cluster_restarts=1)
    output = run_neural(make_bundle(tmp_path, frame), config, tmp_path/"runs")
    history = pd.read_csv(output/"train_history.csv")
    assert len(history) >= 4


def test_source_package_excludes_results_raw_and_bytecode(tmp_path):
    archive = package(tmp_path / "source.zip")
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
    assert "src/training/trainer.py" in names
    assert "notebooks/Kaggle_Training.ipynb" in names
    assert "configs/experiments/profile_core_research.yaml" in names
    assert not any(name.startswith("results/") or name.startswith("data/raw/") or "__pycache__" in name for name in names)


def test_kaggle_notebook_cells_compile_and_call_repository_scripts():
    import ast
    root = Path(__file__).resolve().parents[1]
    for name in ["Kaggle_Setup.ipynb", "Kaggle_Training.ipynb", "Kaggle_Analysis.ipynb"]:
        notebook = json.loads((root/"notebooks"/name).read_text())
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
    text = (root/"notebooks/Kaggle_Training.ipynb").read_text()
    assert "scripts/run_experiment.py" in text and "scripts/run_tuning.py" in text


def test_e7_e8_reuse_verified_e1_predictions(tmp_path, frame):
    from run_analysis import run_analysis
    bundle = make_bundle(tmp_path, frame)
    for name in ("sasrec_ce", "bert4rec"):
        run_neural(bundle, tiny_config(name, max_epochs=1, cpu_threads=1), tmp_path/"runs", "E1")
    efficiency = run_analysis("E7", bundle, tmp_path/"runs", tmp_path/"analysis")
    errors = run_analysis("E8", bundle, tmp_path/"runs", tmp_path/"analysis")
    assert (efficiency/"table_08_efficiency.csv").exists()
    assert (errors/"table_12_error_cases.csv").exists()
    assert len(json.loads((errors/"error_cases.json").read_text())) == 6


def test_aggregation_does_not_mix_sequence_length_or_experiment(tmp_path, frame, monkeypatch):
    import aggregate_results
    bundle = make_bundle(tmp_path, frame)
    for length in (4, 5):
        config = tiny_config("sasrec_ce", max_epochs=1, max_seq_length=length, cpu_threads=1)
        run_neural(bundle, config, tmp_path/"runs", "E5")
    monkeypatch.setattr(sys, "argv", ["aggregate_results.py", "--root", str(tmp_path/"runs"), "--output", str(tmp_path/"tables")])
    aggregate_results.main()
    rows = pd.read_csv(tmp_path/"tables/per_seed_results.csv")
    assert rows.comparison_hash.nunique() == 2
    assert set(rows.max_seq_length) == {4, 5}
    assert (tmp_path/"tables/table_06_sequence_length.csv").exists()
    assert not (tmp_path/"tables/table_02_overall_performance.csv").exists()
