import json
import torch
from test_integration import make_bundle
from test_neural import tiny_config
from src.training import trainer


def test_early_stop_selects_best_checkpoint(tmp_path, frame, monkeypatch):
    bundle = make_bundle(tmp_path, frame)
    original = trainer.evaluate_neural
    calls = []
    def evaluate(model, parts, split="test", *args):
        metrics, predictions = original(model, parts, split, *args)
        if split == "validation":
            metrics["full"]["ndcg@10"] = .5 if not calls else .4
            calls.append(True)
        return metrics, predictions
    monkeypatch.setattr(trainer, "evaluate_neural", evaluate)
    config = tiny_config("sasrec_ce", max_epochs=5, early_stopping_patience=1, cpu_threads=1)
    output = trainer.run_neural(bundle, config, tmp_path / "runs")
    efficiency = json.loads((output / "efficiency.json").read_text())
    assert efficiency["epochs_completed"] == 2
    assert efficiency["converged"] and not efficiency["training_limit_reached"]
    best = torch.load(output / "checkpoint.pt", weights_only=True)
    latest = torch.load(output / "latest.pt", weights_only=True)
    assert latest["best"] == .5
    assert any(not torch.equal(best["model"][key], latest["model"][key]) for key in best["model"])
