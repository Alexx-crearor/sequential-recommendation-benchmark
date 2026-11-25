import json
from pathlib import Path
import pytest
import torch
from test_integration import make_bundle
from test_neural import tiny_config
from src.training.trainer import run_neural
from src.utils.runs import verify_run
from src.utils.artifacts import file_hash


@pytest.mark.parametrize("name", ["gru4rec_ce", "sasrec_bce", "sasrec_ce", "bert4rec", "attention_ablation", "fenrec"])
def test_real_training_integration_resume_exact(name, tmp_path, frame):
    bundle = make_bundle(tmp_path, frame)
    config = tiny_config(name, dropout=.2, attention_dropout=.2, intent_num=2, warm_up_epochs=0,
                         max_epochs=2, pilot=True, max_train_batches=2, validation_user_limit=3,
                         cluster_restarts=1, cluster_iterations=2, cpu_threads=1)
    uninterrupted = run_neural(bundle, config, tmp_path / "continuous")
    interrupted = run_neural(bundle, config, tmp_path / "resumed", stop_after_epochs=1)
    assert json.loads((interrupted / "status.json").read_text())["state"] == "INCOMPLETE"
    with pytest.raises(ValueError, match="Existing INCOMPLETE"):
        run_neural(bundle, config, tmp_path / "resumed")
    resumed = run_neural(bundle, config, tmp_path / "resumed", resume=True)
    assert resumed == interrupted
    verify_run(resumed)
    a = torch.load(uninterrupted / "latest.pt", weights_only=True)
    b = torch.load(resumed / "latest.pt", weights_only=True)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert [row["train_loss"] for row in a["history"]] == [row["train_loss"] for row in b["history"]]
    assert [row["validation_ndcg@10"] for row in a["history"]] == [row["validation_ndcg@10"] for row in b["history"]]
    checksum = file_hash(resumed / "checkpoint.pt")
    assert run_neural(bundle, config, tmp_path / "resumed") == resumed
    assert file_hash(resumed / "checkpoint.pt") == checksum


def test_matching_e2_capacity():
    from src.models.factory import build_model
    counts = []
    for objective, attention in [("next_bce", "causal"), ("next_ce", "causal"), ("masked_ce", "causal"), ("masked_ce", "bidirectional")]:
        config = tiny_config("attention_ablation", objective=objective, attention=attention)
        counts.append(sum(p.numel() for p in build_model("attention_ablation", 6, config).parameters()))
    assert len(set(counts)) == 1


def test_bad_pilot_controls_rejected():
    with pytest.raises(ValueError, match="requires --pilot"):
        tiny_config("sasrec_ce", max_train_batches=1)
    with pytest.raises(ValueError, match="FP32"):
        tiny_config("fenrec", amp=True)
