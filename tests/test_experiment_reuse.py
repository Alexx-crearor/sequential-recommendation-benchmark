import json
from test_integration import make_bundle
from test_neural import tiny_config
from src.training.trainer import run_neural
from src.utils.runs import verify_run
from src.utils.artifacts import file_hash


def test_exact_e1_reuse_and_common_warm_analysis(tmp_path, frame):
    bundle = make_bundle(tmp_path, frame)
    config = tiny_config("sasrec_ce", max_epochs=1, cpu_threads=1)
    original = run_neural(bundle, config, tmp_path / "runs", "E1")
    checkpoint = file_hash(original / "checkpoint.pt")
    reused = run_neural(bundle, config, tmp_path / "runs", "E3")
    assert (reused / "reuse.json").exists()
    assert file_hash(reused / "checkpoint.pt") == checkpoint
    assert file_hash(original / "checkpoint.pt") == checkpoint
    assert "common_warm" in json.loads((reused / "test_metrics.json").read_text())
    assert "common_warm" not in json.loads((original / "test_metrics.json").read_text())
    verify_run(reused)
