"""Publication guards inspect staged blobs, including forced ignored artifacts."""
import importlib.util
from pathlib import Path
import subprocess


def load_auditor():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_publication.py"
    spec = importlib.util.spec_from_file_location("publication_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "-f", name], check=True, capture_output=True)


def test_audit_reads_index_instead_of_unstaged_content(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    stage(tmp_path, "safe.py", "answer = 42\n")
    (tmp_path / "safe.py").write_text("password = '" + "x" * 24 + "'\n")
    assert not load_auditor().audit(tmp_path)["findings"]


def test_audit_rejects_secrets_datasets_and_large_files_without_exposing_values(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    token = "gh" + "p_" + "a" * 36
    stage(tmp_path, "config.py", "TOKEN = '" + token + "'\n")
    stage(tmp_path, "data/raw/events.csv", "user,item\n")
    stage(tmp_path, "large.txt", "a" * 256)
    report = load_auditor().audit(tmp_path, max_bytes=128)
    reasons = {item["reason"] for item in report["findings"]}
    assert "GitHub token" in reasons
    assert "file exceeds publication size limit" in reasons
    assert any("dataset" in reason for reason in reasons)
    assert token not in str(report)
