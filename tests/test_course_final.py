import hashlib
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest

from src.analysis.course_final import import_delta, closed_jobs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archives(tmp_path, changed=False):
    name = "results/core/E1/toy/hash/7/checkpoint.pt"
    old, new, old_raw = tmp_path / "old.zip", tmp_path / "new.zip", tmp_path / "prior_raw"
    old_path = old_raw / name
    old_path.parent.mkdir(parents=True)
    old_path.write_bytes(b"original weights")
    with zipfile.ZipFile(old, "w") as stream:
        stream.writestr(name, b"original weights")
    with zipfile.ZipFile(new, "w") as stream:
        stream.writestr(name, b"changed weights" if changed else b"original weights")
        stream.writestr("results/new/evidence.json", b"{}")
    return old, new, old_raw, [{"member": name, "sha256": sha(old_path)}], name


def test_delta_reuses_immutable_prior_bytes_and_is_idempotent(tmp_path):
    old, new, source, inventory, name = archives(tmp_path)
    target = tmp_path / "new_raw"
    files, checks = import_delta(new, target, old, source, inventory, sha(new), sha(old))
    assert len(checks) == 1 and checks[0]["unchanged"]
    assert next(f for f in files if f["member"] == name)["deduplicated_prior_content"]
    assert (target / name).read_bytes() == (source / name).read_bytes()
    timestamp = (target / name).stat().st_mtime_ns
    import_delta(new, target, old, source, inventory, sha(new), sha(old))
    assert (target / name).stat().st_mtime_ns == timestamp


def test_changed_prior_run_rejected_before_any_extraction(tmp_path):
    old, new, source, inventory, _ = archives(tmp_path, changed=True)
    destination = tmp_path / "new_raw"
    with pytest.raises(ValueError, match="changed"):
        import_delta(new, destination, old, source, inventory, sha(new), sha(old))
    assert not destination.exists()


def test_closed_scope_never_promotes_unrun_experiments():
    jobs = [{"job": i, "path": name, "experiment": exp} for i, (name, exp) in enumerate([("first", "E1"), ("second", "E2"), ("third", "E3")])]
    result = closed_jobs(jobs, {"first", "second"})
    assert len(result) == 2 and all(j["evidence_state"] == "ARCHIVED_SUCCESS" for j in result)
    with pytest.raises(ValueError, match="incomplete"):
        closed_jobs(jobs, {"first"})


def test_closed_driver_rejects_run_even_in_kaggle_environment():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/run_course_remaining.py"), "--run"],
                            env={**os.environ, "KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, capture_output=True, text=True)
    assert result.returncode == 2 and "CLOSED" in result.stderr
