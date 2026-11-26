"""Portable paths and checkpoint installation without experimental training."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import pytest

from src.analysis.phase5c import ARCHIVE_NAME, resolve_archive_path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_phase5c_checkpoints import fetch


def test_archive_default_is_project_relative_not_downloads(tmp_path, monkeypatch):
    monkeypatch.delenv("SEQREC_PHASE5C_ARCHIVE", raising=False)
    assert resolve_archive_path(tmp_path) == tmp_path / "artifacts/phase5c" / ARCHIVE_NAME
    monkeypatch.setenv("SEQREC_PHASE5C_ARCHIVE", "private/original.zip")
    assert resolve_archive_path(tmp_path) == tmp_path / "private/original.zip"
    assert resolve_archive_path(tmp_path, "explicit.zip") == tmp_path / "explicit.zip"


def fixture_bundle(tmp_path):
    root = tmp_path / "clone"
    relative = "checkpoints/phase5c/beauty/model/trials/0/checkpoint.pt"
    content = b"opaque checkpoint bytes for installer test"
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(relative, content)
    bundle = {"filename": "bundle.zip", "url": "https://invalid.example/bundle.zip",
              "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "paths": [relative]}
    manifest = {"files": [{"path": relative, "sha256": hashlib.sha256(content).hexdigest()}],
                "bundles": {"selected": bundle, "all": bundle}, "models": {"model": {"checkpoint": relative}}}
    directory = root / "configs/artifacts"
    directory.mkdir(parents=True)
    (directory / "phase5c_checkpoints.json").write_text(json.dumps(manifest))
    return root, archive, relative, content


def test_install_offline_and_skip_verified_cache(tmp_path, monkeypatch):
    root, archive, relative, content = fixture_bundle(tmp_path)
    fetch(root, archive=archive)
    assert (root / relative).read_bytes() == content
    before = (root / relative).stat().st_mtime_ns
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("Unexpected network access"))
    fetch(root)
    assert (root / relative).stat().st_mtime_ns == before


def test_download_hash_failure_does_not_extract(tmp_path):
    root, archive, relative, _ = fixture_bundle(tmp_path)
    archive.write_bytes(b"corrupted zip")
    with pytest.raises(ValueError, match="SHA256"):
        fetch(root, archive=archive)
    assert not (root / relative).exists()


def test_conflicting_checkpoint_is_preserved(tmp_path):
    root, archive, relative, _ = fixture_bundle(tmp_path)
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"preserve this valid external checkpoint")
    with pytest.raises(ValueError, match="differs"):
        fetch(root, all_trials=True, archive=archive)
    assert path.read_bytes() == b"preserve this valid external checkpoint"
