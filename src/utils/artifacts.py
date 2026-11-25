import hashlib
import json
import os
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def verify_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Dataset artifact hash mismatch: {name}")
    payload = {k: v for k, v in manifest.items() if k != "split_version"}
    if fingerprint(payload) != manifest["split_version"]:
        raise ValueError("Split manifest fingerprint mismatch")
    return manifest
