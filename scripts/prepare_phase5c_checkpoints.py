"""Materialize original checkpoint bytes and build prediction-free Release bundles."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import yaml
import _bootstrap
from src.analysis.phase5c import ARCHIVE_SHA256, resolve_archive_path
from src.utils.artifacts import file_hash, write_json

RELEASE_TAG = "phase5c-checkpoints-v1"
RELEASE_BASE = f"https://github.com/Alexx-crearor/sequential-recommendation-benchmark/releases/download/{RELEASE_TAG}"


def build_bundle(root, files, output):
    expected = {f["path"]: f["sha256"] for f in files}
    if output.exists():
        with zipfile.ZipFile(output) as stream:
            if set(stream.namelist()) != set(expected) or stream.testzip() is not None:
                raise ValueError(f"Existing bundle differs: {output}")
            for name, digest in expected.items():
                if hashlib.sha256(stream.read(name)).hexdigest() != digest:
                    raise ValueError(f"Existing bundle content differs: {name}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for item in sorted(files, key=lambda f: f["path"]):
            info = zipfile.ZipInfo(item["path"], date_time=(2026, 11, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with (root / item["path"]).open("rb") as source, stream.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target)


def prepare(root, archive):
    root, archive = Path(root).resolve(), Path(archive)
    if file_hash(archive) != ARCHIVE_SHA256:
        raise ValueError("Original archive SHA256 differs")
    planned, models = [], {}
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None:
            raise ValueError("Original archive CRC failure")
        for name in sorted(stream.namelist()):
            if name.endswith("/selected_config.yaml"):
                selection = yaml.safe_load(stream.read(name))
                model, trial = selection["model"], selection["trial"]
                models[model] = {"selected_trial": trial,
                                 "checkpoint": f"checkpoints/phase5c/beauty/{model}/trials/{trial}/checkpoint.pt",
                                 "validation_ndcg@10": selection["validation_ndcg@10"]}
                planned.append((name, f"checkpoints/phase5c/beauty/{model}/selected_config.yaml", model, trial, "selection"))
            if name.endswith("/config.yaml"):
                config = yaml.safe_load(stream.read(name))
                model, trial = config["model"], config["tuning_trial"]
                parent = name.rsplit("/", 1)[0]
                status = json.loads(stream.read(parent + "/status.json"))
                for filename, kind in (("checkpoint.pt", "best"), ("latest.pt", "latest"), ("config.yaml", "config")):
                    member = parent + "/" + filename
                    if hashlib.sha256(stream.read(member)).hexdigest() != status["files"][filename]:
                        raise ValueError(f"Source artifact hash mismatch: {member}")
                    planned.append((member, f"checkpoints/phase5c/beauty/{model}/trials/{trial}/{filename}", model, trial, kind))
        if len(models) != 5 or len(planned) != 50:
            raise ValueError("Expected 30 checkpoints, 15 resolved configs and five selections")
        # Preflight every existing destination before materializing any checkpoint.
        files = []
        for member, relative, model, trial, kind in planned:
            content = stream.read(member)
            digest = hashlib.sha256(content).hexdigest()
            path = root / relative
            if path.exists() and file_hash(path) != digest:
                raise ValueError(f"Existing checkpoint artifact differs: {path}")
            files.append({"path": relative, "sha256": digest, "bytes": len(content), "model": model,
                          "trial": trial, "kind": kind, "source_member": member})
        for item in files:
            path = root / item["path"]
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with stream.open(item["source_member"]) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target)
    bundles = {}
    for scope in ("selected", "all"):
        subset = [f for f in files if scope == "all" or (f["trial"] == models[f["model"]]["selected_trial"] and f["kind"] != "latest")]
        filename = f"seqrec-phase5c-{scope}-checkpoints-v1.zip"
        output = root / "artifacts/phase5c" / filename
        build_bundle(root, subset, output)
        bundles[scope] = {"filename": filename, "url": f"{RELEASE_BASE}/{filename}", "sha256": file_hash(output),
                          "bytes": output.stat().st_size, "paths": sorted(f["path"] for f in subset)}
    manifest = {"schema": 1, "release_tag": RELEASE_TAG, "source_archive_sha256": ARCHIVE_SHA256,
                "purpose": "Validation-selected Phase 5C tuning checkpoints, not final test models",
                "models": models, "bundles": bundles, "files": sorted(files, key=lambda f: f["path"]),
                "excluded": ["dataset", "per-user predictions", "raw histories", "training logs", "credentials"]}
    write_json(root / "configs/artifacts/phase5c_checkpoints.json", manifest)
    write_json(root / "checkpoints/phase5c/manifest.json", manifest)
    print(f"Materialized 30 original checkpoints under {root / 'checkpoints/phase5c'}")
    for scope, bundle in bundles.items():
        print(f"{scope}: {bundle['filename']} ({bundle['bytes']} bytes), SHA256={bundle['sha256']}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    args = parser.parse_args()
    prepare(args.root, resolve_archive_path(args.root, args.archive))
