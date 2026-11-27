"""Publish only six new E2 runs; reuse the immutable session-one Release."""
import argparse
import os
from pathlib import Path
import shutil
import pandas as pd
import _bootstrap
from prepare_phase5c_checkpoints import build_bundle
from src.analysis.course_final import ARCHIVE_SHA256
from src.analysis.phase5c import read_json, require
from src.utils.artifacts import file_hash, write_json


def prepare(root):
    root = Path(root).resolve()
    audit = read_json(root / "results/tables/course_final/artifact_audit.json")
    require(audit["state"] == "VERIFIED" and audit["runs_verified"] == 28 and audit["prior_runs_unchanged"] == 22, "Final audit required")
    raw = root / "results/raw/e2_final" / ARCHIVE_SHA256
    rows = pd.read_csv(root / "results/tables/course_final/runs.csv")
    rows = rows[rows.new_in_final_session]
    require(len(rows) == 6, "Expected six new E2 runs only")
    files, models = [], {}
    for row in rows.itertuples(index=False):
        source = raw / row.path
        status = read_json(source / "status.json")
        key = f"{row.experiment}__{row.model}__{row.variant}__{row.seed}"
        base = f"checkpoints/e2_final/{row.experiment}/{row.model}/{row.variant}/{row.seed}"
        models[key] = {"experiment": row.experiment, "model": row.model, "variant": row.variant, "seed": row.seed,
                       "config_hash": row.config_hash, "checkpoint": base+"/checkpoint.pt", "format": "pytorch_weights_only"}
        for name in ("checkpoint.pt", "latest.pt", "config.yaml"):
            checksum = file_hash(source / name)
            require(checksum == status["files"][name], "Changed source checkpoint")
            relative = base+"/"+name
            target = root / relative
            require(not target.exists() or file_hash(target) == checksum, "Existing final checkpoint differs")
            files.append({"path": relative, "sha256": checksum, "bytes": (source / name).stat().st_size,
                          "source_member": row.path+"/"+name, "kind": "latest" if name == "latest.pt" else "best" if name == "checkpoint.pt" else "config"})
    for item in files:
        target, source = root / item["path"], raw / item["source_member"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                with source.open("rb") as origin, target.open("xb") as output:
                    shutil.copyfileobj(origin, output)
    tag, bundles = "e2-final-checkpoints-v1", {}
    for scope in ("selected", "all"):
        subset = [f for f in files if scope == "all" or f["kind"] != "latest"]
        label = "best" if scope == "selected" else "resume"
        filename = f"seqrec-e2-final-{label}-delta-v1.zip"
        path = root / "artifacts/e2_final" / filename
        build_bundle(root, subset, path)
        bundles[scope] = {"filename": filename, "bytes": path.stat().st_size, "sha256": file_hash(path),
                          "url": f"https://github.com/Alexx-crearor/sequential-recommendation-benchmark/releases/download/{tag}/{filename}",
                          "paths": sorted(f["path"] for f in subset)}
        print(filename, path.stat().st_size, flush=True)
    delta = {"schema": 1, "release_tag": tag, "cache_directory": "artifacts/e2_final", "source_archive_sha256": ARCHIVE_SHA256,
             "purpose": "Six newly completed E2 runs only; old 22 runs remain in immutable prior Release",
             "models": models, "files": files, "bundles": bundles}
    write_json(root / "configs/artifacts/e2_final_checkpoints.json", delta)
    previous = read_json(root / "configs/artifacts/master_session1_checkpoints.json")
    require(not (set(previous["models"]) & set(models)), "Duplicate completed checkpoint keys")
    combined = {"schema": 2, "release_tag": tag, "training_closed": True, "source_archive_sha256": ARCHIVE_SHA256,
                "parts": ["configs/artifacts/master_session1_checkpoints.json", "configs/artifacts/e2_final_checkpoints.json"],
                "models": {**previous["models"], **models}, "files": [*previous["files"], *files],
                "best_checkpoints": 28, "latest_states": 27, "prior_22_runs_byte_unchanged": True,
                "distribution": "composite manifests; no repackaging or re-upload of previous weights"}
    write_json(root / "configs/artifacts/course_final_checkpoints.json", combined)
    write_json(root / "checkpoints/e2_final/manifest.json", delta)
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    prepare(parser.parse_args().root)
