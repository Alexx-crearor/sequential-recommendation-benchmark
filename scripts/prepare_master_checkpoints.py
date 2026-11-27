"""Package only verified original official weights/configs for Release distribution."""
import argparse
from pathlib import Path
import shutil
import pandas as pd
import _bootstrap
from prepare_phase5c_checkpoints import build_bundle
from src.analysis.master_session1 import ARCHIVE_SHA256
from src.analysis.phase5c import read_json, require
from src.utils.artifacts import file_hash, write_json


def prepare(root):
    root = Path(root).resolve()
    report = read_json(root / "results/tables/master_session1/artifact_audit.json")
    require(report["state"] == "VERIFIED" and report["successful_runs"] == 22, "Official audit required")
    runs = pd.read_csv(root / "results/tables/master_session1/runs.csv")
    raw = root / "results/raw/master_session1" / ARCHIVE_SHA256
    planned, models = [], {}
    for row in runs.itertuples(index=False):
        source = raw / row.path
        status = read_json(source / "status.json")
        key = f"{row.experiment}__{row.model}__{row.variant}__{row.seed}"
        base = f"checkpoints/master_session1/{row.experiment}/{row.model}/{row.variant}/{row.seed}"
        models[key] = {"model": row.model, "variant": row.variant, "seed": row.seed, "experiment": row.experiment,
                       "config_hash": row.config_hash, "checkpoint": base + "/checkpoint.pt",
                       "format": "numpy_npz" if row.model == "mostpopular" else "pytorch_weights_only"}
        for name in ("checkpoint.pt", "latest.pt", "config.yaml"):
            if name not in status["files"]:
                continue
            digest = file_hash(source / name)
            require(digest == status["files"][name], "Source checkpoint changed")
            relative = base + "/" + name
            target = root / relative
            require(not target.exists() or file_hash(target) == digest, "Existing checkpoint differs")
            planned.append({"path": relative, "sha256": digest, "bytes": (source / name).stat().st_size,
                            "source_member": row.path + "/" + name, "kind": "latest" if name == "latest.pt" else "best" if name == "checkpoint.pt" else "config"})
    for item in planned:
        target = root / item["path"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with (raw / item["source_member"]).open("rb") as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
    tag = "master-session1-checkpoints-v1"
    bundles = {}
    for scope in ("selected", "all"):
        items = [f for f in planned if scope == "all" or f["kind"] != "latest"]
        label = "best" if scope == "selected" else "resume"
        filename = f"seqrec-master-session1-{label}-checkpoints-v1.zip"
        output = root / "artifacts/master_session1" / filename
        build_bundle(root, items, output)
        bundles[scope] = {"filename": filename, "bytes": output.stat().st_size, "sha256": file_hash(output),
                          "url": f"https://github.com/Alexx-crearor/sequential-recommendation-benchmark/releases/download/{tag}/{filename}",
                          "paths": sorted(f["path"] for f in items)}
        print(filename, output.stat().st_size, flush=True)
    manifest = {"schema": 1, "release_tag": tag, "source_archive_sha256": ARCHIVE_SHA256,
                "cache_directory": "artifacts/master_session1", "purpose": "Original completed E1 (16) and partial E2 (6), not pending experiments",
                "models": models, "bundles": bundles, "files": sorted(planned, key=lambda f: f["path"]),
                "excluded": ["datasets", "per-user predictions/histories", "logs", "credentials"]}
    write_json(root / "configs/artifacts/master_session1_checkpoints.json", manifest)
    write_json(root / "checkpoints/master_session1/manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    prepare(parser.parse_args().root)
