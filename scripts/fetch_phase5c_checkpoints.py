"""Download hash-locked Release checkpoints into this clone; no Torch dependency."""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request
import _bootstrap
from kaggle_session import restore
from src.utils.artifacts import file_hash


def verify_files(root, manifest, scope):
    by_path = {item["path"]: item for item in manifest["files"]}
    for name in manifest["bundles"][scope]["paths"]:
        path = root / name
        if not path.is_file() or file_hash(path) != by_path[name]["sha256"]:
            return False
    return True


def fetch(root, all_trials=False, archive=None, manifest_path=None):
    root = Path(root).resolve()
    manifest = json.loads((root / (manifest_path or "configs/artifacts/phase5c_checkpoints.json")).read_text(encoding="utf-8"))
    scope = "all" if all_trials else "selected"
    bundle = manifest["bundles"][scope]
    if verify_files(root, manifest, scope):
        print("Checkpoints already present; every file SHA256 verified.")
    else:
        path = Path(archive).resolve() if archive else root / manifest.get("cache_directory", "artifacts/phase5c") / bundle["filename"]
        if not path.exists():
            if archive:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            print(f"Downloading {scope} checkpoint bundle ({bundle['bytes']/1048576:.1f} MiB)...", flush=True)
            request = urllib.request.Request(bundle["url"], headers={"User-Agent": "seqrec-checkpoint-fetch/1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target)
            if file_hash(temporary) != bundle["sha256"]:
                raise ValueError("Downloaded bundle SHA256 mismatch; .partial retained, not extracted")
            temporary.replace(path)
        if file_hash(path) != bundle["sha256"]:
            raise ValueError("Existing bundle SHA256 mismatch; preserved without extraction")
        restore(path, root)
        if not verify_files(root, manifest, scope):
            raise ValueError("Checkpoint member verification failed")
    for model, record in sorted(manifest["models"].items()):
        print(f"{model}: {root / record['checkpoint']}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    parser.add_argument("--all-trials", action="store_true", help="Download all 15 trials' best/latest states (30 checkpoints)")
    parser.add_argument("--archive", type=Path, help="Use a manually downloaded checkpoint bundle offline")
    args = parser.parse_args()
    fetch(args.root, args.all_trials, args.archive)
