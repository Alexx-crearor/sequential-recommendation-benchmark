"""Fetch all 28 final-course best checkpoints using old + delta Releases."""
import argparse
import json
from pathlib import Path
import _bootstrap
from fetch_phase5c_checkpoints import fetch
from src.utils.artifacts import file_hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    parser.add_argument("--with-resume", action="store_true", help="Also fetch all 27 latest optimizer/RNG states")
    args = parser.parse_args()
    manifest = json.loads((args.root / "configs/artifacts/course_final_checkpoints.json").read_text(encoding="utf-8"))
    for part in manifest["parts"]:
        fetch(args.root, args.with_resume, manifest_path=part)
    expected = [f for f in manifest["files"] if args.with_resume or f["kind"] != "latest"]
    if any(file_hash(args.root / f["path"]) != f["sha256"] for f in expected):
        raise ValueError("Composite final checkpoint integrity failure")
    print("Verified", manifest["best_checkpoints"], "best checkpoints", "+ 27 latest states" if args.with_resume else "")
