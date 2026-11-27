"""Fetch original official-session checkpoints; no private predictions required."""
import argparse
from pathlib import Path
import _bootstrap
from fetch_phase5c_checkpoints import fetch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    parser.add_argument("--with-resume", action="store_true", help="Include 21 neural latest optimizer/RNG states")
    parser.add_argument("--archive", type=Path, help="Offline public checkpoint bundle")
    args = parser.parse_args()
    fetch(args.root, args.with_resume, args.archive, "configs/artifacts/master_session1_checkpoints.json")
