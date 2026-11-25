"""Fetch pinned upstream comparison code without changing existing checkouts."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *map(str, args)], text=True).strip()


def fetch(destination, names=None):
    manifest = json.loads((ROOT / "configs/reference_repositories.json").read_text())
    for name in names or manifest:
        record = manifest[name]
        path = destination / name
        if path.exists():
            if not (path / ".git").exists():
                raise ValueError(f"Existing path is not a checkout: {path}")
            origin = git("-C", path, "remote", "get-url", "origin")
            revision = git("-C", path, "rev-parse", "HEAD")
            if origin.removesuffix(".git") != record["url"].removesuffix(".git") or revision != record["commit"]:
                raise ValueError(f"Existing checkout differs; preserve it and choose another --destination: {path}")
            if git("-C", path, "status", "--porcelain", "--untracked-files=no"):
                raise ValueError(f"Existing checkout has local changes: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", record["url"], str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", record["commit"]], check=True)
        print(f"{name}: {git('-C', path, 'rev-parse', 'HEAD')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "references")
    parser.add_argument("--only", nargs="+", choices=["BERT4Rec", "FENRec", "GRU4Rec", "SASRec"])
    args = parser.parse_args()
    fetch(args.destination.resolve(), args.only)
