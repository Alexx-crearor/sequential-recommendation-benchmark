"""Explicit downloads only. Existing files are preserved."""
import argparse
import os
from pathlib import Path
from urllib.request import urlopen
import shutil
import yaml
import _bootstrap
from src.utils.artifacts import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["beauty", "sports", "toys"])
    parser.add_argument("--kind", choices=["interactions", "metadata", "both"], default="both")
    args = parser.parse_args()
    config = yaml.safe_load((_bootstrap.ROOT / "configs/datasets" / f"{args.dataset}.yaml").read_text())
    for kind in (["interactions", "metadata"] if args.kind == "both" else [args.kind]):
        path = Path(config[kind])
        if path.exists():
            print(f"Preserved existing {path}; sha256={file_hash(path)}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with urlopen(config[f"{kind}_url"], timeout=60) as response, open(temporary, "wb") as output:
            shutil.copyfileobj(response, output)
        os.replace(temporary, path)
        write_json(str(path) + ".source.json", {"url": config[f"{kind}_url"], "sha256": file_hash(path)})
        print(path)


if __name__ == "__main__":
    main()
