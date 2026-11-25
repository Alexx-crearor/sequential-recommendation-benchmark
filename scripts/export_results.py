"""Create a new portable ZIP of result artifacts without altering source runs."""
import argparse
from pathlib import Path
import zipfile
import _bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Archive exists; choose a new output name")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.root.rglob("*")):
            if path.is_file() and path.resolve() != args.output.resolve() and not path.name.endswith(".partial"):
                archive.write(path, path.relative_to(args.root.parent))
    print(args.output)


if __name__ == "__main__":
    main()
