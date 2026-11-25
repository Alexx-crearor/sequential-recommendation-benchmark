"""Package source or restore exported result trees without overwriting conflicts."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import zipfile
import _bootstrap
from src.utils.artifacts import file_hash


def restore(archive_path, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        # Validate every destination/conflict before writing anything.
        entries = []
        for info in archive.infolist():
            name = PurePosixPath(info.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or any(":" in part for part in name.parts):
                raise ValueError("Unsafe archive path")
            path = destination.joinpath(*name.parts).resolve()
            if not path.is_relative_to(destination):
                raise ValueError("Archive path escapes destination")
            if info.is_dir():
                continue
            if path.exists():
                import hashlib
                digest = hashlib.sha256()
                with archive.open(info) as source:
                    for block in iter(lambda: source.read(1048576), b""):
                        digest.update(block)
                if digest.hexdigest() != file_hash(path):
                    raise ValueError(f"Existing artifact differs; restore into a new directory: {path}")
                continue
            entries.append((info, path))
        for info, path in entries:
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
    return destination


def package(output, include_data=False):
    output = Path(output)
    if output.exists():
        raise ValueError("Output archive exists; choose a new name")
    output.parent.mkdir(parents=True, exist_ok=True)
    root = _bootstrap.ROOT
    directories = ["src", "scripts", "configs", "tests", "docs", "notebooks", "benchmarks"]
    if include_data:
        directories.append("data/processed")
    files = [root/name for name in ["README.md", "IMPLEMENTATION_PLAN.md", "pyproject.toml", "requirements.txt", "requirements-lock.txt", "requirements-neural.txt"]]
    for directory in directories:
        files.extend((root/directory).rglob("*"))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc" and not path.name.endswith(".partial"):
                archive.write(path, path.relative_to(root))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["package", "restore"])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--include-data", action="store_true")
    args = parser.parse_args()
    if args.action == "package":
        print(package(args.archive, args.include_data))
    else:
        if args.destination is None:
            parser.error("restore requires --destination")
        print(restore(args.archive, args.destination))
