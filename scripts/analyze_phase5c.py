"""Local-only ZIP verification, aggregation and plotting; never trains or tunes."""
import argparse
from pathlib import Path
import _bootstrap
from src.analysis.phase5c import analyze_archive, resolve_archive_path
from gen_fig_phase5c import generate_figures
from write_phase5c_report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Original private ZIP; default artifacts/phase5c/ under repository root")
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    archive = resolve_archive_path(args.root, args.archive)
    if not archive.is_file():
        parser.error(f"Original private archive required at {archive}. Place it there or pass --archive. Public checkpoints: python scripts/fetch_phase5c_checkpoints.py")
    _, report = analyze_archive(archive, args.root)
    if not args.skip_figures:
        generate_figures(args.root)
    write_report(args.root)
    print(f"Phase 5C VERIFIED: {report['trials']} trials, {report['checkpoint_files_verified']} checkpoints; source SHA256={report['archive_sha256']}")


if __name__ == "__main__":
    main()
