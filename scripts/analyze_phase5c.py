"""Local-only ZIP verification, aggregation and plotting; never trains or tunes."""
import argparse
from pathlib import Path
import _bootstrap
from src.analysis.phase5c import analyze_archive, ARCHIVE_NAME
from gen_fig_phase5c import generate_figures
from write_phase5c_report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path.home() / "Downloads" / ARCHIVE_NAME)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    _, report = analyze_archive(args.archive, args.root)
    if not args.skip_figures:
        generate_figures(args.root)
    write_report(args.root)
    print(f"Phase 5C VERIFIED: {report['trials']} trials, {report['checkpoint_files_verified']} checkpoints; source SHA256={report['archive_sha256']}")


if __name__ == "__main__":
    main()
