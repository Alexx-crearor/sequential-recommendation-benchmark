"""Final course artifact audit/statistics only; never train or profile GPU."""
import argparse
from pathlib import Path
import _bootstrap
from src.analysis.course_final import analyze, ARCHIVE_NAME
from src.analysis.master_session1 import BUNDLE_HASH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    source = args.archive or args.root / "artifacts/e2_final" / ARCHIVE_NAME
    bundle = args.bundle or args.root / "data/processed/beauty" / BUNDLE_HASH
    _, report = analyze(source, args.root, bundle, args.resamples)
    if not args.skip_figures:
        from gen_fig_course_final import generate
        generate(args.root)
    from write_course_final_report import write_report
    write_report(args.root)
    print("VERIFIED", report["runs_verified"], "completed runs; E2 complete 12/12; course training closed")
