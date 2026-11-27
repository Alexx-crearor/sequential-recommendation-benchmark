"""Personal-machine-only analysis of saved official E1/E2 predictions and timings."""
import argparse
from pathlib import Path
import _bootstrap
from src.analysis.master_session1 import analyze, ARCHIVE_NAME, BUNDLE_HASH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    archive = args.archive or args.root / "artifacts/master_session1" / ARCHIVE_NAME
    bundle = args.bundle or args.root / "data/processed/beauty" / BUNDLE_HASH
    _, report = analyze(archive, args.root, bundle, args.resamples)
    if not args.skip_figures:
        from gen_fig_master_session1 import generate
        generate(args.root)
    from write_master_session1_report import write_report
    write_report(args.root)
    print(f"VERIFIED {report['successful_runs']} official runs; E1 complete, E2 partial; E7 DEFERRED")
