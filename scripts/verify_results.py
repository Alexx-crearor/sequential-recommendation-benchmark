import argparse
from pathlib import Path
import _bootstrap
from src.utils.runs import verify_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/runs"))
    args = parser.parse_args()
    failures, count = [], 0
    for status in args.root.rglob("status.json"):
        count += 1
        try:
            import json
            if json.loads(status.read_text())["state"] == "VALIDATED":
                from src.utils.validation_runs import verify_validation_run
                verify_validation_run(status.parent)
            else:
                verify_run(status.parent)
            print(f"VALID {status.parent}")
        except Exception as error:
            failures.append(str(status.parent))
            print(f"INVALID {status.parent}: {error}")
    print(f"Verified {count} runs; {len(failures)} invalid")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
