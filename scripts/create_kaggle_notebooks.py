"""Generate thin Kaggle orchestration notebooks; model code stays in src/."""
import json
from pathlib import Path
import _bootstrap


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


def write(name, cells):
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "cells": cells}
    for index, cell in enumerate(cells): cell["id"] = f"cell-{index:03d}"
    path = _bootstrap.ROOT / "notebooks" / name
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def main():
    common = '''from pathlib import Path
import os, sys, json, subprocess
INPUT_DIR = Path(os.environ.get("SEQREC_INPUT_DIR", "/kaggle/input"))
WORK_DIR = Path(os.environ.get("SEQREC_WORK_DIR", "/kaggle/working"))
REPO = Path(os.environ.get("SEQREC_REPO_DIR", str(WORK_DIR / "sequential-recommendation")))
PYTHON = os.environ.get("SEQREC_PYTHON", str(WORK_DIR / "seqrec-env/bin/python"))
if not Path(PYTHON).exists():
    PYTHON = sys.executable
def run(*args):
    subprocess.run([PYTHON, *map(str, args)], cwd=REPO, check=True)
print({"repo": str(REPO), "python": PYTHON, "input": str(INPUT_DIR), "work": str(WORK_DIR)})
'''
    setup = [markdown("# Kaggle setup\nThin wrappers around repository CLIs. Enable Internet for installation and select a GPU accelerator. Source and result archives are separate uploads. No previous working directory is assumed to survive a session."),
             code(common), code('''# Set these to your attached Kaggle Dataset paths.
SOURCE_ARCHIVE = INPUT_DIR / "seqrec-source/seqrec-kaggle-source.zip"
PRIOR_RESULTS_ARCHIVE = None  # e.g. INPUT_DIR / "previous-run/seqrec-results-session1.zip"
REPO.mkdir(parents=True, exist_ok=True)
# Validate paths before extracting the trusted source package.
import zipfile
with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
    for member in archive.infolist():
        target = (REPO / member.filename).resolve()
        if not target.is_relative_to(REPO.resolve()):
            raise ValueError("Unsafe archive path")
        if target.exists() and not member.is_dir():
            raise ValueError("Repository already exists: use it or select a fresh REPO directory")
    archive.extractall(REPO)
print(REPO)
'''), code('''# Use a separate supported Python environment when the Kaggle kernel is newer.
if sys.version_info[:2] not in [(3, 10), (3, 11)]:
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    subprocess.run([sys.executable, "-m", "uv", "python", "install", "3.11"], check=True)
    subprocess.run([sys.executable, "-m", "uv", "venv", "--seed", "--python", "3.11", str(WORK_DIR / "seqrec-env")], check=True)
    PYTHON = str(WORK_DIR / "seqrec-env/bin/python")
run("-m", "pip", "install", "-r", "requirements-lock.txt", "--extra-index-url", "https://download.pytorch.org/whl/cu121")
run("scripts/check_environment.py")
run("scripts/run_smoke_test.py", "--device", "cpu")
# Run CUDA smoke after check_environment confirms a usable accelerator.
run("scripts/run_smoke_test.py", "--device", "cuda")
'''), code('''if PRIOR_RESULTS_ARCHIVE is not None:
    run("scripts/kaggle_session.py", "restore", "--archive", PRIOR_RESULTS_ARCHIVE, "--destination", REPO)
    subprocess.run([PYTHON, "scripts/verify_results.py", "--root", "results/core"], cwd=REPO, check=False)  # INCOMPLETE is expected before resume
# Existing processed bundles may be attached read-only or packaged with source.
DATASET = "beauty"
BUNDLE = None  # set Path('/kaggle/input/seqrec-data/data/processed/beauty/HASH') to reuse an attached bundle
if BUNDLE is None:
    existing = sorted((REPO / "data/processed" / DATASET).glob("*/manifest.json"))
    if not existing:
        run("scripts/download_data.py", "--dataset", DATASET)
        run("scripts/preprocess_data.py", "--dataset", DATASET)
    bundles = sorted((REPO / "data/processed" / DATASET).glob("*/manifest.json"))
    if len(bundles) != 1:
        raise ValueError("Select the exact bundle explicitly; multiple or no bundles found")
    BUNDLE = bundles[0].parent
run("scripts/verify_data.py", "--bundle", BUNDLE, "--output", "results/reports/kaggle_data_audit.json")
(WORK_DIR / "seqrec_session.json").write_text(json.dumps({"repo": str(REPO), "python": PYTHON, "bundle": str(BUNDLE), "dataset": DATASET}, indent=2))
'''), markdown("Save seqrec_session.json and the exported result ZIP as notebook outputs. A later Kaggle session must attach the ZIP and restore it; /kaggle/working is temporary.")]
    training = [markdown("# Kaggle training\nSelect one configuration or bounded tuning job. Defaults only generate a dry-run manifest. All implementations are imported by repository scripts."), code(common), code('''SESSION_FILE = WORK_DIR / "seqrec_session.json"
session = json.loads(SESSION_FILE.read_text())
REPO, PYTHON, BUNDLE = Path(session["repo"]), session["python"], Path(session["bundle"])
PROFILE = "core_research"  # pilot | core_research | full_research
MODE = "dry-run"  # dry-run | tune | train | analysis | benchmark
EXPERIMENT = "E1"
DATASET = session["dataset"]
MODEL = "sasrec_ce"
VARIANT = None  # A1-A4 for E2; F0-F5 for E4
SEED = 7
RETENTION = None
MAX_SEQ_LENGTH = None
DEVICE = "cuda:0"
FINAL_SEEDS = False
SELECTION = None  # path to selected_config.yaml from validation-only tuning
RESUME = True
RESULTS_ROOT = REPO / ("results/pilots_phase5a" if PROFILE == "pilot" else "results/core" if PROFILE == "core_research" else "results/runs")
OVERRIDES = []  # e.g. ['batch_size=64','gradient_accumulation_steps=2'] for independent CE/BCE only
'''), code('''run("scripts/check_environment.py")
run("scripts/verify_data.py", "--bundle", BUNDLE)
if MODE == "benchmark":
    run("scripts/benchmark_pipeline.py", "--bundle", BUNDLE, "--device", DEVICE,
        "--output", "results/phase5a/kaggle_benchmark_session1", "--batches", "20", "--repeats", "2", "--full-validation")
elif MODE == "tune":
    command = ["scripts/run_tuning.py", "--profile", PROFILE, "--bundle", BUNDLE, "--model", MODEL,
               "--device", DEVICE, "--results-root", "results/tuning"]
    if RESUME: command.append("--resume")
    run(*command)
else:
    command = ["scripts/run_experiment.py", "--profile", PROFILE, "--experiment", EXPERIMENT,
               "--dataset", DATASET, "--bundle", BUNDLE, "--device", DEVICE, "--results-root", RESULTS_ROOT]
    if MODE == "dry-run": command += ["--dry-run", "--output", "results/tables/kaggle_manifest.csv"]
    if MODE not in ("dry-run", "train", "analysis"): raise ValueError("Unknown MODE")
    if MODE != "analysis":
        if MODEL: command += ["--model", MODEL]
        if VARIANT: command += ["--variant", VARIANT]
        if SEED is not None: command += ["--seed", str(SEED)]
    if FINAL_SEEDS: command.append("--final-seeds")
    if RETENTION is not None: command += ["--retention", str(RETENTION)]
    if MAX_SEQ_LENGTH is not None: command += ["--max-seq-length", str(MAX_SEQ_LENGTH)]
    if SELECTION: command += ["--selection", str(SELECTION)]
    if RESUME: command.append("--resume")
    for override in OVERRIDES: command += ["--set", override]
    run(*command)
'''), code('''# Review utilization while working (training CLIs write epoch memory and timings).
subprocess.run(["nvidia-smi"], check=False)
subprocess.run([PYTHON, "scripts/verify_results.py", "--root", str(RESULTS_ROOT)], cwd=REPO, check=False)
# Export explicitly before the session expires; choose a new name each time.
EXPORT_PATH = WORK_DIR / "seqrec-results-session1.zip"
run("scripts/export_results.py", "--root", REPO / "results", "--output", EXPORT_PATH)
print("Download or Save Version with output:", EXPORT_PATH)
''')]
    analysis = [markdown("# Kaggle analysis\nRead verified real artifacts only. Pilot, tuning and final research roots stay separate. No placeholder figures or test-driven configuration selection."), code(common), code('''session = json.loads((WORK_DIR / "seqrec_session.json").read_text())
REPO, PYTHON, BUNDLE = Path(session["repo"]), session["python"], Path(session["bundle"])
RESULTS_ROOT = REPO / "results/core"
run("scripts/verify_results.py", "--root", RESULTS_ROOT)
run("scripts/aggregate_results.py", "--root", RESULTS_ROOT, "--output", "results/tables/core")
'''), code('''# Run these after the preselected E1 runs have completed.
RUN_E7_E8 = False
if RUN_E7_E8:
    for experiment in ["E7", "E8"]:
        run("scripts/run_analysis.py", "--experiment", experiment, "--bundle", BUNDLE,
            "--results-root", RESULTS_ROOT, "--output", "results/analysis/core", "--device", "cuda:0")
'''), code('''import pandas as pd
table = REPO / "results/tables/core/per_seed_results.csv"
if table.exists():
    display(pd.read_csv(table))
else:
    print("No verified final results yet. Pilot and validation-only tuning artifacts are excluded.")
'''), code('''EXPORT_PATH = WORK_DIR / "seqrec-analysis-session1.zip"
run("scripts/export_results.py", "--root", REPO / "results", "--output", EXPORT_PATH)
print(EXPORT_PATH)
''')]
    write("Kaggle_Setup.ipynb", setup)
    write("Kaggle_Training.ipynb", training)
    write("Kaggle_Analysis.ipynb", analysis)


if __name__ == "__main__": main()
