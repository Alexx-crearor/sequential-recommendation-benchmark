# Phase 5A update

For the current profiles, notebooks, bounded tuning and cross-session workflow, use [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md). The earlier Phase 3/4 instructions below remain historical context.

# Kaggle execution: E1-E5

Select a GPU accelerator and enable Internet for initial setup/data download. Upload this repository as a Kaggle Dataset or clone your own repository into a writable directory. Keep code and outputs under `/kaggle/working`; attached datasets under `/kaggle/input` are read-only. All examples below assume the repository root is the working directory. No full grid has been launched locally.

## Environment

Use Python 3.10 or 3.11. If the notebook's Python is newer, create a separate 3.11 environment in a notebook setup cell:

```bash
python -m pip install uv
uv python install 3.11
uv venv --seed --python 3.11 /kaggle/working/seqrec-env
/kaggle/working/seqrec-env/bin/python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Use `/kaggle/working/seqrec-env/bin/python` in place of `python` in the following commands. If Python is already 3.10/3.11, install directly with the same requirements command. The local validated build is PyTorch 2.5.1+cu121. Check actual GPU execution instead of assuming a runtime is compatible:

```bash
python scripts/check_environment.py
python scripts/run_smoke_test.py --device cpu
python scripts/run_smoke_test.py --device cuda
```

CUDA selectors accept `cuda:0` or `cuda:1`; runs are single-device. Multiple GPUs can run separate selected configurations into separate roots. FENRec uses sklearn CPU Lloyd clustering with GPU sequence encoding and loss. Set `--set cpu_threads=4` to control CPU work. AMP is opt-in using `--set amp=true` for the Phase 3 models. FENRec requires FP32. If a batch does not fit, lower `--set batch_size=64` and record that configuration; contrastive batch size changes its negative set, so match it across all E4 variants. The current trainer does not implement gradient accumulation.

## Data preparation

```bash
python scripts/download_data.py --dataset beauty
python scripts/preprocess_data.py --dataset beauty
python scripts/verify_data.py --bundle data/processed/beauty/HASH --output results/reports/beauty_data_audit.json
```

Repeat for `sports` and `toys` for E1. Replace `HASH` with the exact directory printed by preprocessing; do not use a different split for different models. Bundles include selected metadata, mappings, split manifests and statistics. To use attached raw data, pass `--interactions /kaggle/input/YOUR_DATA/reviews_Beauty_5.json.gz --metadata /kaggle/input/YOUR_DATA/meta_Beauty.json.gz`. Do not write into the attached dataset.

## Pilot and dry-run gate

```bash
python scripts/run_pilots.py --bundle data/processed/beauty/HASH --device cuda:0
python scripts/run_experiment.py --experiment all --dry-run
```

Pilots use two epochs, twenty batches per epoch and 512 validation users, then complete test-user predictions. They are tagged and stored under `results/pilots`, excluded from final aggregation. Standard FENRec pilots do not reach the twenty-epoch hard-negative warm-up. Its post-warm-up branch has separate tests and a separately configured pilot; never relabel this as converged training.

## Individual full runs

```bash
# E1: choose a model and dataset explicitly; TIGER remains Phase 6.
python scripts/run_experiment.py --experiment E1 --dataset beauty --model sasrec_ce --seed 7 --bundle data/processed/beauty/HASH --device cuda:0 --config configs/experiments/full.yaml

# E2: all four variants use the same post-norm backbone and parameter count.
python scripts/run_experiment.py --experiment E2 --dataset beauty --variant A3 --seed 7 --bundle data/processed/beauty/HASH --device cuda:0 --config configs/experiments/full.yaml

# E3: same seed-specific retention manifest across methods; held-outs stay fixed.
python scripts/run_experiment.py --experiment E3 --dataset beauty --model fenrec --retention 0.4 --seed 7 --bundle data/processed/beauty/HASH --device cuda:0 --config configs/experiments/full.yaml

# E4: F0 through F5, identical reference backbone and batch size.
python scripts/run_experiment.py --experiment E4 --dataset beauty --variant F5 --retention 0.4 --seed 7 --bundle data/processed/beauty/HASH --device cuda:0 --config configs/experiments/full.yaml

# E5: requested sequence length; replace model to compare others.
python scripts/run_experiment.py --experiment E5 --dataset beauty --model bert4rec --max-seq-length 100 --seed 7 --bundle data/processed/beauty/HASH --device cuda:0 --config configs/experiments/full.yaml
```

Drop `--seed` to execute all three seeds for the selected subset. Drop a variant/retention/length selector to run that axis deliberately. E1 execution requires an implemented `--model`; otherwise the planned TIGER entry causes an explicit error before training begins. MostPopular executes once and uses CPU. Running E1-E5 independently permits review between experiments.

Exact E1 configurations can be reused by E3 at retention 1 and E5 at length 50; complete verified artifacts are copied with `reuse.json`. E2 now uses matched post-norm blocks for all four variants and cannot reuse SASRec/BERT4Rec E1. FENRec F5 is kept separately labeled from the E1 default and is not automatically aliased.

## Resume, verification and export

Reissue the **same command** with `--resume`. `latest.pt` restores the last complete epoch's model, optimizer, scaler and all RNG states; the best validation checkpoint is independent. Interrupted partial epochs replay from the preceding epoch. Changes to training settings produce a new configuration hash and therefore a new run. To test interruption deliberately, use `scripts/run_train.py --stop-after-epochs 1` with unchanged `--set max_epochs=...`, then repeat with `--resume` and without the stop flag. This is not a way to extend a completed fixed-budget run.

Verified SUCCESS runs skip automatically. Incomplete/corrupt runs are preserved; corrupted SUCCESS artifacts fail verification. A run that exhausts its training limit is flagged in `efficiency.json`; SUCCESS means pipeline/artifact integrity, not convergence.

```bash
python scripts/verify_results.py --root results/runs
python scripts/aggregate_results.py
python scripts/export_results.py --root results --output /kaggle/working/seqrec-results-attempt1.zip
```

Save the notebook version with outputs to retain the ZIP, or download/export it before the Kaggle session expires. Restore the whole result tree in a later session to preserve latest/best checkpoints and configuration paths. Source runs remain unchanged. The current aggregate CLI generates verified overall/per-seed results only; the full E2-E5 analysis tables and plots are a later analysis milestone.

Bounded tuning is not automated yet. Use at most the documented twelve validation-only trials per model/dataset, record every configuration, and choose settings before examining test results. Full E1-E5 training commands are operational; final scientific comparisons still need this controlled selection and multi-seed analysis.
