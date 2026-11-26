# Sequential recommendation benchmark

A reproducible PyTorch benchmark for chronological next-item recommendation, comparing recurrent, causal attention, bidirectional masked-item and FENRec models under full-catalogue evaluation. Phase 1-4 implementations and real Amazon Beauty pilot validation are complete. Pilot scores are pipeline checks, not final research results. TIGER and final multi-seed research runs remain future work.

Phase 5A adds an optimized pipeline and Kaggle execution package. Start with [KAGGLE_GUIDE.md](docs/KAGGLE_GUIDE.md), [measured benchmarks and budgets](docs/EXPERIMENT_BUDGET.md), and [the implementation audit](docs/PHASE5A_AUDIT.md). The three Kaggle notebooks call repository scripts rather than duplicating models.

## Phase 5C — verified Kaggle tuning

The original `seqrec-phase5c-final-20261126_132556_321898.zip` is the sole experimental source for this phase (SHA256 `7861c626561fc1d7914abb8ee5a7eb4d552a932aca5dfbc54419648efb0176fa`). All **15 validation-only trials** are verified: three each for SASRec-CE/BCE, GRU4Rec-CE, BERT4Rec and FENRec, Beauty seed 7. All 30 best/latest checkpoints and 150 status-hashed artifacts pass integrity checks. Selected configurations are available in [configs/selected/phase5c/beauty](configs/selected/phase5c/beauty).

Read the [Vietnamese analysis and verification limits](docs/PHASE5C_ANALYSIS_VI.md), [all-trial summary](results/tables/phase5c/tuning_trials.csv), [selected hyperparameters](results/tables/phase5c/selected_hyperparameters.csv), [audit](results/tables/phase5c/artifact_audit.json) and [analysis notebook](notebooks/Phase5C_Analysis.ipynb). Figures are saved as 300 DPI PNG and vector PDF under `results/figures/phase5c/`. Nine trials exhausted their epoch budgets; six stopped by patience. This completes the exported bounded tuning batch, not final test evaluation or a claim of convergence for capped runs.

Reproduce analysis **on a personal machine**, with the original private ZIP available:

```powershell
python -m pip install -r requirements-analysis.txt
python scripts/analyze_phase5c.py --archive "$env:USERPROFILE/Downloads/seqrec-phase5c-final-20261126_132556_321898.zip"
```

The notebook also accepts `SEQREC_PHASE5C_ARCHIVE`. Raw extraction is immutable under ignored `results/raw/phase5c/<sha>/`; private derived provenance is ignored under `results/derived/phase5c/<sha>/`. Only aggregate tables, figures, selection YAMLs and analysis source are versioned. Missing original ZIP prevents full re-audit; figures can still be regenerated from committed aggregate CSVs. The archive's dataset audit and split IDs are consistent, but dataset bytes and a training-source attestation are absent from the ZIP. Final E1 multi-seed runs, E2–E5, final-checkpoint E7/E8 and TIGER/E6 remain pending. Kaggle is reserved for training/testing/evaluation/GPU measurement; analysis and repository publication run locally.

## Reconstruct from Git

After cloning the confirmed GitHub URL, enter the repository and use Python 3.10 or 3.11:

```bash
python -m venv .venv
# Linux/Kaggle: source .venv/bin/activate
# PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
python scripts/fetch_references.py
python -m pytest -q
python scripts/check_environment.py
```

Reference downloads are optional at runtime but required for all numerical source-comparison tests. Their URLs and exact commits are locked in `configs/reference_repositories.json`. The fetcher never resets an existing checkout. Read [Git publication and reconstruction](docs/GITHUB_PUBLICATION.md) for exclusions, audits and push commands.

Datasets, per-user predictions, checkpoints, training logs and local ZIP exports are deliberately absent from Git. Recreate data using the commands below; restore full prior runs separately using the Kaggle guide. Lightweight pilot/performance summaries and manifests remain versioned under `results/`. Pin the code revision, dependencies, dataset/split hashes and complete resolved run configuration; use seed 7 for tuning and 7/42/2026 for selected final runs. Hardware or parallel clustering changes can affect floating-point trajectories.

Profiles preserve the original grid: `pilot`, `core_research` (Beauty, initial seed 7; E3 0.4/0.7/1.0; E5 10/20/50), and `full_research` (original E1-E8 including future TIGER). Core has 45 initial entries or 129 with `--final-seeds`; Full retains 209. Default roots keep pilots, core results and full research separate.

```powershell
python scripts/run_experiment.py --profile core_research --experiment all --dry-run --output results/tables/core_initial_manifest.csv
python scripts/run_tuning.py --profile core_research --model sasrec_ce --bundle $bundle --device cuda:0 --trials 6 --results-root results/tuning --resume
python scripts/kaggle_session.py package --archive seqrec-kaggle-source.zip --include-data
```

Set `$bundle` to the verified bundle path shown below before running tuning. Tuning writes VALIDATED artifacts and never evaluates test. It produces a validation-selected YAML for `--selection`; pilot-only selections are rejected for research execution. Whole-grid local execution has not been launched.

Exact full-catalogue ranking is now device resident. GRU padding and host-to-device input assembly are vectorized, train-only tensors are cached, validation/checkpoint cadence and loader settings are configurable, and independent CE/BCE losses support correctly weighted gradient accumulation. Active FENRec contrastive variants reject accumulation because it would split their negative pools. CE activation checkpointing and Phase 3 AMP remain opt-in; AMP was slower on the local T1000. Core training enforces a sufficient post-warm-up FENRec budget. Latest checkpoints retain the corresponding best weights, scaler, optimizer and all RNG states for recovery.

Phase 1-4 implementation for a research-oriented university project. Existing Phase 2 components are preserved and extended. See [implementation status](IMPLEMENTATION_PLAN.md), [method audit](docs/METHOD_REFERENCES.md), [protocol](docs/EXPERIMENT_PROTOCOL.md), and [Kaggle workflow](docs/KAGGLE_WORKFLOW.md).

Implemented models: MostPopular, GRU4Rec-CE, SASRec-BCE, SASRec-CE, BERT4Rec, the capacity-matched E2 attention variants, and FENRec F0-F5. Neural models share score/checkpoint/evaluation interfaces and objective-aware training. Architectural adaptations and source-code decisions are explicit in the method audit. TIGER remains Phase 6.

Training supports full-catalogue validation, validation-selected best weights, patience-based early stopping, separate latest checkpoints, epoch-boundary RNG/optimizer/scaler resume, deterministic seeds, optional Phase 3 CUDA AMP, configurable batches, finite-gradient checks, GPU memory monitoring and complete held-out predictions. FENRec runs FP32, includes both coarse/fine CL branches, epoch-wise training-only clustering, time-decayed labels, detached mixed hard negatives and adversarial noise negatives.

## Install and verify

Use Python 3.10 or 3.11. The validated local environment is Python 3.10.11, PyTorch 2.5.1+cu121, and a Quadro T1000 with 4 GiB. CPU and CUDA forward/backward execution both passed.

```powershell
python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
python scripts/check_environment.py
python scripts/run_smoke_test.py --device cpu
python scripts/run_smoke_test.py --device cuda
```

Without CUDA, use `--device cpu`. CUDA tests explicitly skip when unavailable; CPU development and training remain supported. Source-equivalence tests need the pinned FENRec checkout described in [validation notes](docs/PHASE34_VALIDATION.md).

## Real data and pilots

```powershell
python scripts/download_data.py --dataset beauty
python scripts/preprocess_data.py --dataset beauty
```

Use the printed bundle path. On this checkout:

```powershell
$bundle = 'data/processed/beauty/298d55546282277e0e2ba77a752e8be10e9067b1a05fee87ab09b6a5bffed6da'
python scripts/verify_data.py --bundle $bundle --output results/reports/beauty_data_audit.json
python scripts/run_pilots.py --bundle $bundle --device cuda:0
```

Beauty has 22,363 users, 12,101 items and 198,502 events after filtering. All models use the same hashed split. Real pilot artifacts live under `results/pilots`; summaries derive from verified artifacts and are excluded from final research tables. Pilots deliberately limit training to two epochs/twenty batches and select checkpoints on 512 validation users, then save all test users. They are not converged comparisons.

FENRec's default twenty-epoch warm-up is preserved in the main pilot. Its separate post-warm-up branch-verification pilot explicitly changes warm_up_epochs to zero:

```powershell
python scripts/run_train.py --bundle $bundle --model fenrec --seed 7 --device cuda:0 --experiment PILOT --results-root results/pilots_postwarm --config configs/experiments/fenrec_postwarm_pilot.yaml
```

## Select experiments

```powershell
python scripts/run_experiment.py --experiment all --dry-run --output results/tables/experiment_manifest_phase34.csv
python scripts/run_experiment.py --experiment E1 --dataset beauty --model sasrec_ce --seed 7 --bundle $bundle --device cuda:0 --config configs/experiments/full.yaml
python scripts/run_experiment.py --experiment E2 --dataset beauty --variant A3 --seed 7 --bundle $bundle --device cuda:0 --config configs/experiments/full.yaml
python scripts/run_experiment.py --experiment E3 --dataset beauty --model fenrec --retention 0.4 --seed 7 --bundle $bundle --device cuda:0 --config configs/experiments/full.yaml
python scripts/run_experiment.py --experiment E4 --dataset beauty --variant F5 --retention 0.4 --seed 7 --bundle $bundle --device cuda:0 --config configs/experiments/full.yaml
python scripts/run_experiment.py --experiment E5 --dataset beauty --model bert4rec --max-seq-length 100 --seed 7 --bundle $bundle --device cuda:0 --config configs/experiments/full.yaml
```

Repeat a command with `--resume` to restore the latest complete epoch. Verified completed runs skip automatically. Failed/incomplete attempts and prior status records are preserved. Changing a training setting creates a new hash; use `--set KEY=VALUE` for overrides. Device index can be selected via `cuda:0`/`cuda:1`. E1 requires selecting an implemented model because TIGER remains gated. No full grid was launched.

E2 uses a shared post-norm backbone across A1-A4 so it does not reuse original SASRec E1 checkpoints. E3 keeps validation/test fixed and records retention events, realized ratios and common warm-target users. Exact E1 settings can be reused by E3/E5 with provenance. Full E1-E5 comparisons still require controlled validation-only tuning and three seeds; bounded validation-only tuning is available, while final ablation tables/figures require completed research runs.

## Verify and export

```powershell
python scripts/verify_results.py --root results/pilots
python scripts/verify_results.py --root results/runs
python scripts/aggregate_results.py
python scripts/estimate_budget.py --root results/pilots
python scripts/export_results.py --root results --output results-export-attempt1.zip
```

SUCCESS certifies complete, consistent artifacts rather than convergence. Training-limit exhaustion is flagged in efficiency.json. Neural checkpoint.pt stores best weights/configuration and latest.pt stores resume state; MostPopular retains its safe NPZ checkpoint format. Full-catalogue ranking excludes padding, masks seen history items and preserves the target exception. All targets, including cold items, remain in evaluation.

For Kaggle environment setup, attached datasets, individual selectors, resume and artifact download, follow [KAGGLE_WORKFLOW.md](docs/KAGGLE_WORKFLOW.md). The same Python CLIs run on Windows PowerShell, Linux, Colab and Kaggle; Kaggle notebooks provide setup, training and analysis orchestration without duplicating models.


## Reference papers

- [GRU4Rec](https://arxiv.org/abs/1511.06939): the implemented CE user-history adaptation is documented explicitly.
- [SASRec](https://arxiv.org/abs/1808.09781): original BCE and separately named CE objectives.
- [BERT4Rec](https://arxiv.org/abs/1904.06690): bidirectional masked-item learning.
- [FENRec](https://arxiv.org/abs/2412.11589): soft labels and enduring hard-negative contrastive learning.
- [TIGER](https://arxiv.org/abs/2305.05065): planned, not implemented.

See [METHOD_REFERENCES.md](docs/METHOD_REFERENCES.md) for pinned source URLs, mathematical/source comparisons and every implementation deviation. Third-party reference code is downloaded separately, remains untracked, and retains its upstream licensing terms.
