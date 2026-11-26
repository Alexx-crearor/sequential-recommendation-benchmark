# Kaggle guide: Phase 5A

This workflow prepares and runs selected jobs; it does not assume previous Kaggle working files persist. Notebooks contain setup/CLI calls only. The source grid and completed local pilots are preserved.

## 1. Package and upload

From the local repository in PowerShell:

```powershell
python scripts/kaggle_session.py package --archive seqrec-kaggle-source.zip --include-data
```

The ZIP includes source, configurations, tests, notebooks, docs, frozen benchmark code and processed data. It excludes raw downloads, old result trees and reference repository clones. Omitting `--include-data` produces a smaller code-only package; Kaggle can download/preprocess datasets separately. Choose a fresh ZIP filename on subsequent packaging attempts.

Create a private Kaggle Dataset and upload the ZIP. Create a Kaggle Notebook and attach that dataset using the notebook's data/input control. Upload/import `notebooks/Kaggle_Setup.ipynb` as the notebook, select a GPU accelerator in its session settings, and enable Internet for installation. See [official Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks) for current notebook/data/session controls. Resource quotas depend on the account/session and are not assumed here.

## 2. Configure input and working directories

In the setup notebook, set `SOURCE_ARCHIVE` to the actual attached ZIP path. Default roots are `/kaggle/input` and `/kaggle/working`. Override `SEQREC_INPUT_DIR`, `SEQREC_WORK_DIR`, `SEQREC_REPO_DIR` and `SEQREC_PYTHON` if needed. The source package expands into a writable working repository. Re-running extraction into an existing repository intentionally stops instead of overwriting it; reuse the existing repository or choose a fresh directory.

Run setup cells in order. They create Python 3.11 via uv if the kernel is outside 3.10/3.11, install the validated dependencies, detect GPU execution, run CPU/CUDA smoke checks, locate or build the dataset bundle and verify its hashes/leakage boundaries. Direct CLI setup:

```bash
python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
python scripts/check_environment.py
python scripts/run_smoke_test.py --device cpu
python scripts/run_smoke_test.py --device cuda
```

Use the separate environment's Python executable when setup created one. CUDA requests do not silently fall back to CPU; select `--device cpu` explicitly when needed. Source-comparison tests skip visibly if the optional pinned reference checkout is absent; portable mathematical, leakage, resume and ranking tests still run.

## 3. Prepare data once

If processed Beauty is included, select its exact bundle. If using an attached processed bundle, it may remain under the read-only input directory because training writes only to the chosen results root. Otherwise:

```bash
python scripts/download_data.py --dataset beauty
python scripts/preprocess_data.py --dataset beauty
python scripts/verify_data.py --bundle data/processed/beauty/HASH --output results/reports/kaggle_data_audit.json
```

Use the exact printed `HASH`; never substitute another model's split. For raw attached files pass preprocessing's `--interactions`/`--metadata` options. Full Research E1 needs separate Sports and Toys preparation. Setup stores `seqrec_session.json` in the working directory, containing repository/Python/bundle paths for the training and analysis notebooks. Paths must be reselected if an input dataset's mount name changes in a later session.

## 4. Review the selected budget

Open/import `Kaggle_Training.ipynb` in the same session or copy its thin orchestration cells after setup. For a separate notebook/session, attach the same source/data and restore prior results first. Set `PROFILE`, `EXPERIMENT`, `MODEL`, `VARIANT`, `SEED`, `RETENTION`, `MAX_SEQ_LENGTH`, `DEVICE` and `RESULTS_ROOT` in the selection cell. The default `MODE='dry-run'` launches no training.

Profiles:

- `pilot`: short, explicitly tagged pipeline checks; never final research evidence.
- `core_research`: Beauty, implemented E1 models, E2, E3 ratios 0.4/0.7/1.0, E4, E5 lengths 10/20/50, E7/E8; initially seed 7.
- `full_research`: preserves the original E1-E8 axes, Sports/Toys and future TIGER entries. TIGER/E6 execution remains gated until implemented.

```bash
python scripts/run_experiment.py --profile core_research --experiment all --dry-run --output results/tables/core_initial_manifest.csv
python scripts/run_experiment.py --profile core_research --final-seeds --experiment all --dry-run --output results/tables/core_final_manifest.csv
python scripts/run_experiment.py --profile full_research --experiment all --dry-run --output results/tables/full_manifest.csv
```

Initial Core has 45 entries including two analysis tasks; final Core has 129; Full has 209. Exact E1 checkpoint reuse reduces training work. These are counts, not measured runtime. Read `EXPERIMENT_BUDGET.md` before scheduling jobs. Do a Kaggle-specific controlled benchmark with a new output directory:

```bash
python scripts/benchmark_pipeline.py --bundle data/processed/beauty/HASH --device cuda:0 --batch-size 128 --batches 20 --repeats 2 --cluster-benchmark --full-validation --output results/phase5a/kaggle_control_b128
```

## 5. Tune on validation only

Set `MODE='tune'` for one model, or run:

```bash
python scripts/run_tuning.py --profile core_research --bundle data/processed/beauty/HASH --model sasrec_ce --device cuda:0 --trials 6 --results-root results/tuning --resume
```

Candidates include the model defaults first, then deterministic bounded samples of the profile search space. Learning rate, physical batch and dropout are searched; BERT adds mask ratio, FENRec adds decay and CL weights. Seed is fixed to 7. At most twelve trials are permitted. Maximum epochs/patience/cadence are model-specific; FENRec retains its twenty-epoch warm-up and minimum post-warm-up budget.

Trials produce `VALIDATED`, `validation_selection.json`, `validation_predictions.parquet` and a `selected_config.yaml`, never test scores/predictions. Use the selected YAML for final seeds. A pilot-only tuning selection is rejected by research execution. Selection is never based on the old Beauty pilot test scores. The completed Phase 5C Kaggle export contains three trials per model (15 total); its verified selections are now frozen in `configs/selected/phase5c/beauty/`. Do not rerun that batch merely to prepare the next phase. See [the Vietnamese analysis](PHASE5C_ANALYSIS_VI.md) for actual values and stopping-budget limits. This is a bounded batch, not an exhaustive search.

For E2 choose one backbone hyperparameter setting and keep it fixed across A1-A4. For E4 reuse the selected FENRec setting across all variants. Do not tune each ablation on test outcomes.

## 6. Run individual research configurations

Replace `HASH` and `SELECTION.yaml` with actual paths. Dropping `--seed` with `--final-seeds` deliberately expands to all three final seeds; otherwise initial Core uses only seed 7.

```bash
# E1 selected model; repeat for gru4rec_ce, sasrec_bce, bert4rec, fenrec, mostpopular.
python scripts/run_experiment.py --profile core_research --final-seeds --experiment E1 --dataset beauty --model sasrec_ce --seed 7 --bundle data/processed/beauty/HASH --device cuda:0 --selection SELECTION.yaml --results-root results/core --resume

python scripts/run_experiment.py --profile core_research --experiment E2 --dataset beauty --variant A3 --seed 7 --bundle data/processed/beauty/HASH --device cuda:0 --results-root results/core --resume
python scripts/run_experiment.py --profile core_research --experiment E3 --dataset beauty --model sasrec_ce --retention 0.7 --seed 7 --bundle data/processed/beauty/HASH --device cuda:0 --selection SELECTION.yaml --results-root results/core --resume
python scripts/run_experiment.py --profile core_research --experiment E4 --dataset beauty --variant F3 --retention 0.4 --seed 7 --bundle data/processed/beauty/HASH --device cuda:0 --results-root results/core --resume
python scripts/run_experiment.py --profile core_research --experiment E5 --dataset beauty --model bert4rec --max-seq-length 20 --seed 7 --bundle data/processed/beauty/HASH --device cuda:0 --results-root results/core --resume
```

Use a model-matching selection file for each model. MostPopular requires no tuning and always executes once on CPU. E2 all four variants use matched capacity and corruption settings. E3 fixed held-outs and profile-specific common warm-target users remain intact. No sampled evaluation is used.

Safe knobs: `--set batch_size=64`, `--set eval_batch_size=128`, `--set num_workers=2`, `--set validation_every=2`, `--set checkpoint_every=1`, `--set activation_checkpoint_ce=true`. Independent CE/BCE objectives support `--set gradient_accumulation_steps=2`, with target-count weighting and a partial final group handled correctly. FENRec F2-F5 reject accumulation because splitting their negative pool changes their objective. Lowering the physical FENRec batch changes that research setting and must be matched across variants. Phase 3 AMP is opt-in (`--set amp=true`); FENRec remains FP32. Do not assume mixed precision improves a Turing mobile GPU; measure on Kaggle hardware first.

## 7. Resume across sessions

During the session, training writes epoch histories, best `checkpoint.pt`, resumable `latest.pt`, memory and timing records. Default checkpoint cadence is every epoch. With a longer cadence an interruption may lose more than one epoch. Partial epochs replay from the last complete saved epoch. Latest includes the corresponding best state, optimizer/scaler and Python/NumPy/PyTorch/CUDA RNGs. Different hardware/software can change floating-point trajectories; exact resume tests use the same environment.

Before the Kaggle session expires:

```bash
python scripts/verify_results.py --root results/core
python scripts/export_results.py --root results --output /kaggle/working/seqrec-results-session1.zip
```

A deliberately stopped or interrupted run is INCOMPLETE and the verifier reports that state; preserve/export it anyway. Save the notebook version with outputs and download the ZIP (and `seqrec_session.json`), or upload the ZIP as a private Kaggle Dataset. Do not rely on `/kaggle/working` persistence.

In a new session: attach the source/data and prior result ZIP; re-run setup; set `PRIOR_RESULTS_ARCHIVE` and restore, or run:

```bash
python scripts/kaggle_session.py restore --archive /kaggle/input/prior-results/seqrec-results-session1.zip --destination /kaggle/working/sequential-recommendation
```

Restore rejects traversal and conflicting existing files, and skips byte-identical files. Use a new working directory if a different artifact tree already exists. Reissue the exact training/tuning command with `--resume`, including the same profile, selected hyperparameters and results root. Verified completed SUCCESS/VALIDATED configurations skip automatically. Changing hashed settings creates a new run instead of silently extending a completed budget.

Parallel Lloyd clustering can differ at floating-point round-off even with the same seed. Small deterministic resume fixtures pass exactly; large FENRec training is not claimed bitwise identical across independent clustering fits or hardware. A same-checkpoint ranking audit distinguishes those fit differences from evaluator correctness.

## 8. Verify, analyze and export

Perform training, testing, evaluation and synchronized GPU measurements on Kaggle; export the artifacts and perform statistical analysis, visualization, E8 interpretation and publication on the personal machine. The older `Kaggle_Analysis.ipynb` is retained for compatibility; use its verification/export steps on Kaggle, not its analysis/visualization cells. E7/E8 never train or choose a configuration by test score. GPU measurement E7 may run on Kaggle:

```bash
python scripts/run_experiment.py --profile core_research --experiment E7 --dataset beauty --bundle data/processed/beauty/HASH --device cuda:0 --results-root results/core
python scripts/export_results.py --root results --output /kaggle/working/seqrec-results-session2.zip
```

After restoring exported results and the matching local dataset on the personal machine:

```bash
python scripts/run_experiment.py --profile core_research --experiment E8 --dataset beauty --bundle data/processed/beauty/HASH --results-root results/core
python scripts/aggregate_results.py --root results/core --output results/tables/core
```

Phase 5C analysis is exclusively local: `python scripts/analyze_phase5c.py --archive PATH_TO_ORIGINAL_ZIP`, or `notebooks/Phase5C_Analysis.ipynb`. The source ZIP stays private and raw artifacts are not committed. Next-phase execution can use `--selection configs/selected/phase5c/beauty/MODEL.yaml`; each model must receive its matching selection file and original split.

Use an E1 results root containing only preselected final configurations. E8 rejects multiple configurations for the same model/seed instead of choosing by test scores. Pilot/tuning roots remain separate. Final scientific figures, paired multi-seed analyses and the Vietnamese research report are later deliverables, not inferred from short performance benchmarks.
