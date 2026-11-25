# Phase 3/4 file inventory

New model/data modules:

- `src/models/base.py`, `factory.py`, `gru4rec.py`, `sasrec.py`, `bert4rec.py`, `fenrec.py`
- `src/data/neural_dataset.py`
- `src/training/__init__.py`, `checkpoint.py`, `losses.py`, `trainer.py`
- `src/evaluation/neural_ranking.py`, `efficiency.py`
- `src/utils/configuration.py`

Extended working Phase 2 modules:

- `src/data/splitting.py`: vectorized chronological integrity check, same split semantics
- `src/evaluation/evaluate.py`: neural evaluation dispatch
- `src/utils/runs.py`: neural checkpoint validation, finite histories, full user/target coverage and common-warm metric checks
- `src/utils/grid.py`: remove invalid E2-to-E1 aliases after matching the E2 backbone

New execution/verification scripts:

- `scripts/check_environment.py`, `verify_data.py`, `run_pilots.py`
- `scripts/estimate_budget.py`, `export_results.py`, `generate_phase34_report.py`

Extended scripts:

- `scripts/run_train.py`, `run_experiment.py`, `run_evaluate.py`, `run_smoke_test.py`, `aggregate_results.py`

New configurations/dependency files:

- `configs/models/{gru4rec_ce,sasrec_bce,sasrec_ce,bert4rec,attention_ablation,fenrec}.yaml`
- `configs/experiments/{pilot,fenrec_postwarm_pilot,full}.yaml`
- `requirements-neural.txt`, `requirements-lock.txt`
- Modified `configs/experiments/experiment_grid.yaml`, `pyproject.toml`, `.gitignore`

New tests:

- `tests/test_neural.py`, `test_fenrec.py`, `test_training.py`
- `tests/test_reference_backbone.py`, `test_early_stopping.py`, `test_experiment_reuse.py`

Documentation:

- New `docs/KAGGLE_WORKFLOW.md`, `PHASE34_VALIDATION.md`, `PHASE34_RESULTS.md`, `PHASE34_FILES.md`
- Updated `README.md`, `IMPLEMENTATION_PLAN.md`, `docs/EXPERIMENT_PROTOCOL.md`, `METHOD_REFERENCES.md`, `DATASET_REPORT.md`

Generated evidence:

- Beauty raw archives/provenance and immutable processed bundle under `data/`
- Six main completed pilots under `results/pilots/`, one supplemental F5 post-warm-up pilot under `results/pilots_postwarm/`
- `results/pilots/pilot_summary.csv`
- Environment/data audit JSON, JUnit XML, budget and supplemental FENRec summaries under `results/reports/`
- `results/tables/experiment_manifest_phase34.csv`
- Added a pinned read-only original GRU4Rec checkout under ignored `references/GRU4Rec/`

Existing completed artifacts were preserved. No final experiment grid or final research scores were generated.
