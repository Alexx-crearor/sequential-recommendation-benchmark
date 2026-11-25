# Git publication and clean reconstruction

The publication preparation found no existing `.git` directory or remote. A new local `main` repository is prepared with the requested initial commit and annotated tag `v0.1.0-phase4`. The snapshot includes subsequent Phase 5A tooling as documented in README; the requested commit/tag names identify the completed Phase 1-4 foundation. No remote push is performed.

## Included and excluded content

Versioned: `src/`, `scripts/`, `configs/`, `tests/`, `notebooks/`, `docs/`, frozen original benchmark hot paths under `benchmarks/`, README, implementation plan, dependency files, Git settings, and selected lightweight pilot/benchmark CSV/JSON summaries and manifests. Figures under `results/figures/` and tables under `results/tables/` are allowed subject to the publication size audit.

Excluded, preserved locally: the complete `data/` and `references/` trees, generated run directories, checkpoints, per-user predictions/evaluation-user files, training histories/logs, traces, intermediate caches, Python environments, credentials and portable ZIP exports. Existing old pilots and post-warm-up runs are not removed. `.gitignore` does not remove already tracked files: future accidental additions must be reviewed with `git ls-files` and untracked with `git rm --cached PATH` when appropriate, never by deleting the local artifact.

The result whitelist preserves `results/pilots/pilot_summary.csv`, selected controlled Phase 5A measurement/correctness files, full-epoch measurements and budget manifests. Full local evidence is deliberately not reconstructible from Git alone: regenerate it with the benchmark commands or restore a separately exported results ZIP. Summary tables are explicitly pilot/runtime evidence, not final scientific comparisons.

## Audit before every publication

```bash
git status --short
git add .
python scripts/audit_publication.py --output results/publication/staged_audit.json
git diff --cached --stat
git diff --cached --check
```

The audit reads actual index blobs, rejecting files above 5 MiB, Git submodule/gitlink entries, dataset/checkpoint/prediction paths and common literal credential patterns without printing credential values. This is a bounded heuristic scan, not proof that every possible secret format is absent. Review newly staged files, especially notebooks and figures, before publication. Audit reports and test logs stay ignored in `results/publication/`.

## Reconstruct a clean clone

```bash
git clone CONFIRMED_GITHUB_URL sequential-recommendation
cd sequential-recommendation
git checkout v0.1.0-phase4
python -m venv .venv
# Activate .venv, then:
python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
python scripts/fetch_references.py
python -m pytest -q
python scripts/check_environment.py
python scripts/download_data.py --dataset beauty
python scripts/preprocess_data.py --dataset beauty
python scripts/verify_data.py --bundle data/processed/beauty/HASH --output results/reports/beauty_data_audit.json
```

Python 3.10/3.11 is required. Replace HASH with preprocessing's printed content-hash directory. CPU execution is supported; CUDA tests skip visibly on hosts without CUDA. Reference source-comparison tests skip if optional clones are omitted. Fetching references restores those tests without access to local datasets/checkpoints. Locked dependency installation needs package network access; reference/data downloads need their upstream services. A new environment cannot restore unpublished local training state from the Git tag.

Use [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md) and the three `notebooks/Kaggle_*.ipynb` notebooks for configurable input/working paths, setup, selected experiment commands, checkpoint restoration across sessions and artifact export. Package a fresh ZIP with `python scripts/kaggle_session.py package --archive seqrec-github-kaggle-source.zip`; add `--include-data` only for a separate private Kaggle upload, never a Git commit. Existing local ZIPs are snapshots and are not refreshed implicitly.

## Push only after the GitHub URL is confirmed

The local commit command is:

```bash
git commit -m "Complete Phase 1-4: benchmark foundation, neural models and pilot validation"
git tag -a v0.1.0-phase4 -m "Phase 1-4 benchmark foundation and pilot validation"
```

These are performed during preparation; do not repeat them. After confirming the actual empty GitHub repository URL, use:

```bash
git remote add origin CONFIRMED_GITHUB_URL
git remote -v
git push -u origin main
git push origin v0.1.0-phase4
```

If an origin already exists then inspect it first; change it with `git remote set-url origin CONFIRMED_GITHUB_URL` only after confirming the replacement. If the remote already contains commits, inspect and reconcile them before pushing. No force push is required or recommended by this workflow.
