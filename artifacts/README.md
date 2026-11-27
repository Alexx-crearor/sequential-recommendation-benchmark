# Local artifact downloads

## Official master session 1

Original private ZIP: `artifacts/master_session1/seqrec-master-results-20261126_234725_576013.zip`, SHA256 `9f6b30dd3d29750b21ef64ddd7c8fea88c66d1c3ca52ea44a9809b3cc323cfab`. The repository owner's copy is preserved here, alongside its untouched external original. Run `python scripts/analyze_master_session1.py` locally with the matching Beauty bundle to re-audit and analyze; public-summary notebook mode does not require the private raw ZIP.

Public prediction-free [official checkpoint Release](https://github.com/Alexx-crearor/sequential-recommendation-benchmark/releases/tag/master-session1-checkpoints-v1): `python scripts/fetch_master_checkpoints.py` downloads all 22 completed runs' best weights/configs; `--with-resume` adds all 21 neural latest states (43 checkpoints total). Bundles cache in `artifacts/master_session1/` and extract into `checkpoints/master_session1/`. The original ZIP, private predictions/logs and datasets are never published as Release assets. Restore that private full ZIP for the original result-tree continuation; checkpoint-only bundles are not full run exports.

## Phase 5C

The original private Phase 5C archive belongs at:

```text
artifacts/phase5c/seqrec-phase5c-final-20261126_132556_321898.zip
```

This is the default path for `python scripts/analyze_phase5c.py` and the analysis notebook, relative to the repository root. No script assumes another user's Downloads directory. Explicit `--archive` and `SEQREC_PHASE5C_ARCHIVE` overrides are supported.

The repository owner's original archive is copied here without modifying its external original. It contains private per-user validation artifacts, so it is not published. Public, prediction-free checkpoint bundles are distributed separately through [GitHub Releases](https://github.com/Alexx-crearor/sequential-recommendation-benchmark/releases/tag/phase5c-checkpoints-v1).

After a clean clone:

```bash
python scripts/fetch_phase5c_checkpoints.py
# Optional: every trial's best/latest checkpoint instead of selected best weights only.
python scripts/fetch_phase5c_checkpoints.py --all-trials
```

Downloads are cached in this directory, verified against the versioned manifest, and extracted into `checkpoints/phase5c/`. Binary files remain excluded from Git history. Checkpoint bundles do not replace the original private ZIP when auditing all original predictions/logs.
