# Local artifact downloads

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
