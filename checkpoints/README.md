# Phase 5C checkpoints

The original 30 best/latest checkpoints are materialized locally as:

```text
checkpoints/phase5c/beauty/<model>/trials/<0|1|2>/checkpoint.pt
checkpoints/phase5c/beauty/<model>/trials/<0|1|2>/latest.pt
checkpoints/phase5c/beauty/<model>/trials/<0|1|2>/config.yaml
checkpoints/phase5c/beauty/<model>/selected_config.yaml
```

These binary files are downloaded from [the checkpoint Release](https://github.com/Alexx-crearor/sequential-recommendation-benchmark/releases/tag/phase5c-checkpoints-v1), not stored as large Git blobs. A normal Git clone does not fetch Release assets automatically. Run:

```bash
python scripts/fetch_phase5c_checkpoints.py
```

This downloads the five selected best checkpoints and their configurations (about 15 MiB). Add `--all-trials` for all 15 trials, including 30 best/latest checkpoints (about 226 MiB). The downloader checks bundle and per-file SHA256 values, skips verified existing files and rejects conflicting files without overwriting them. It uses Python's standard library; benchmark dependencies are required only for loading models.

The selected trials are SASRec-CE T2, SASRec-BCE T0, GRU4Rec-CE T1, BERT4Rec T1 and FENRec T2. Resolve the path from the versioned manifest rather than from an owner's machine:

```python
import json
from pathlib import Path
from src.models.base import SequentialModel

manifest = json.loads(Path("configs/artifacts/phase5c_checkpoints.json").read_text())
checkpoint = Path(manifest["models"]["fenrec"]["checkpoint"])
model = SequentialModel.load(checkpoint, device="cpu")
```

All weights are from validation-only tuning at seed 7; they are not final multi-seed/test models. `latest.pt` retains optimizer/scaler/RNG and epoch-level histories. The public packages omit raw predictions, per-user histories, datasets and logs. They are not a complete original run directory; verified full-run reconstruction requires the private original ZIP. No checkpoint is retrained, rewritten or converted for distribution.
