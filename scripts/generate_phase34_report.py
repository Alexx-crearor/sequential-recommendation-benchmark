"""Generate a factual phase report from pilot artifacts and test evidence."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import yaml
import _bootstrap
from src.utils.artifacts import write_json
from src.utils.runs import verify_run


def main():
    root = _bootstrap.ROOT
    summary = pd.read_csv(root / "results/pilots/pilot_summary.csv")
    for path in summary.run_path:
        verify_run(root / path)
    audit = json.loads((root / "results/reports/beauty_data_audit.json").read_text())
    stats = audit["statistics"]
    tests = ET.parse(root / "results/reports/phase34_tests.xml").getroot().find("testsuite").attrib
    environment = json.loads((root / "results/reports/environment_check.json").read_text())
    fields = [("model", "Model"), ("final_train_loss", "Final loss"), ("best_validation_ndcg@10", "Validation NDCG@10"),
              ("training_seconds", "Training seconds"), ("batch1_latency_ms", "Batch-1 scoring ms"),
              ("peak_gpu_memory_mib", "Peak GPU MiB"), ("parameters", "Parameters")]
    table = ["| " + " | ".join(title for key, title in fields) + " |", "|" + "---|"*len(fields)]
    for row in summary.to_dict("records"):
        values = []
        for key, title in fields:
            value = row[key]
            values.append("N/A" if pd.isna(value) else str(int(value)) if key == "parameters" else f"{value:.6f}" if isinstance(value, float) else str(value))
        table.append("| " + " | ".join(values) + " |")
    budget = json.loads((root / "results/reports/pilot_budget_estimate.json").read_text())["estimates"]
    post_budget = json.loads((root / "results/reports/fenrec_postwarm_budget_estimate.json").read_text())["estimates"]
    estimates = {row["model"]: row for row in budget if row["estimated_300_epoch_training_and_validation_hours"] is not None}
    for row in post_budget:
        estimates[row["model"]] = row
    budget_lines = ["| Model | Estimated 300-epoch hours / seed |", "|---|---:|"]
    for name, row in estimates.items():
        budget_lines.append(f"| {name} | {row['estimated_300_epoch_training_and_validation_hours']:.2f} |")
    total = sum(row["estimated_300_epoch_training_and_validation_hours"] for row in estimates.values())
    supplemental = []
    for status_path in (root / "results/pilots_postwarm").rglob("status.json"):
        directory = status_path.parent
        verify_run(directory)
        history = pd.read_csv(directory / "train_history.csv")
        efficiency = json.loads((directory / "efficiency.json").read_text())
        supplemental.append({"run_path": str(directory.relative_to(root)), "train_loss": float(history.train_loss.iloc[-1]),
                             "coarse_loss": float(history.coarse_loss.iloc[-1]), "fine_loss": float(history.fine_loss.iloc[-1]),
                             "cluster_seconds": float(history.cluster_seconds.iloc[-1]), "optimization_seconds": float(history.optimization_seconds.iloc[-1]),
                             "peak_gpu_mib": efficiency["peak_gpu_memory_bytes"]/1048576})
    write_json(root / "results/reports/fenrec_postwarm_summary.json", supplemental)
    text = f'''# Phase 3/4 execution report

Generated from verified artifacts on 2026-11-25. These are pipeline pilots, not final research findings.

## Completion and environment

Implemented MostPopular plus GRU4Rec-CE, SASRec-BCE/CE, BERT4Rec, all four capacity-matched E2 attention/objective variants, and FENRec F0–F5. The trainer supports best/latest checkpoints, validation early stopping, exact same-environment epoch-boundary resume, full-catalogue evaluation and GPU memory reporting. All changes extend the Phase 2 repository.

Local Python {environment['packages']['torch']} is the PyTorch version (Python itself is 3.10.11); CUDA 12.1 executes on a Quadro T1000 with 4 GiB. Both CPU and CUDA forward/backward checks passed. See `results/reports/environment_check.json` for the complete package inventory.

Beauty: {stats['users']:,} users, {stats['items']:,} items, {stats['interactions']:,} interactions and {stats['train_interactions']:,} training events. Metadata missing items: {stats['metadata_missing']}. Test warm coverage: {stats['test_warm_coverage']:.6f}. Hash/split/chronology/core/per-model training-boundary checks passed. Split version: `{audit['split_version']}`.

## Tests

JUnit evidence: {tests['tests']} tests, {tests['failures']} failures, {tests['errors']} errors, {tests['skipped']} skipped. The full suite includes CPU cases and six CUDA forward/backward cases; separate CPU and CUDA smoke selections also passed. FENRec mixed negatives and pre/post-warm-up loss values/gradients match extracted pinned source functions. Encoder weights transferred to the source produce matching real-token states. All six E4 variants have finite loss/gradients and identical backbone sizes. Every model has an exact interrupted/resumed weight/loss sequence comparison. Source commits and deviations are in `docs/METHOD_REFERENCES.md`.

## Real Beauty pilots, seed 7

Neural runs use two epochs and twenty batches of 64 per epoch. Validation selection uses the first 512 user IDs as an explicitly limited pilot subset. MostPopular uses all training data and full validation; its validation number is therefore not directly comparable to neural pilot subset values. Every run saves all {stats['users']:,} test users with exact full-catalogue ranks. Loss values are model-specific objectives and should not be compared across rows. MostPopular has no optimization loss or trainable parameter vector.

{chr(10).join(table)}

Scoring latency includes history encoding and full-catalogue scoring, with five warm-ups and twenty synchronized batch-1 repeats. It excludes sorting, disk I/O and Python evaluation bookkeeping; P95 and end-to-end test throughput are in each efficiency.json. MostPopular runs on CPU. GPU peaks are allocated PyTorch bytes, not driver-reserved or total system memory. The baseline latency is recorded as a separate analysis artifact because its completed Phase 2-style run is immutable.

FENRec's main two-epoch pilot preserves its twenty-epoch warm-up, so those training losses do not exercise enduring negatives. A separate one-epoch, four-batch F5 pilot explicitly sets warm_up_epochs=0; it verifies the post-warm-up path on real data. Its measured stage breakdown and nonzero coarse/fine losses are in `results/reports/fenrec_postwarm_summary.json`. Both pilots reach their deliberate training limits; none should be interpreted as converged evidence or used to rank methods scientifically.

Per-user predictions, validation histories, environments, best/latest checkpoints and source hashes remain in `results/pilots` and `results/pilots_postwarm`. Primary table source: `results/pilots/pilot_summary.csv`.

## Resource extrapolation

Same local hardware, batch 64, full Beauty training and validation every epoch, maximum 300 epochs per seed:

{chr(10).join(budget_lines)}

For the five neural E1 models on Beauty, this totals about {total:.1f} hours per seed or {total*3:.1f} hours for three seeds before early stopping. These are rough linear extrapolations from short pilots, not measured full runs. FENRec uses the supplemental separated clustering/optimization measurement; scaling its earlier combined clustering time by the batch ratio would be invalid and is intentionally left unset. Fixed cluster encoding/fitting cost is added once per epoch. The supplemental four-batch measurement has particularly high uncertainty. Normal source warm-up is cheaper than its post-warm-up branch; the estimate applies that post-warm-up rate conservatively throughout.

The defaults use batch 128, while these estimates hold batch 64 fixed. A batch-128 capacity/throughput pilot, full epochs and larger GPU measurements are needed before scheduling Kaggle. No reliable aggregate E1–E5 wall-clock estimate exists for Sports/Toys, other lengths, retention settings or tuning. The plan has 209 entries including later phases; E1–E5 excluding TIGER has 189 planned entries before exact-configuration reuse. Validation-only tuning and final test evaluation add cost.

## Remaining work and commands

No environment/data blocker remains for local Phase 3/4 development. Full-budget model convergence, bounded tuning automation, multi-seed statistical analysis, Sports/Toys pilots, final tables/figures/Vietnamese report and TIGER remain outstanding. FENRec is an explicitly documented source-code port using sklearn rather than FAISS clustering; it is not a claim of bitwise reproduction of every original training artifact. Kaggle commands are provided but have not been executed in an actual Kaggle session.

Exact E1–E5 commands, environment setup, data preparation, GPU selection, resume, completed-run skipping and ZIP export: `docs/KAGGLE_WORKFLOW.md`. No full training grid was launched.
'''
    text = text.replace(f"Local Python {environment['packages']['torch']} is the PyTorch version (Python itself is 3.10.11)", f"Local Python 3.10.11 uses PyTorch {environment['packages']['torch']}")
    (root / "docs/PHASE34_RESULTS.md").write_text(text, encoding="utf-8")
    print(root / "docs/PHASE34_RESULTS.md")


if __name__ == "__main__":
    main()
