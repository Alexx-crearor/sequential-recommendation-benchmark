"""Build the Phase 5A report from actual measured artifacts, not entered scores."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import torch
import _bootstrap
from src.utils.artifacts import write_json, file_hash


def main():
    root = _bootstrap.ROOT
    directory = root/"results/phase5a"
    benchmark = directory/"benchmark_isolated_fp32_b128"
    frame = pd.read_csv(benchmark/"measurements.csv")
    averages = frame.groupby(["model", "mode"]).mean(numeric_only=True)
    lines = ["| Model | Original batch s | Optimized batch s | Batch speedup | Original validation s | Optimized validation s |", "|---|---:|---:|---:|---:|---:|"]
    for name in ["gru4rec_ce", "sasrec_bce", "sasrec_ce", "bert4rec", "fenrec"]:
        before, after = averages.loc[(name,"original")], averages.loc[(name,"optimized")]
        lines.append(f"| {name} | {before.seconds_per_batch:.4f} | {after.seconds_per_batch:.4f} | {before.seconds_per_batch/after.seconds_per_batch:.2f}x | {before.validation_seconds:.3f} | {after.validation_seconds:.3f} |")
    full = pd.read_csv(directory/"full_epoch_b128/full_epoch_measurements.csv")
    full_lines = ["| Model | Complete training epoch s | Clustering s | Full validation s | Examples/s | Peak allocated MiB |", "|---|---:|---:|---:|---:|---:|"]
    for row in full.itertuples():
        full_lines.append(f"| {row.model} | {row.full_epoch_training_seconds:.2f} | {row.cluster_seconds:.2f} | {row.full_validation_seconds:.2f} | {row.examples_per_second:.0f} | {row.peak_gpu_memory_mib:.1f} |")
    equivalence = json.loads((directory/"same_checkpoint_equivalence.json").read_text())
    if any(row["rank_mismatches"] or row["top_k_mismatches"] or not row["metrics_equal"] for row in equivalence):
        raise ValueError("Ranking audit failed; cannot report validated optimization")
    tests = ET.parse(directory/"tests.xml").getroot().find("testsuite").attrib
    budget = json.loads((directory/"budget/profile_budget.json").read_text())["estimates"]
    budget_lines = ["| Profile | Entries | Nominal minimum-epoch floor h | Maximum known-Beauty h | Unknown entries |", "|---|---:|---:|---:|---:|"]
    for row in budget:
        budget_lines.append(f"| {row['profile']} | {row['entries']} | {row['known_beauty_budget_floor_hours']:.1f} | {row['known_beauty_max_budget_hours']:.1f} | {row['unknown_entries']} |")
    e1_hours = float(((full.full_epoch_training_seconds+full.cluster_seconds+full.full_validation_seconds)*300/3600).sum())
    centers_a = torch.load(benchmark/"original_centers.pt", weights_only=True)
    centers_b = torch.load(benchmark/"optimized_centers.pt", weights_only=True)
    center_difference = float((centers_a-centers_b).abs().max())
    amp = pd.read_csv(directory/"benchmark_amp_b128/measurements.csv")
    amp_lines = []
    for name in ("sasrec_ce", "bert4rec"):
        fp32 = averages.loc[(name,"optimized")].seconds_per_batch
        fp16 = amp.loc[(amp.model==name)&(amp["mode"]=="optimized"), "seconds_per_batch"].mean()
        amp_lines.append(f"- {name}: FP32 {fp32:.4f} s/batch; AMP {fp16:.4f} s/batch ({fp16/fp32:.2f}x the FP32 time).")
    samples = json.loads((benchmark/"gpu_samples.json").read_text())
    utilization = [row["utilization_percent"] for row in samples if row["gpu"]==0]
    audit = {"date": "2026-11-26", "original_grid_sha256": file_hash(root/"configs/experiments/experiment_grid.yaml"),
             "tests": tests, "same_checkpoint_ranking_users_per_model": 1024, "same_checkpoint_ranking_mismatches": 0,
             "independent_fenrec_centroid_max_abs_difference": center_difference, "measured_models": full.model.tolist(),
             "gpu_utilization_over_complete_benchmark_mean_percent": sum(utilization)/len(utilization) if utilization else None,
             "benchmark_source": str(benchmark.relative_to(root))}
    write_json(directory/"completion_audit.json", audit)
    text = f'''# Experiment budget and Phase 5A performance report

Generated 2026-11-26 from measured artifacts. No full research grid was launched. Prior pilot artifacts and the original experiment grid are preserved.

## Actual implementation status

MostPopular, GRU4Rec-CE, SASRec-BCE/CE, BERT4Rec, causal masked E2 and FENRec F0-F5 are operational. Architectural/source differences remain documented in METHOD_REFERENCES.md. Phase 5A adds exact device-resident ranking, cached train-only tensors, vectorized GRU/padding, safe independent-objective accumulation, optional CE activation checkpointing, configurable loaders/validation/checkpoint cadence, validation-only bounded tuning, three experiment profiles and Kaggle orchestration notebooks. TIGER is still a future implementation, retained in Full Research.

## Before/after controlled short benchmark

Quadro T1000 4 GiB; Python 3.10.11; PyTorch 2.5.1+cu121; FP32; batch 128; d=64; two layers; maximum length 50; seed 7. Two repeats of twenty measured batches follow one sacrificial warm-up update and a full model/optimizer/RNG reset. Validation rows below use 1,024 users and the complete 12,101-item catalogue. Original hot-path snapshots are under benchmarks/phase34. Both modes use the same data, initialization, objectives and precision; AMP is a separate experiment. Reported times are synchronized wall time, including data loading and transfer.

{chr(10).join(lines)}

Source: `results/phase5a/benchmark_isolated_fp32_b128/measurements.csv`. An earlier exploratory pass overlapped correctness work and is not the source of this table. Short timing variance and CPU load remain relevant; repeat on Kaggle before scheduling. GPU samples span the entire benchmark, including CPU preparation/clustering, so their average is not a training-only utilization claim.

## Complete optimized epoch measurements

One full training epoch per model, followed by full validation on all 22,363 Beauty users. FENRec uses epoch index 21 to measure post-warm-up compute; this is a runtime benchmark, not a convergence claim. Training seconds exclude the separately listed epoch-wise clustering. The other models have 175 batches; FENRec has 1,027. Peak memory includes the measured training/validation pass.

{chr(10).join(full_lines)}

Source: `results/phase5a/full_epoch_b128/full_epoch_measurements.csv`. FENRec clustering retains its exact configured backend/centroids/restarts/iteration budget; no infrequent refresh or approximate replacement was introduced. Its trace splits CPU Lloyd fitting and prefix encoding. Full-catalogue CE remains exact. Larger physical batches remain a model setting for contrastive comparisons.

## Bottlenecks and precision findings

The original cProfile recorded 1,535,373 calls and 2.875 seconds for 1,024-user validation, with about 1.203 seconds in repeated history construction. NumPy per-user masking/sorting and tiny CUDA history copies were prominent. The optimized path transfers only final ranks/top-K and processes catalogue chunks on-device. GRU per-user GPU control flow was the largest training bottleneck. Cached tensor construction costs more up front but amortizes across epochs. Full FENRec still performs three independent-dropout forward passes and full prefix clustering; collapsing those would change its objective and was not done.

Detailed CPU/CUDA operator traces are in `results/phase5a/trace_sasrec_ce` and `trace_fenrec`. Profiler instrumentation inflates timings; the benchmark table, not profiler absolute time, is used for cost estimates. Checkpoint write times, examples/s and memory are in the benchmark CSV. No OOM occurred at batch 128; memory was not the binding limit at that tested capacity. Worker/pinning settings remain configurable, with zero workers as the measured portable default.

AMP was not faster locally:

{chr(10).join(amp_lines)}

AMP remains opt-in for Phase 3 and disabled for FENRec. Reduced memory alone is not proof of speed. CE activation checkpointing is a separate opt-in memory/compute trade-off with gradient-equivalence tests.

## Numerical and leakage verification

{tests['tests']} tests passed with {tests['failures']} failures, {tests['errors']} errors and {tests['skipped']} skips in the recorded local suite. Tests include cache/transform equality, GRU states/gradients, exact ranking ties/masks/cold targets, CE recomputation gradients, accumulation including partial groups, seeded worker replay, validation-only guards, warm-up-aware patience, sparse-cadence resume, archive safety and E7/E8 artifact reuse.

All five real checkpoints produced identical top-K, target ranks and metrics between evaluators for 1,024 validation users each: zero same-weight discrepancies. In the independently refitted FENRec training benchmark, centroid round-off reached {center_difference:.3g}; one target-rank difference after training was observed while top-K remained identical. That is preserved in correctness.json, not hidden. Parallel Lloyd fits can perturb trained weights despite fixed seeds. Same-checkpoint evaluator equivalence and unit gradient checks distinguish numerical fit variation from protocol changes.

The two-trial real-Beauty tuning smoke completed validation-only runs with no test metrics/predictions. Its selected YAML is marked pilot_only and cannot select a final research configuration. A full hyperparameter search and long-run convergence analysis have not been executed locally.

## Reduced and full profiles

Core Research uses Beauty, all implemented E1 models, matched E2, E3 retention 0.4/0.7/1.0, all six E4 variants, E5 lengths 10/20/50, and E7/E8 reuse. Initial tuning/convergence uses seed 7; final runs support 7/42/2026 after validation selection. The Full Research profile preserves all original axes and future TIGER entries. The Pilot profile is explicitly non-research.

{chr(10).join(budget_lines)}

These are local accelerator-session wall-hour scenarios, not guaranteed convergence times or measured Kaggle budgets. Known Beauty E1 length-50 costs use complete measured epochs and validation. Other lengths, retention ratios, E2 and E4 variants are explicitly labeled linear/conservative surrogates; all FENRec variants use the full branch's batch-rate estimate, so simpler variants are overestimated. The nominal floor counts enforced minimum training, not predicted stopping. Exact duplicate settings are charged once; shared E1 artifacts are reused. The figures exclude tuning, one-time final test evaluation, E7/E8, setup/export and unmeasured datasets/generative work. Full Research's fifty unknown entries are additional costs, not zero-cost experiments.

The old approximately 83-hour number was a short-pilot extrapolation at batch 64, not a measured run. At batch 128, the new complete-epoch measurements imply about {e1_hours:.1f} hours for the five neural Beauty E1 models at 300 epochs with validation every epoch, before early stopping. This comparison combines pipeline optimization, physical batch change and a better measurement basis; it is not a pure speedup claim. Core profiles further reduce epoch caps and validation cadence but still require staged scheduling.

Core tuning is bounded to six default trials per model (hard maximum twelve), with nominal 60-epoch trial caps and 80 for GRU/BERT; FENRec has a minimum forty-epoch allocation including twenty post-warm-up epochs. The actual selected learning rate/batch/dropout/model-specific parameters require validation-only search on Kaggle; no test-based tuning occurred. Tuning can be scheduled model-by-model with fewer explicit trials before committing to the full bound.

## Deliverables, execution and remaining blockers

Exact upload/setup/data/selection/resume/export steps: [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md). Thin notebooks: Kaggle_Setup.ipynb, Kaggle_Training.ipynb and Kaggle_Analysis.ipynb. Updated manifests and estimate provenance: `results/phase5a/budget`. Source/result ZIP helpers preserve conflicts and support cross-session restoration. Never assume a previous /kaggle/working directory survives.

No local environment or Beauty data blocker remains. Actual Kaggle session execution, Kaggle-specific throughput/quota measurement, long-run convergence/validation tuning, Sports/Toys costs, final multi-seed science and TIGER remain outstanding. Source-comparison tests need optional reference checkouts when running the packaged project. The optimize-for-gpu workflow/software citation is recorded in PHASE5A_AUDIT.md.
'''
    (root/"docs/EXPERIMENT_BUDGET.md").write_text(text, encoding="utf-8")
    print(root/"docs/EXPERIMENT_BUDGET.md")


if __name__ == "__main__": main()
