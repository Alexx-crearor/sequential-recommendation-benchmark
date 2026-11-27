# Experiment budget and Phase 5A performance report

Current course budget update (2026-11-27): the historical scenarios below are preserved. Actual official-session measurements and the reduced nine-job remaining-work scenario are now in [MASTER_SESSION1_ANALYSIS_VI.md](MASTER_SESSION1_ANALYSIS_VI.md) and `results/tables/master_session1/remaining_budget_scenarios.csv`. E4/E5 are deferred and TIGER/CRAMER are theory-only in the course scope. Do not treat the original Full/Core grid caps below as the current course execution requirement.

Generated 2026-11-26 from measured artifacts. No full research grid was launched. Prior pilot artifacts and the original experiment grid are preserved.

## Actual implementation status

MostPopular, GRU4Rec-CE, SASRec-BCE/CE, BERT4Rec, causal masked E2 and FENRec F0-F5 are operational. Architectural/source differences remain documented in METHOD_REFERENCES.md. Phase 5A adds exact device-resident ranking, cached train-only tensors, vectorized GRU/padding, safe independent-objective accumulation, optional CE activation checkpointing, configurable loaders/validation/checkpoint cadence, validation-only bounded tuning, three experiment profiles and Kaggle orchestration notebooks. TIGER is still a future implementation, retained in Full Research.

## Before/after controlled short benchmark

Quadro T1000 4 GiB; Python 3.10.11; PyTorch 2.5.1+cu121; FP32; batch 128; d=64; two layers; maximum length 50; seed 7. Two repeats of twenty measured batches follow one sacrificial warm-up update and a full model/optimizer/RNG reset. Validation rows below use 1,024 users and the complete 12,101-item catalogue. Original hot-path snapshots are under benchmarks/phase34. Both modes use the same data, initialization, objectives and precision; AMP is a separate experiment. Reported times are synchronized wall time, including data loading and transfer.

| Model | Original batch s | Optimized batch s | Batch speedup | Original validation s | Optimized validation s |
|---|---:|---:|---:|---:|---:|
| gru4rec_ce | 0.3355 | 0.0448 | 7.49x | 3.655 | 0.360 |
| sasrec_bce | 0.0529 | 0.0373 | 1.42x | 2.195 | 0.325 |
| sasrec_ce | 0.0410 | 0.0349 | 1.17x | 2.223 | 0.348 |
| bert4rec | 0.0556 | 0.0512 | 1.09x | 2.655 | 0.368 |
| fenrec | 0.1197 | 0.0954 | 1.25x | 2.527 | 0.321 |

Source: `results/phase5a/benchmark_isolated_fp32_b128/measurements.csv`. An earlier exploratory pass overlapped correctness work and is not the source of this table. Short timing variance and CPU load remain relevant; repeat on Kaggle before scheduling. GPU samples span the entire benchmark, including CPU preparation/clustering, so their average is not a training-only utilization claim.

## Complete optimized epoch measurements

One full training epoch per model, followed by full validation on all 22,363 Beauty users. FENRec uses epoch index 21 to measure post-warm-up compute; this is a runtime benchmark, not a convergence claim. Training seconds exclude the separately listed epoch-wise clustering. The other models have 175 batches; FENRec has 1,027. Peak memory includes the measured training/validation pass.

| Model | Complete training epoch s | Clustering s | Full validation s | Examples/s | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|
| gru4rec_ce | 8.51 | 0.00 | 5.88 | 2629 | 134.9 |
| sasrec_bce | 7.63 | 0.00 | 4.87 | 2931 | 128.0 |
| sasrec_ce | 6.07 | 0.00 | 4.80 | 3682 | 177.3 |
| bert4rec | 8.08 | 0.00 | 4.81 | 2768 | 162.4 |
| fenrec | 102.69 | 60.52 | 3.78 | 1280 | 386.3 |

Source: `results/phase5a/full_epoch_b128/full_epoch_measurements.csv`. FENRec clustering retains its exact configured backend/centroids/restarts/iteration budget; no infrequent refresh or approximate replacement was introduced. Its trace splits CPU Lloyd fitting and prefix encoding. Full-catalogue CE remains exact. Larger physical batches remain a model setting for contrastive comparisons.

## Bottlenecks and precision findings

The original cProfile recorded 1,535,373 calls and 2.875 seconds for 1,024-user validation, with about 1.203 seconds in repeated history construction. NumPy per-user masking/sorting and tiny CUDA history copies were prominent. The optimized path transfers only final ranks/top-K and processes catalogue chunks on-device. GRU per-user GPU control flow was the largest training bottleneck. Cached tensor construction costs more up front but amortizes across epochs. Full FENRec still performs three independent-dropout forward passes and full prefix clustering; collapsing those would change its objective and was not done.

Detailed CPU/CUDA operator traces are in `results/phase5a/trace_sasrec_ce` and `trace_fenrec`. Profiler instrumentation inflates timings; the benchmark table, not profiler absolute time, is used for cost estimates. Checkpoint write times, examples/s and memory are in the benchmark CSV. No OOM occurred at batch 128; memory was not the binding limit at that tested capacity. Worker/pinning settings remain configurable, with zero workers as the measured portable default.

AMP was not faster locally:

- sasrec_ce: FP32 0.0349 s/batch; AMP 0.0483 s/batch (1.38x the FP32 time).
- bert4rec: FP32 0.0512 s/batch; AMP 0.0642 s/batch (1.25x the FP32 time).

AMP remains opt-in for Phase 3 and disabled for FENRec. Reduced memory alone is not proof of speed. CE activation checkpointing is a separate opt-in memory/compute trade-off with gradient-equivalence tests.

## Numerical and leakage verification

129 tests passed with 0 failures, 0 errors and 0 skips in the recorded local suite. Tests include cache/transform equality, GRU states/gradients, exact ranking ties/masks/cold targets, CE recomputation gradients, accumulation including partial groups, seeded worker replay, validation-only guards, warm-up-aware patience, sparse-cadence resume, archive safety and E7/E8 artifact reuse.

All five real checkpoints produced identical top-K, target ranks and metrics between evaluators for 1,024 validation users each: zero same-weight discrepancies. In the independently refitted FENRec training benchmark, centroid round-off reached 3.58e-07; one target-rank difference after training was observed while top-K remained identical. That is preserved in correctness.json, not hidden. Parallel Lloyd fits can perturb trained weights despite fixed seeds. Same-checkpoint evaluator equivalence and unit gradient checks distinguish numerical fit variation from protocol changes.

The two-trial real-Beauty tuning smoke completed validation-only runs with no test metrics/predictions. Its selected YAML is marked pilot_only and cannot select a final research configuration. A full hyperparameter search and long-run convergence analysis have not been executed locally.

## Reduced and full profiles

Core Research uses Beauty, all implemented E1 models, matched E2, E3 retention 0.4/0.7/1.0, all six E4 variants, E5 lengths 10/20/50, and E7/E8 reuse. Initial tuning/convergence uses seed 7; final runs support 7/42/2026 after validation selection. The Full Research profile preserves all original axes and future TIGER entries. The Pilot profile is explicitly non-research.

| Profile | Entries | Nominal minimum-epoch floor h | Maximum known-Beauty h | Unknown entries |
|---|---:|---:|---:|---:|
| pilot_initial | 55 | 0.3 | 0.4 | 0 |
| core_research_initial | 45 | 16.9 | 54.4 | 0 |
| core_research_final | 129 | 50.8 | 163.2 | 0 |
| full_research_final | 209 | 43.5 | 562.4 | 50 |

These are local accelerator-session wall-hour scenarios, not guaranteed convergence times or measured Kaggle budgets. Known Beauty E1 length-50 costs use complete measured epochs and validation. Other lengths, retention ratios, E2 and E4 variants are explicitly labeled linear/conservative surrogates; all FENRec variants use the full branch's batch-rate estimate, so simpler variants are overestimated. The nominal floor counts enforced minimum training, not predicted stopping. Exact duplicate settings are charged once; shared E1 artifacts are reused. The figures exclude tuning, one-time final test evaluation, E7/E8, setup/export and unmeasured datasets/generative work. Full Research's fifty unknown entries are additional costs, not zero-cost experiments.

The old approximately 83-hour number was a short-pilot extrapolation at batch 64, not a measured run. At batch 128, the new complete-epoch measurements imply about 18.1 hours for the five neural Beauty E1 models at 300 epochs with validation every epoch, before early stopping. This comparison combines pipeline optimization, physical batch change and a better measurement basis; it is not a pure speedup claim. Core profiles further reduce epoch caps and validation cadence but still require staged scheduling.

Core tuning is bounded to six default trials per model (hard maximum twelve), with nominal 60-epoch trial caps and 80 for GRU/BERT; FENRec has a minimum forty-epoch allocation including twenty post-warm-up epochs. The actual selected learning rate/batch/dropout/model-specific parameters require validation-only search on Kaggle; no test-based tuning occurred. Tuning can be scheduled model-by-model with fewer explicit trials before committing to the full bound.

## Deliverables, execution and remaining blockers

Exact upload/setup/data/selection/resume/export steps: [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md). Thin notebooks: Kaggle_Setup.ipynb, Kaggle_Training.ipynb and Kaggle_Analysis.ipynb. Updated manifests and estimate provenance: `results/phase5a/budget`. Source/result ZIP helpers preserve conflicts and support cross-session restoration. Never assume a previous /kaggle/working directory survives.

No local environment or Beauty data blocker remains. Actual Kaggle session execution, Kaggle-specific throughput/quota measurement, long-run convergence/validation tuning, Sports/Toys costs, final multi-seed science and TIGER remain outstanding. Source-comparison tests need optional reference checkouts when running the packaged project. The optimize-for-gpu workflow/software citation is recorded in PHASE5A_AUDIT.md.
