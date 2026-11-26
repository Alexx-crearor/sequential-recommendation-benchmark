# Experiment protocol v3

## Data and information boundary

Use the 2014 category 5-core review archives and matching metadata, not Amazon 2018/2023. Reapply iterative 5-core after deduplication. Treat all ratings 1–5 as implicit interaction events; no positive-rating threshold. Retain only the earliest event for each user/item pair. Sort by timestamp then lexicographic ASIN, with rating resolving duplicate-event ties. Timestamp ties impose a deterministic order, not evidence of actual within-day order. Assign sorted user/item IDs starting at 1; 0 is padding only.

Use all filtered items as the static eligible catalogue, including items absent from training. This is a transductive catalogue assumption: item identity is known, held-out interactions are not learning data. No future popularity or target label enters fitting. Core selection uses the full interaction graph as a conventional benchmark construction step; it is not a global temporal deployment simulation. Document this limitation.

For each user hold out the last event for test and penultimate for validation. All previous events form train. Construct prefixes only from train. Validation receives train history. Test receives train plus validation history after configuration selection; validation events never become training labels. Popularity counts use train only. FENRec windows, contrastive supervised-positive grouping and any learned statistics must use train only. Metadata must exclude reviews, ratings, also-bought links and evaluation-derived fields. Missing metadata never drops a target.

Bundles contain train/validation/test parquet, complete interactions, ID mappings, selected metadata, statistics and a SHA-256 manifest. Their directory key includes raw hashes and preprocessing policy; split_version hashes the actual artifact manifest. Verify every file before every run. Bundles are immutable: invalid existing bundles raise an error rather than being overwritten.

## Ranking and coverage

Score every eligible item with full-catalogue rank computation, in item batches. Mask previously observed history items except the target; preserve this exception for external repeat-policy variants too. Exclude padding/MASK from the catalogue. Resolve equal scores by ascending stable item ID for reproducibility. Report Recall and NDCG at 5/10/20, MRR@10. Save each user's complete history, target, top 20, exact target rank, warm indicator and metric contributions. For one relevant target Recall equals hit rate.

Report full coverage and train-warm target subset separately, with user counts and warm coverage. Empty subset metrics are null, never fabricated zeros. For sparse models, additionally intersect warm target users across compared retention manifests before common-subset comparisons. Keep held-outs unchanged when subsampling train. Use a single seed-specific event selection for all models; retain at least one train event per user and record realized retention and selected event IDs. The realized ratio may exceed the requested ratio for short histories.

Generative retrieval must use the same catalogue, history policy and targets, deduplicate mapped items, and count missing/invalid outputs. Missing retrieved targets have zero top-K contributions. Do not substitute full-catalogue scoring for beam retrieval. Report beam budget and retrieval coverage separately.

## Training and comparisons

Use seeds 7, 42, 2026 for stochastic models; MostPopular once (directory seed 7 is a bookkeeping value). Use the centralized initial defaults, then at most 12 validation-only tuning trials per model/dataset. Reuse a selected configuration across seeds; log search space, trial seeds, elapsed time and budget. Select validation NDCG@10; break ties by earlier epoch. Never select on test. Flag 300-epoch exhaustion without convergence. Neural epoch-boundary resume restores optimizer, AMP scaler and Python/NumPy/PyTorch/CUDA RNG states. Exact continuation is tested under the same software/device configuration.

Match E2 A3/A4 capacity, corruption masks and training samples. Do not infer that A4 equals E1 BERT4Rec without identical settings. E4 switches require a verified mapping to the source components, including the focal weighting decision. E6 T1/T2 match code length, codebooks, generator capacity, budget and training. Report atomic T0 capacity differences explicitly.

## Statistics, efficiency and artifacts

Report per-seed metrics and mean/sample SD across seeds. A single deterministic run has undefined sample SD, not a claimed variance estimate. Bootstrap paired per-user contributions with 10,000 seeded resamples and percentile 95% intervals, aligning user IDs and split hashes. Specify whether comparisons are within each seed or use per-user seed-averaged contributions; prefer the latter and acknowledge it does not capture retraining uncertainty.

Profile hardware/software and inference batch size, warm-up count, repeated timing count, CUDA synchronization, mean/P95, throughput and peak allocated memory. Reset CUDA peak measurements before each profiled phase. Keep training time, preprocessing time and end-to-end retrieval time separate. Measured Beauty pilot estimates are recorded under results/reports; they are same-hardware extrapolations, not measured full-grid costs.

Each run must write configuration, environment, history, validation/test metrics, efficiency, predictions, checkpoint and log. Hash all artifacts in status. SUCCESS requires finite and consistent contributions, complete unique-user predictions and a readable checkpoint. FAILED records the exception. An existing incomplete or corrupted run is preserved and rejected. Automatic skipping requires full verification. Neural recovery replays from the latest complete epoch; best weights remain separate. Prior attempt status records are preserved.

Smoke fixtures are temporary and excluded from research tables. All final tables, figures and Vietnamese numerical conclusions must cite verified real run artifacts. Cases support descriptive interpretation; they do not establish causation.


Phase 3/4 decisions: E2 A1-A4 share the same post-norm backbone and output head; no E1 alias is assumed. FENRec primary weighting follows the pinned code, with both coarse/fine CL branches and sklearn CPU Lloyd clustering. Pilots explicitly hash training/validation truncation controls and are excluded from final aggregation. The default hard-negative warm-up remains 20 epochs; the separately named post-warm-up pilot changes it to zero solely to exercise the branch. Full tests use every test user, even when pilot validation is limited.

## Phase 5A profile and execution amendments

The original experiment_grid.yaml is retained. Core Research explicitly narrows datasets to Beauty, E3 retention to 0.4/0.7/1.0 and E5 lengths to 10/20/50; Full Research preserves all original axes. Initial Core uses seed 7; final seeds 7/42/2026 follow validation selection. Core's common-warm target intersection uses its three retention ratios. Hashes include profile, runtime settings and retention provenance.

All ranking optimizations preserve the eligible catalogue, history/target mask, item-ID tie ordering, metrics and held-outs. Catalogue chunks stay on-device; no sampled evaluation or approximate retrieval was introduced. Same-checkpoint real-data comparisons produce identical ranks/top-K/metrics. CPU parallel clustering may vary at floating-point round-off; numerical validation is not a claim of bitwise equality across independent fits/hardware.

Core validates every two epochs by default; Full validates every epoch. First/final validation always runs. Patience counts validation checks, not unvalidated epochs. Model-specific minimum/maximum epochs are recorded. FENRec research budgets include twenty post-warm-up epochs before patience can terminate training; pilots may deliberately use shorter tagged verification budgets. This changes selection frequency in Core explicitly rather than silently claiming the original schedule.

Independent CE/BCE gradient accumulation weights microbatches by valid target count (FENRec F0/F1 by examples), including partial final groups. Active FENRec CL accumulation is rejected because it would change the batch negative pool. Physical contrastive batch size is fixed across component comparisons unless explicitly selected and documented. Optional AMP/CE activation recomputation are recorded; FENRec stays FP32.

Bounded tuning uses validation only, at seed 7, and never calls test evaluation. Its completed state is VALIDATED, distinct from final SUCCESS. Trial plans, budgets, hyperparameters, convergence flags and selection provenance are saved. Pilot-only selected configurations cannot be used as final research selections. The Phase 5C export now verifies fifteen research trials and five validation-selected configurations; final multi-seed test runs are still unexecuted by that export.

## Phase 5C evidence and analysis boundary

The source ZIP SHA256 is `7861c626561fc1d7914abb8ee5a7eb4d552a932aca5dfbc54419648efb0176fa`. All Phase 5C numerical tables/figures derive exclusively from that original archive. Local pilots, Phase 5A speed measurements and other result trees are not pooled into tuning results. The original selection rule, objectives, full-catalogue protocol, held-out split and training budgets are unchanged. Three trials per model use seed 7. Future final selections are frozen in `configs/selected/phase5c/beauty/`; selecting by test outcomes is prohibited.

Raw import is content-addressed and rejects conflicting artifacts before extraction. Git stores aggregate/epoch-level summaries, figures, hashes and selected configs; raw per-user histories/predictions, checkpoints, logs, datasets and ZIPs stay local. The archived PASSED data audit and all split IDs agree; actual dataset bytes and training-source attestation are absent from the export, so this audit does not independently rehash the dataset or certify unseen code execution.

Only scheduled validation checks enter curves and best-epoch selection. Best/final epoch numbers shown to readers are 1-based, with raw 0-based epochs retained in CSV. Patience counts validation checks and respects minimum/post-warm-up gates. Budget exhaustion is reported separately from patience stopping; neither alone proves global convergence. No single-seed tuning scores are presented as final-test estimates or statistically significant model improvements. Loss scales are compared within the same trial/model objective, not as cross-model ranking.

Training time includes FENRec clustering; it must not be counted twice. Logged training/validation/checkpoint component sums are not end-to-end session wall time. Peak GPU memory is allocated memory as recorded during Kaggle execution. Kaggle is used for training/testing/evaluation/GPU measurements; artifact audit, tables, visualization and repository publication are performed on the personal machine, with no experimental retraining.

Checkpoint cadence is configurable; checkpoints contain the matching best weights with latest optimizer/scaler/RNG state. An interruption can replay up to the checkpoint interval. Export/restore does not assume Kaggle working files persist and rejects conflicting existing artifacts. Aggregation groups experiments, variants, retention, sequence length and complete comparable hyperparameters separately; seed-specific retention hashes are retained per run but not used to split the across-seed comparison group.
