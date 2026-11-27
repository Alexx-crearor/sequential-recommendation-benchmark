# Implementation plan

## Official session 1 update — 2026-11-27

Verified the original master ZIP `seqrec-master-results-20261126_234725_576013.zip`, SHA256 `9f6b30dd3d29750b21ef64ddd7c8fea88c66d1c3ca52ea44a9809b3cc323cfab`: 279 files, 240 status-hashed artifacts, 43 checkpoints, 22 completed runs. Actual local Beauty bytes match the archived manifest; chronology and all per-user target/history/ranking contributions pass. Original private ZIP and all prior artifacts are preserved. Session state is BUDGET_STOP after the user-reported approximately 8.5 hours; original grid coverage remains 22/127.

E1 COMPLETE 16/16 (five neural models × seeds 7/42/2026 plus one deterministic MostPopular). E2 PARTIAL 6/12: A1/A2 seeds 7/42 and A3/A4 seed 7 are present. E3/E4/E5 have no completed runs in the export. E7 dedicated GPU measurements DEFERRED; real embedded training/inference measurements are analyzed without remeasurement. E8 local prediction-based analysis COMPLETE. Reports include all Recall/NDCG/MRR full/warm means and sample SD, exact empirical paired-user bootstrap (10,000 resamples of seed-averaged contributions), partial E2, errors/history/popularity, eight PNG/PDF figure pairs and four notebooks. See `docs/MASTER_SESSION1_ANALYSIS_VI.md`.

**Current course scope overrides the historical milestones below because of GPU limits and course requirements, not model test scores:**

- Keep E1 all three neural seeds (already complete).
- Finish the current E2 design: six missing runs.
- E3 only SASRec-CE/BERT4Rec/FENRec, retention 0.4/1.0, seed 7. Three ratio-1.0 sources are REUSE_AVAILABLE from E1, not yet E3 SUCCESS; three new ratio-0.4 runs remain.
- E4/E5 DEFERRED. TIGER and CRAMER THEORY_ONLY for this course scope; future full-design code/references remain preserved.
- Prioritize E7/E8 reuse; do not relabel dedicated E7 as complete.

Frozen `configs/experiments/course_jobs.json` selects 34 original artifact requests, retaining resolved config hashes and the original common-warm protocol. Only nine new training jobs remain, plus three E3 reuse operations. `scripts/run_course_remaining.py` prints the plan locally; explicit --run is restricted to Kaggle, refuses retraining missing archived SUCCESS sources, and restores/skips verified completed runs. Original 127-job progress and original Full/Core grids remain untouched. Statistical analysis/visualization/publication are local; no experimental training/tuning or new GPU measurements were performed locally.

Original completed best/resume weights are materialized under `checkpoints/master_session1/` and distributed as prediction-free Release bundles; raw source ZIP is under `artifacts/master_session1/`. Both are ignored as Git blobs. Reproduction and per-user E8 evidence stay under content-addressed raw/derived directories. Final conclusions are limited to the completed Beauty design; pending/deferred/theory-only work is not claimed complete.

## Phase 5C update — 2026-11-26

Checkpoint distribution follow-up: the original private ZIP is now copied inside `artifacts/phase5c/`, and all 30 original checkpoints are materialized under `checkpoints/phase5c/beauty/<model>/trials/<trial>/`. CLI/notebook defaults are repository-relative, with explicit overrides. Public prediction-free checkpoint bundles are distributed via GitHub Release `phase5c-checkpoints-v1`; `scripts/fetch_phase5c_checkpoints.py` downloads selected/all weights with ZIP/member SHA256 verification. Large binaries remain outside Git history; no new training or numerical research result is introduced.

**Completed: the exported bounded hyperparameter tuning batch, 15/15 trials (3/model), Beauty seed 7.** The sole Phase 5C evidence source is `seqrec-phase5c-final-20261126_132556_321898.zip`, SHA256 `7861c626561fc1d7914abb8ee5a7eb4d552a932aca5dfbc54419648efb0176fa`. Verified ZIP CRC/member hashes, 150 status-hashed artifacts, 30 best/latest checkpoints, plan/config hashes, full validation contributions/top-K coverage, shared split/population and validation-only winner selection. Original ZIP, prior pilots and raw artifact bytes are preserved. No tuning or real-model training is launched locally.

Selections for SASRec-CE/BCE, GRU4Rec-CE, BERT4Rec and FENRec are frozen under `configs/selected/phase5c/beauty/`. The reproducible CPU analysis, all-trial/epoch tables, academic PNG/PDF figures and Vietnamese interpretation are linked in `docs/PHASE5C_ANALYSIS_VI.md` and `notebooks/Phase5C_Analysis.ipynb`. Nine trials hit epoch caps, six stop by patience; capped trials are not labeled converged. Three trials/model are a completed bounded batch, not an exhaustive search.

Verification limit: split IDs agree with the archived PASSED data audit and every config/checkpoint. Dataset bytes and a training source/Git attestation are not included in the ZIP; independent dataset rehashing and proof of unseen execution are outside this archive audit. Validation performance is not final test performance.

Remaining: preselected final E1 at seeds 7/42/2026; E2–E5 controlled experiments; E7/E8 on final artifacts; Sports/Toys measurements; TIGER/E6; final multi-seed statistical analysis and report. Phase 5B execution status is not inferred from this export. Kaggle executes training/testing/evaluation/GPU measurements; all analysis/visualization/publication is local.

## Phase 5A update - 2026-11-26

Implemented exact GPU-streamed catalogue ranking, vectorized GRU/input assembly, cached training tensors, efficient configurable loaders, independent-objective gradient accumulation, optional CE recomputation/AMP, validation/checkpoint cadence, warm-up-aware patience, validation-only bounded tuning, pilot/core/full profiles and thin Kaggle setup/training/analysis notebooks. E7/E8 have verified E1 artifact reuse CLIs. Aggregation now separates experiments/variants/retention/length/hyperparameter settings instead of pooling incompatible runs.

Actual CPU/CUDA profiling, matched original/optimized short benchmarks, five complete optimized epoch measurements, same-checkpoint ranking audits and a real two-trial validation-only tuning smoke are recorded under `results/phase5a`. Existing Beauty pilots are preserved. See `docs/EXPERIMENT_BUDGET.md` for generated evidence and estimates, and `docs/KAGGLE_GUIDE.md` for source packaging, exact individual commands, restoration and export.

Historical Phase 5A state: Core initial selection uses seed 7 and final seed expansion 7/42/2026 follows validation selection. Phase 5C above now supplies the actual bounded research selections and Kaggle tuning measurements. Final convergence studies, Kaggle session quotas, Sports/Toys profiling, TIGER and final scientific reporting remain outstanding. No full local experiment grid was launched.

## Scope and current state

Updated 2026-11-25: Phase 1-4 is implemented. Phase 2 working modules and tests were retained. Python 3.10.11 now has PyTorch 2.5.1+cu121; CPU and Quadro T1000 CUDA execution are verified. Real Amazon Beauty has been downloaded, hashed and preprocessed, with a passing per-model leakage audit and successful MostPopular evaluation.

Completed neural models: GRU4Rec-CE, SASRec-BCE, SASRec-CE, BERT4Rec, capacity-matched E2 causal/bidirectional variants, and FENRec F0-F5. The trainer supports early stopping, independent best/latest checkpoints, full-catalogue evaluation, deterministic epoch-boundary resume, FP32 FENRec and opt-in Phase 3 AMP. E1-E5 selectors and a Kaggle workflow are operational. Main and post-warm-up pilots are isolated from final research results. The execution report and machine-generated pilot/budget artifacts record actual outcomes.

## Milestones and acceptance gates

1. **Phase 1:** inspect the six references and original sources; pin source commits; define evaluation and unresolved decisions. See `docs/METHOD_REFERENCES.md` and `docs/EXPERIMENT_PROTOCOL.md`.
2. **Phase 2:** pass metric, data-integrity, no-leakage, candidate-mask, checkpoint, corruption, and completed-run skip tests. Download real category files explicitly; hash inputs and generate per-category bundles. CPU integration fixtures exercise the complete path without creating benchmark results.
3. **Phase 3 - implemented and tested:** installed a compatible CUDA PyTorch build. Implemented GRU4Rec-CE, SASRec-BCE/CE, BERT4Rec and the E2 variants with one scoring interface. Causality, padding, shifted labels, masked inference and CPU/CUDA checks pass. Architectural/objective adaptations are documented.
4. **Phase 4 - implemented and tested:** resolved the FENRec discrepancy using the pinned code as the primary branch; implemented training-only future windows, augmentation, loss schedules, false-negative masking and all ablation switches. Component outputs/gradients match pinned source tests; F0-F5 have finite component-level ablation tests. Real standard and post-warm-up Beauty pilots completed.
5. **Phase 5:** measure neural pilot epoch time/memory on real data, estimate grid cost, finalize bounded validation-only tuning and checkpoint resume. Execute non-generative E1 then E2–E5 selectively. No full grid is launched at this milestone.
6. **Phase 6:** choose full TIGER or explicitly named TIGER-Lite; audit the content/tokenizer fit population and availability assumptions. Implement and test residual quantization, disambiguation, generation, constrained/unconstrained decoding diagnostics and E6 controls.
7. **Phase 7:** profile E1 checkpoints with warm-up and synchronized CUDA timing; preserve stage-specific generative cost. Pair predictions by user and split for error analysis.
8. **Phase 8:** produce the 12 requested tables, 13 real-data figures, nine analysis notebooks and Vietnamese report from verified artifacts. No placeholder figures or inferred numerical findings.

## Resource planning

The baseline has no gradient training and uses catalogue frequency storage. Neural initial settings are in `configs/experiments/experiment_grid.yaml`. Full batch-128 epochs now fit on the local 4 GiB GPU. Independent CE/BCE objectives support weighted accumulation; active FENRec contrastive variants cannot split negative pools. Physical batch changes must be matched across contrastive comparisons. Keep model capacity unchanged across matched comparisons. Phase 3 AMP is validated but was not faster locally. Exact scoring remains O(users times catalogue) and streams every item chunk.

Dry-run planning fingerprints are not executable configuration hashes: actual split versions, implementation versions, tuned values and retention manifests must be resolved first. E2 now uses a capacity-matched post-norm backbone and does not reuse E1. E3 retention 1.0 identifies possible E1 reuse; E5 length 50 also identifies possible reuse. FENRec F5 and semantic T2 remain separate until their complete settings match. E7/E8 reuse E1 artifacts rather than training new models. Pilot-derived estimates are generated under results/reports; Sports/Toys and the full retention/length axes remain unmeasured.

## Unresolved details

- FENRec primary loss follows code weighting; paper_margin is a separately labeled margin-placement sensitivity branch. CPU sklearn clustering replaces GPU FAISS and is explicitly documented.
- Training-only future windows and same-target positives are implemented without upstream caches. Source encoder/loss values and gradients are numerically checked.
- Decide TIGER pretrained encoder/model version, tokenizer hyperparameters and metadata-only transductive fit policy. Record missing metadata coverage and fit costs.
- Neural resume preserves status attempts and restores the latest full epoch including optimizer/scaler/RNG. Exactness is tested on the same software/device configuration; changing settings creates a new run hash.
- MostPopular retains safe NPZ checkpoint contents; neural best/latest checkpoints use PyTorch state dictionaries and safe weights-only loading.
- CPU and CUDA smoke tests now cover all Phase 3/4 models. Remaining TIGER-specific tests are Phase 6 gates.
- Real-data preprocessing and actual-model leakage checks must pass before any large run.

## Remaining work

The bounded validation-only tuner is implemented; the exported Phase 5C batch of fifteen research trials is completed and audited. An exhaustive search and final training grid have not been run. Final multi-seed statistical comparisons, TIGER and final-test publication figures/report remain outstanding. Phase 3/4 results are historical in `docs/PHASE34_RESULTS.md`; Phase 5C analysis is in `docs/PHASE5C_ANALYSIS_VI.md`; execution commands and historical estimates are in `docs/KAGGLE_GUIDE.md` and `docs/EXPERIMENT_BUDGET.md`.
