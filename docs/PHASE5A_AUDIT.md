# Phase 5A implementation audit (2026-11-26)

Phase 1-4 code, configurations, tests, documentation and seven completed Beauty pilot artifact trees were inspected. Existing model architecture/objective implementations were retained. No old result tree was deleted or rewritten. The prior grid remains `configs/experiments/experiment_grid.yaml`.

| Component | Actual status | Remaining limitation |
|---|---|---|
| MostPopular | Exact training frequencies, safe NPZ checkpoint, full-catalogue evaluation | Deterministic baseline; no optimization/convergence claim |
| GRU4Rec-CE | Explicit user-history CE adaptation, packed GRU, tied outputs | Not original session-parallel ranking-loss training |
| SASRec-BCE | Original-style blocks and one excluded-history negative per position | Full-catalogue evaluator differs from original sampled evaluation by design |
| SASRec-CE | Same SASRec blocks, full-catalogue CE | Named objective adaptation |
| BERT4Rec | Dynamic Cloze, bidirectional visibility, appended MASK inference | Original TF offline duplication/optimizer/segment defaults remain documented adaptations |
| E2 causal masked variant | Shared post-norm backbone, causal masked CE | All E2 settings must share selected hyperparameters and capacity |
| FENRec | Source-code primary loss, training-only decay labels, coarse/fine CL, detached mixed/noise negatives, warm-up | CPU sklearn clustering differs from upstream GPU FAISS; source discrepancy remains explicitly resolved in favor of code |
| F0-F5 | All six supported switches and identical backbone capacity | Contrastive effective batch is a research setting; cannot split negative pools silently |
| TIGER | Not implemented | Future Phase 6, retained in full manifest and gated at execution |

Before Phase 5A the pipeline lacked bounded tuning, gradient accumulation, configurable validation/checkpoint cadence, efficient device-resident ranking and Kaggle session notebooks. These are now implemented. E7/E8 CLIs reuse verified E1 artifacts for measured efficiency and descriptive error cases; the final comprehensive scientific figures/report remain future work.

## Measured profiling evidence

The initial unmodified 1,024-user SASRec validation profile recorded 1,535,373 calls in 2.875 seconds. History reconstruction accounted for about 1.203 seconds; ranking repeatedly performed per-user NumPy masking/sorting. CUDA history assembly issued a separate small transfer for each user. Frozen pre-optimization modules in `benchmarks/phase34/` permit controlled replay; they are not runtime model implementations.

Other identified costs: Python loops around each GRU sequence; dynamic re-padding and same-target history scans per FENRec sample; full training-prefix encoding and unchanged Lloyd clustering every FENRec epoch; retained full-catalogue CE activations; validation every epoch; checkpoint and artifact verification. A first exploratory benchmark overlapped correctness work and is retained as exploratory evidence only. The isolated benchmark output is the source for budget/report numbers.

## Safe changes and explicit exclusions

- Exact catalogue ranking stays on the scoring device and streams all catalogue chunks. Stable score sorting retains item-ID ties, full history masking and target exemption. Only final top-K/ranks transfer to CPU.
- Histories are assembled in one CPU array and copied once. GRU stable gather/scatter replaces per-user GPU loops without changing real-token outputs/gradients.
- Training-only padded tensors and deterministic same-target positive indices are cached with a configurable host-memory cap. Dynamic masks/negatives remain freshly sampled. Cache construction has an up-front cost and is not claimed to be faster to initialize.
- Optional pinned/nonblocking transfer, worker counts and prefetch settings are configurable. Workers restart per epoch and receive seeded Python/NumPy RNGs; persistent-worker RNG is not silently omitted from resume state.
- CE/BCE accumulation uses exact target-count weighting and handles partial final groups. F0/F1 can accumulate by examples. Active FENRec contrastive variants reject accumulation because their complete batch defines the negative pool.
- Optional activation checkpointing recomputes full-catalogue CE chunks during backward; no catalogue item is sampled away. Phase 3 AMP remains opt-in and separately measured. FENRec remains FP32.
- FENRec cluster encoding uses cached input batches and a capped GPU embedding buffer; the clustering algorithm, refit frequency, number of centroids, restarts and iteration budget remain unchanged.
- Validation/checkpoint cadence and model-specific training limits are hashed. First/final validation always runs. Patience counts validation checks; FENRec minimum post-warm-up training is enforced in research profiles. Latest checkpoints embed the matching best checkpoint so rollback cannot pair stale optimizer state with newer best weights.
- Validation-only trials end as VALIDATED with validation predictions and no test metrics/predictions. Final SUCCESS retains the established artifact contract.

## Protocol changes

Ranking, catalogue coverage, held-out targets, metrics, repeats and dataset split hashes are unchanged. Core retention ratios are explicitly 0.4/0.7/1.0 and its common-warm intersection uses those ratios; full retains 0.2/0.4/0.6/0.8/1.0. Core sequence lengths are 10/20/50. Core validation cadence changes selection opportunities and is documented as a new profile rather than a reproduction of the original every-epoch training schedule. Old fingerprints/artifacts remain valid; new pipeline settings have new fingerprints.

## Software/process reference

The evidence-driven optimization workflow used the local optimize-for-gpu skill. Its software reference was checked against the current arXiv record:

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065).

PyTorch implementation guidance: [AMP and accumulation](https://docs.pytorch.org/docs/2.5/notes/amp_examples.html), [DataLoader and worker seeding](https://docs.pytorch.org/docs/2.5/data.html). No RAPIDS dependency was introduced into this PyTorch-native project.
