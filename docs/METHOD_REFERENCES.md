# Method references and implementation audit

Inspected 2026-11-25. Paper HTML method sections and repository files were inspected for planning. Phase 3/4 PyTorch ports now have objective, mask, finite-gradient, checkpoint and resume tests; numerical source comparisons are listed below. Repository clones in ignored `references/` are read-only comparison material, not runtime dependencies.

| Reference | Source and pinned revision | Audit status |
|---|---|---|
| Survey | [2412.12770v2](https://arxiv.org/html/2412.12770v2) | Taxonomy/context: ID-based, side-information and generative SR; no benchmark implementation |
| GRU4Rec | [1511.06939v4](https://arxiv.org/html/1511.06939v4); [original Theano code](https://github.com/hidasib/GRU4Rec/tree/a4ed5fbdba35bcc18bf1d4a9b76692ef462bb284) | Inspected session-parallel, loss and embedding options; our CE baseline is explicitly adapted |
| SASRec | [1808.09781v1](https://arxiv.org/html/1808.09781); [original code](https://github.com/kang205/SASRec/tree/e3738967fddab206d6eeb4fda433e7a7034dd8b1) | Inspected `model.py`, causal attention invocation, shifted targets and masked BCE |
| BERT4Rec | [1904.06690v2](https://arxiv.org/html/1904.06690v2); [original code](https://github.com/FeiSun/BERT4Rec/tree/615eaf2004abecda487a38d5b0c72f3dcfcae5b3) | Inspected README and `gen_data_fin.py` corruption/forced-last routines |
| TIGER | [2305.05065v3](https://arxiv.org/html/2305.05065v3) | Method inspected; official original implementation not verified; do not label a third-party port official |
| FENRec | [2412.11589v4](https://arxiv.org/html/2412.11589v4); [official code](https://github.com/uikdwnd/FENRec/tree/fd111eccc36bd37dac57b05133966548ab2547bb) | Inspected `src/datasets.py` soft labels and `src/trainers.py` hard-negative generation/loss |
| CRAMER | [2608.25370v2](https://arxiv.org/html/2608.25370v2) | Literature only: request-conditioned masks on a frozen recommender; different task from conventional next-item E1 |

## Reference checkout lock for Git publication

Verified directly from each nested repository's `origin` and `HEAD` on 2026-11-26:

| Local checkout | Original repository URL | Exact commit |
|---|---|---|
| BERT4Rec | https://github.com/FeiSun/BERT4Rec.git | `615eaf2004abecda487a38d5b0c72f3dcfcae5b3` |
| FENRec | https://github.com/uikdwnd/FENRec.git | `fd111eccc36bd37dac57b05133966548ab2547bb` |
| GRU4Rec | https://github.com/hidasib/GRU4Rec.git | `a4ed5fbdba35bcc18bf1d4a9b76692ef462bb284` |
| SASRec | https://github.com/kang205/SASRec.git | `e3738967fddab206d6eeb4fda433e7a7034dd8b1` |

Reconstruct these optional comparison sources with `python scripts/fetch_references.py`. The machine-readable lock is `configs/reference_repositories.json`. Use `--only FENRec` for source-equivalence tests alone, or `--destination PATH` for another directory. Existing origins/revisions and tracked modifications are checked before reuse; untracked test caches are tolerated. The script never resets or deletes existing files. Clones remain ignored under `references/`; they are neither vendored source nor Git submodules. Benchmark execution does not require installing upstream training dependencies. Respect each upstream license when using or redistributing its code. TIGER has no verified official source lock and is not downloaded by this script.

## SASRec equation and code map

Paper III-A–D specifies positional/item embeddings, causal self-attention, feedforward blocks and tied inner-product item scoring. III-E specifies shifted next-item targets and positive/negative sigmoid BCE, ignoring padding. Original `model.py` samples one negative per position and Adam beta2=0.98. The original evaluator scores sampled candidates; this benchmark deliberately uses full-catalogue evaluation. SASRec-CE changes the objective and must be labeled separately. Preserve the original normalization/residual/feedforward choices or explicitly record architecture changes before results.

GRU4Rec equations (2)-(5) define gated recurrence. Sections 3.1.1-3.1.3 specify session-parallel training, sampled outputs and ranking losses. Our named GRU4Rec-CE uses PyTorch `nn.GRU` gates, learned input embeddings, tied output weights, packed user-history batches, Adam and full-softmax CE. It is an explicitly modified baseline, not a numerical port of Theano recurrence/training or a reproduction of original ranking-loss results.

## BERT4Rec equation and code map

Equations (4)–(6) define the Transformer stack; (7) defines GELU projection and tied item output; (8) averages masked-item negative log probabilities. Section 3.6 describes Cloze learning and MASK inference. The pinned generator exposes `mask_prob` default 1.0; its comment about 80% masking is not the effective default. `create_masked_lm_predictions_force_last` replaces the final item with MASK. Our inference will append MASK to history without accepting the target as input. Test corruption protocol and exclude PAD/MASK output logits.

## FENRec audit and primary reference choice

Equations (6)–(8) normalize decay weights over the next three interactions. `_get_soft_label` in `src/datasets.py:369` builds that window. Equations (9)–(12) normalize anchor/negative mixtures, restore negative norms and stop gradients. `_get_enduring_hard_negatives` in `src/trainers.py:307` implements mixing and detachment. Equations (13)–(16) add similarity weighting and combine objectives.

The paper's (13)/(15) places a margin additively in the negative logit. Code at `src/trainers.py:379` multiplies the logit by `(similarity + m)`. The implemented primary branch explicitly follows the pinned code. Source warm-up, mixing/weight schedules, noise negatives and same-label masking are implemented. Training-only future windows replace upstream caches. All F0-F5 switches are tested; source loss values and gradients are compared numerically before and after warm-up. Backend and numerical differences are documented below.

## TIGER equation and component map

Section 3.1 defines residual nearest-codeword assignment, residual subtraction, summed code embeddings, reconstruction loss and codebook/commitment penalties with stop-gradient. It also requires collision disambiguation. Section 3.2 uses an encoder-decoder trained to generate next-item semantic tokens; retrieval uses beam decoding and item mapping. A compact text encoder or architecture change will be named TIGER-Lite. Encoder version, fit catalogue, tokenizer dimensions and official source availability remain unresolved. T0/T1/T2 are unimplemented, not approximated by scoring models.

## Implemented methods and deviations

MostPopular and all requested Phase 3/4 neural configurations are implemented. GRU4Rec-CE is an explicit CE adaptation; it is not a reproduction of original session-parallel training/loss. Real Beauty pilots are isolated under results/pilots. TIGER remains unimplemented. Preprocessing choices are benchmark-specific and documented in the protocol. No published scores are copied into result tables.


## Phase 3/4 implementation decisions

- **GRU4Rec-CE:** compact padded histories before a packed GRU; train all nonpadding shifted positions with full-catalogue CE and tied input/output embeddings. This is the named user-history adaptation, not original GRU4Rec's ranking loss or session-parallel optimizer protocol.
- **SASRec:** preserve scaled item embeddings, unnormalized keys/values with normalized queries, query residual, d-to-d ReLU feedforward blocks, pre-feedforward normalization/residual, final layer norm and tied item dot products. BCE draws one negative per valid position, excluding all of that user's training items. Adam beta2 is 0.98. CE changes only the objective. PAD and unused MASK classes never compete in output. Two extra reserved input IDs are allocated uniformly across backbones. The configured ffn_dim applies to post-norm backbones; original SASRec intentionally uses width d.
- **BERT4Rec:** post-norm attention, GELU feedforward, input normalization, tied GELU/projection/normalization output head, and appended MASK inference. We omit the constant segment embedding used by the generic upstream BERT implementation. Use left padding consistently; positions are absolute within the fixed window. Initialize normal rather than TF truncated normal. Dynamic fresh masks replace the original offline duplicated TFRecords; the explicit last-mask probability is 0.1. Use Adam with fixed learning rate and validation early stopping instead of the upstream warm-up/decay optimizer. Full-catalogue evaluation replaces sampled evaluation.
- **E2:** all A1-A4 now use the same post-norm attention-ablation backbone and output head. A1/A2 have causal shifted BCE/CE; A3/A4 have identical masked CE sampling and differ only in attention visibility. Parameter counts match across all four. This controlled experiment intentionally does not reuse E1 SASRec/BERT4Rec.
- **FENRec primary branch:** source-code weighting is selected by fenrec_loss_reference=code, resolving the margin discrepancy in favor of the pinned repository. F0/F1 disable both CL branches; F2/F4 use standard coarse/fine InfoNCE; F3/F5 enable source hard weighting, normalized mixtures and noise negatives after warm-up. Both CL branches use source mean reductions, so no extra factor of two is introduced. Same-target/cluster false negatives are masked. Same-target positive selection follows the source's final-different-history rule; dropout supplies another view when only a single matching history exists.
- **FENRec backend:** port the reference post-norm causal encoder and CE-on-final-prefix objective, not the distinct original SASRec block. Enumerate prefixes and subsequent windows from retained training events only; windows end at the training boundary. No upstream pickle cache is reused. Fit epoch-wise raw-centroid Lloyd k-means with sklearn (seed 1, 20 iterations, five restarts) rather than GPU FAISS. Query raw centers, then normalize positive centroid vectors. This backend change is explicit and prevents a claim of bitwise reproduction of the FAISS implementation. The optional paper_margin mode is a margin-placement sensitivity branch retaining the code's noise mechanism, not a claim of a separate faithful full-paper implementation.
- **Numerical safeguards:** zero padded queries/states and embedding rows; finite attention fill for all-pad rows; epsilon-safe vector normalization; logsumexp weighted contrastive denominators; detached synthetic/noise negatives. Normalized vector tests use nonzero vectors to match the source exactly. FENRec stays FP32. Phase 3 CUDA autocast forward/backward is tested; AMP is opt-in.
- **FENRec source validation:** test_reference_backbone.py transfers weights into the pinned source encoder and checks real-token outputs. test_fenrec.py extracts source functions by AST without importing their GPU-only dependencies, then compares mixed negatives, pre/post-warm-up loss values and gradients. Standalone manual formula tests still run without reference checkouts; source-comparison tests explicitly skip if checkouts are missing.
- **Run protocol:** best weights come from validation NDCG@10; last-epoch state includes optimizer, scaler and all RNGs. Resume is epoch-boundary exact under the same software/device configuration. Full test predictions are created only after stopping. Pilot truncation controls are hashed and rejected unless pilot=true. Test labels never enter model selection.
