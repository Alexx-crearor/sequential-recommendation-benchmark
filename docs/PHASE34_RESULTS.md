# Phase 3/4 execution report

Generated from verified artifacts on 2026-11-25. These are pipeline pilots, not final research findings.

## Completion and environment

Implemented MostPopular plus GRU4Rec-CE, SASRec-BCE/CE, BERT4Rec, all four capacity-matched E2 attention/objective variants, and FENRec F0–F5. The trainer supports best/latest checkpoints, validation early stopping, exact same-environment epoch-boundary resume, full-catalogue evaluation and GPU memory reporting. All changes extend the Phase 2 repository.

Local Python 3.10.11 uses PyTorch 2.5.1+cu121; CUDA 12.1 executes on a Quadro T1000 with 4 GiB. Both CPU and CUDA forward/backward checks passed. See `results/reports/environment_check.json` for the complete package inventory.

Beauty: 22,363 users, 12,101 items, 198,502 interactions and 153,776 training events. Metadata missing items: 0. Test warm coverage: 0.993829. Hash/split/chronology/core/per-model training-boundary checks passed. Split version: `2228499f23f01d419863d0f09b3671f098672471887c4fb37b85bae26d42f91e`.

## Tests

JUnit evidence: 98 tests, 0 failures, 0 errors, 0 skipped. The full suite includes CPU cases and six CUDA forward/backward cases; separate CPU and CUDA smoke selections also passed. FENRec mixed negatives and pre/post-warm-up loss values/gradients match extracted pinned source functions. Encoder weights transferred to the source produce matching real-token states. All six E4 variants have finite loss/gradients and identical backbone sizes. Every model has an exact interrupted/resumed weight/loss sequence comparison. Source commits and deviations are in `docs/METHOD_REFERENCES.md`.

## Real Beauty pilots, seed 7

Neural runs use two epochs and twenty batches of 64 per epoch. Validation selection uses the first 512 user IDs as an explicitly limited pilot subset. MostPopular uses all training data and full validation; its validation number is therefore not directly comparable to neural pilot subset values. Every run saves all 22,363 test users with exact full-catalogue ranks. Loss values are model-specific objectives and should not be compared across rows. MostPopular has no optimization loss or trainable parameter vector.

| Model | Final loss | Validation NDCG@10 | Training seconds | Batch-1 scoring ms | Peak GPU MiB | Parameters |
|---|---|---|---|---|---|---|
| mostpopular | N/A | 0.007941 | 0.000592 | 0.027625 | 0.000000 | 0 |
| gru4rec_ce | 9.682467 | 0.000000 | 6.505570 | 8.619010 | 109.843262 | 824512 |
| sasrec_bce | 1.367175 | 0.004446 | 1.470429 | 6.973275 | 101.895020 | 820032 |
| sasrec_ce | 9.146076 | 0.005775 | 1.233962 | 5.675650 | 128.458008 | 820032 |
| bert4rec | 9.262555 | 0.003134 | 1.606528 | 5.659440 | 123.656250 | 894277 |
| fenrec | 16.494477 | 0.000756 | 97.160618 | 4.887695 | 182.823242 | 877888 |

Scoring latency includes history encoding and full-catalogue scoring, with five warm-ups and twenty synchronized batch-1 repeats. It excludes sorting, disk I/O and Python evaluation bookkeeping; P95 and end-to-end test throughput are in each efficiency.json. MostPopular runs on CPU. GPU peaks are allocated PyTorch bytes, not driver-reserved or total system memory. The baseline latency is recorded as a separate analysis artifact because its completed Phase 2-style run is immutable.

FENRec's main two-epoch pilot preserves its twenty-epoch warm-up, so those training losses do not exercise enduring negatives. A separate one-epoch, four-batch F5 pilot explicitly sets warm_up_epochs=0; it verifies the post-warm-up path on real data. Its measured stage breakdown and nonzero coarse/fine losses are in `results/reports/fenrec_postwarm_summary.json`. Both pilots reach their deliberate training limits; none should be interpreted as converged evidence or used to rank methods scientifically.

Per-user predictions, validation histories, environments, best/latest checkpoints and source hashes remain in `results/pilots` and `results/pilots_postwarm`. Primary table source: `results/pilots/pilot_summary.csv`.

## Resource extrapolation

Same local hardware, batch 64, full Beauty training and validation every epoch, maximum 300 epochs per seed:

| Model | Estimated 300-epoch hours / seed |
|---|---:|
| bert4rec | 6.45 |
| gru4rec_ce | 10.77 |
| sasrec_bce | 5.72 |
| sasrec_ce | 5.67 |
| fenrec | 54.07 |

For the five neural E1 models on Beauty, this totals about 82.7 hours per seed or 248.0 hours for three seeds before early stopping. These are rough linear extrapolations from short pilots, not measured full runs. FENRec uses the supplemental separated clustering/optimization measurement; scaling its earlier combined clustering time by the batch ratio would be invalid and is intentionally left unset. Fixed cluster encoding/fitting cost is added once per epoch. The supplemental four-batch measurement has particularly high uncertainty. Normal source warm-up is cheaper than its post-warm-up branch; the estimate applies that post-warm-up rate conservatively throughout.

The defaults use batch 128, while these estimates hold batch 64 fixed. A batch-128 capacity/throughput pilot, full epochs and larger GPU measurements are needed before scheduling Kaggle. No reliable aggregate E1–E5 wall-clock estimate exists for Sports/Toys, other lengths, retention settings or tuning. The plan has 209 entries including later phases; E1–E5 excluding TIGER has 189 planned entries before exact-configuration reuse. Validation-only tuning and final test evaluation add cost.

## Remaining work and commands

No environment/data blocker remains for local Phase 3/4 development. Full-budget model convergence, bounded tuning automation, multi-seed statistical analysis, Sports/Toys pilots, final tables/figures/Vietnamese report and TIGER remain outstanding. FENRec is an explicitly documented source-code port using sklearn rather than FAISS clustering; it is not a claim of bitwise reproduction of every original training artifact. Kaggle commands are provided but have not been executed in an actual Kaggle session.

Exact E1–E5 commands, environment setup, data preparation, GPU selection, resume, completed-run skipping and ZIP export: `docs/KAGGLE_WORKFLOW.md`. No full training grid was launched.
