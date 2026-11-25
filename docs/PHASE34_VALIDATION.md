# Phase 3/4 validation map

All original Phase 2 tests remain in the suite. New coverage:

| Requirement | Evidence |
|---|---|
| Training-only samples, negative exclusions, future windows | `test_neural.py`, `test_fenrec.py`, `scripts/verify_data.py` |
| Causal visibility and bidirectional right context | causal perturbation and bidirectional sensitivity tests |
| PAD handling and masked inference | all-pad finite states, zero PAD gradient, PAD embedding perturbation, MASK append tests |
| Objective and gradient health | CPU all models; CUDA all models and Phase 3 autocast; post-warm-up FENRec |
| FENRec equations/source | hand-calculated decay and mixture examples, source function value/gradient comparisons, source encoder output comparison |
| Ablations | all six switches, identical backbone parameter counts, disabled loss branches, contradictory switches rejected |
| Ranking/checkpoint | full-catalogue item batching versus scalar reference; restored score equivalence |
| Resume/skipping | each model compares interrupted/resumed epoch weights and loss/validation sequences exactly with uninterrupted training |
| Early stopping | controlled validation decline verifies patience and independent best/latest checkpoints |
| E2 control | identical capacity across the four attention/objective settings |

Source functions are read from the pinned local repository for comparison; no upstream training is run. In a portable checkout without references, those source-comparison tests skip visibly. Remaining standalone equation tests are not skipped. To restore the reference tests, clone `https://github.com/uikdwnd/FENRec.git` into `references/FENRec` and checkout `fd111eccc36bd37dac57b05133966548ab2547bb`.

`results/reports/environment_check.json` records actual CPU and CUDA tensor backward execution. `results/reports/beauty_data_audit.json` records real-data statistics, hashes, split integrity and per-model sample counts. `results/pilots/pilot_summary.csv` is generated from successful verified runs, not entered by hand. The final Phase 3/4 report records the measured pilot values and test totals after execution completes.
