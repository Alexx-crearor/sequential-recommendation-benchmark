# Danh mục hình và bảng cho báo cáo/slide

Dùng PDF vector cho báo cáo học thuật, PNG 300 DPI cho slide. Hình/bảng Phase 5C chỉ là validation tuning; E1/E2 là official test; runtime là log Kaggle chứ không dedicated profiling. Không dùng hình E2 partial lịch sử thay hình complete.

| ID | Caption đề xuất | Dữ liệu / PNG | PDF vector | Phạm vi |
|---|---|---|---|---|
| FIG01 | E1 test mean và sample SD | [mở](../results/figures/course_final/e1_test_metrics.png) | [PDF](../results/figures/course_final/e1_test_metrics.pdf) | official test/recorded dynamics; caption distinguishes SD and CI |
| FIG02 | E1 Recall/NDCG theo K | [mở](../results/figures/course_final/e1_recall_ndcg_at_k.png) | [PDF](../results/figures/course_final/e1_recall_ndcg_at_k.pdf) | official test/recorded dynamics; caption distinguishes SD and CI |
| FIG03 | E1 paired-user intervals | [mở](../results/figures/course_final/e1_paired_bootstrap.png) | [PDF](../results/figures/course_final/e1_paired_bootstrap.pdf) | official test/recorded dynamics; caption distinguishes SD and CI |
| FIG04 | E1 training/validation curves | [mở](../results/figures/course_final/e1_training_validation.png) | [PDF](../results/figures/course_final/e1_training_validation.pdf) | official test/recorded dynamics; caption distinguishes SD and CI |
| FIG05 | E2 đủ 12/12 điểm theo seeds | [mở](../results/figures/course_final/e2_complete_results.png) | [PDF](../results/figures/course_final/e2_complete_results.pdf) | official test/recorded dynamics; caption distinguishes SD and CI |
| FIG06 | E2 mean/SD ba seeds | [mở](../results/figures/course_final/e2_test_metrics.png) | [PDF](../results/figures/course_final/e2_test_metrics.pdf) | official test/recorded dynamics; caption distinguishes SD and CI |
| FIG07 | Ba E2 contrasts với paired 95% CI | [mở](../results/figures/course_final/e2_paired_contrasts.png) | [PDF](../results/figures/course_final/e2_paired_contrasts.pdf) | official test/recorded dynamics; caption distinguishes SD and CI |
| FIG08 | E2 training dynamics | [mở](../results/figures/course_final/e2_training_validation.png) | [PDF](../results/figures/course_final/e2_training_validation.pdf) | official test/recorded dynamics; caption distinguishes SD and CI |
| FIG09 | E2 runtime/latency/memory đã ghi | [mở](../results/figures/course_final/e2_recorded_efficiency.png) | [PDF](../results/figures/course_final/e2_recorded_efficiency.pdf) | official test/recorded dynamics; caption distinguishes SD and CI |
| FIG10 | E1 runtime/latency/memory đã ghi | [mở](../results/figures/course_final/recorded_efficiency.png) | [PDF](../results/figures/course_final/recorded_efficiency.pdf) | official test/recorded dynamics; caption distinguishes SD and CI |
| FIG11 | E8 error groups | [mở](../results/figures/course_final/e8_error_groups.png) | [PDF](../results/figures/course_final/e8_error_groups.pdf) | official test/recorded dynamics; caption distinguishes SD and CI |
| FIG12 | E8 phân đoạn history/popularity | [mở](../results/figures/course_final/e8_history_popularity.png) | [PDF](../results/figures/course_final/e8_history_popularity.pdf) | official test/recorded dynamics; caption distinguishes SD and CI |
| FIG13 | Phase 5C validation_curves | [mở](../results/figures/phase5c/validation_curves.png) | [PDF](../results/figures/phase5c/validation_curves.pdf) | validation tuning only; never final test |
| FIG14 | Phase 5C training_loss | [mở](../results/figures/phase5c/training_loss.png) | [PDF](../results/figures/phase5c/training_loss.pdf) | validation tuning only; never final test |
| FIG15 | Phase 5C trial_validation_comparison | [mở](../results/figures/phase5c/trial_validation_comparison.png) | [PDF](../results/figures/phase5c/trial_validation_comparison.pdf) | validation tuning only; never final test |
| FIG16 | Phase 5C selected_trial_resources | [mở](../results/figures/phase5c/selected_trial_resources.png) | [PDF](../results/figures/phase5c/selected_trial_resources.pdf) | validation tuning only; never final test |
| TAB01 | Trạng thái phạm vi đã đóng | [mở](../results/tables/course_final/course_status.csv) | — | aggregate evidence; retain stated split/units |
| TAB02 | Mean/SD E1 và E2 full/warm, mọi K | [mở](../results/tables/course_final/metric_summary.csv) | — | aggregate evidence; retain stated split/units |
| TAB03 | Mọi test metric từng seed | [mở](../results/tables/course_final/per_seed_metrics.csv) | — | aggregate evidence; retain stated split/units |
| TAB04 | Paired differences và CI | [mở](../results/tables/course_final/paired_bootstrap.csv) | — | aggregate evidence; retain stated split/units |
| TAB05 | Provenance, stopping và runtime từng run | [mở](../results/tables/course_final/runs.csv) | — | aggregate evidence; retain stated split/units |
| TAB06 | E2 hiệu năng tính toán mean/SD | [mở](../results/tables/course_final/e2_efficiency_summary.csv) | — | aggregate evidence; retain stated split/units |
| TAB07 | Phát hiện khoa học và limitation | [mở](../results/tables/course_final/scientific_findings.csv) | — | aggregate evidence; retain stated split/units |
| TAB08 | Hyperparameters chọn bằng validation | [mở](../results/tables/course_final/tuning_selected.csv) | — | aggregate evidence; retain stated split/units |
| TAB09 | E2 loss đầu/cuối, best epoch và stopping | [mở](../results/tables/course_final/e2_training_dynamics.csv) | — | aggregate evidence; retain stated split/units |
| TAB10 | SD của paired differences qua ba seeds | [mở](../results/tables/course_final/e2_contrast_seed_summary.csv) | — | aggregate evidence; retain stated split/units |
| TAB11 | E8 nhóm lỗi từng seed | [mở](../results/tables/course_final/e8_error_groups.csv) | — | aggregate evidence; retain stated split/units |
| TAB12 | E8 metrics theo history/popularity | [mở](../results/tables/course_final/e8_segments.csv) | — | aggregate evidence; retain stated split/units |
| TAB13 | E8 paired hit/miss | [mở](../results/tables/course_final/e8_pair_outcomes.csv) | — | aggregate evidence; retain stated split/units |
| TAB14 | 262 tệp cũ không thay đổi | [mở](../results/tables/course_final/prior_run_comparison.csv) | — | aggregate evidence; retain stated split/units |

CSV giữ số thực gốc; chỉ làm tròn khi dàn trang. Không tự thay SD N/A của MostPopular thành 0. CI là percentile paired-user, không phải sample SD. Các raw case examples/history/predictions không thuộc publication catalog. Các statements causal/global phải theo giới hạn trong COURSE_FINAL_REPORT_VI.md.
