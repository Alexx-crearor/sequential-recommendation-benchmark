"""Final Vietnamese results, scientific findings and report-ready asset catalog."""
import argparse
from pathlib import Path
import pandas as pd
import _bootstrap
from write_phase5c_report import markdown_table
from src.analysis.master_session1 import MODELS, LABELS
from src.analysis.course_final import CONTRASTS, VARIANT_DESIGN, ARCHIVE_SHA256
from src.analysis.phase5c import read_json
from src.utils.artifacts import write_json


def records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def write_report(root):
    root = Path(root)
    tables = root / "results/tables/course_final"
    audit = read_json(tables / "artifact_audit.json")
    summary, runs = pd.read_csv(tables / "metric_summary.csv"), pd.read_csv(tables / "runs.csv")
    bootstrap = pd.read_csv(tables / "paired_bootstrap.csv")
    tuning = pd.read_csv(root / "results/tables/phase5c/selected_hyperparameters.csv")
    tuning["evaluation_split"] = "validation"
    tuning.to_csv(tables / "tuning_selected.csv", index=False)
    e2 = summary[(summary.experiment == "E2") & (summary.group == "full")]
    e1 = summary[(summary.experiment == "E1") & (summary.group == "full")]
    e2_rows, e1_rows = [], []
    per_seed = pd.read_csv(tables / "per_seed_metrics.csv")
    seed_contrasts = []
    for left, right in CONTRASTS:
        for group in ("full", "warm"):
            for metric in sorted(e2.metric.unique()):
                data = per_seed[(per_seed.experiment == "E2") & (per_seed.group == group) & (per_seed.metric == metric)]
                pivot = data.pivot(index="seed", columns="variant", values="value")
                for seed, delta in (pivot[left]-pivot[right]).items():
                    seed_contrasts.append({"left": left, "right": right, "group": group, "metric": metric, "seed": int(seed), "difference": float(delta)})
    seed_frame = pd.DataFrame(seed_contrasts)
    seed_frame.to_csv(tables / "e2_paired_seed_differences.csv", index=False)
    seed_summary = seed_frame.groupby(["left", "right", "group", "metric"]).difference.agg(["mean", "std", "count"]).reset_index()
    seed_summary.to_csv(tables / "e2_contrast_seed_summary.csv", index=False)
    for variant in ("A1", "A2", "A3", "A4"):
        data = e2[e2.variant == variant].set_index("metric")
        e2_rows.append([variant, *[f"{data.loc[m, 'mean']:.6f} ± {data.loc[m, 'sd']:.6f}" for m in ("recall@10", "ndcg@10", "mrr@10")]])
    for model in MODELS:
        data = e1[e1.model == model].set_index("metric")
        e1_rows.append([LABELS[model], *[f"{data.loc[m, 'mean']:.6f} ± {data.loc[m, 'sd']:.6f}" if data.loc[m, "n_runs"] > 1 else f"{data.loc[m, 'mean']:.6f} (SD N/A)" for m in ("recall@10", "ndcg@10", "mrr@10")]])
    findings, contrast_rows = [], []
    descriptions = {("A2", "A1"): "Next-item CE cao hơn next-item BCE trong cấu hình matched đã chạy",
                    ("A3", "A2"): "Masked-item CE causal thấp hơn next-item CE causal trong cấu hình đã chạy",
                    ("A4", "A3"): "Bidirectional masked-item cao hơn causal masked-item trong cấu hình đã chạy"}
    limits = {("A2", "A1"): "Loss form, sampled-vs-full negative coverage và gradient normalization khác nhau; không chỉ một công thức scalar",
              ("A3", "A2"): "Objective package đổi masking, vị trí/số lượng targets và mật độ supervision; không cô lập riêng corruption hay target density",
              ("A4", "A3"): "Matched attention contrast có điều kiện trên backbone/hyperparameters/budget này; không chứng minh bidirectional luôn tốt hơn causal"}
    for index, (left, right) in enumerate(CONTRASTS, 1):
        row = bootstrap[(bootstrap.experiment == "E2") & (bootstrap.group == "full") & (bootstrap.metric == "ndcg@10") & (bootstrap.left == left) & (bootstrap.right == right)].iloc[0]
        seed_sd = seed_frame[(seed_frame.left == left) & (seed_frame.right == right) & (seed_frame.group == "full") & (seed_frame.metric == "ndcg@10")].difference.std(ddof=1)
        contrast_rows.append([f"{left} − {right}", f"{row.difference:.6f}", f"{seed_sd:.6f}", f"[{row.ci_low:.6f}, {row.ci_high:.6f}]", "7,42,2026"])
        findings.append({"id": f"F{index}", "experiment": "E2", "contrast": f"{left}-{right}", "metric": "test_ndcg@10",
                         "difference": row.difference, "ci_low": row.ci_low, "ci_high": row.ci_high, "seeds": "7,42,2026",
                         "observation_vi": descriptions[(left, right)], "limitation_vi": limits[(left, right)],
                         "inference_scope": "pointwise user-bootstrap conditional on fitted seeds; no multiplicity correction"})
    pair = bootstrap[(bootstrap.experiment == "E1") & (bootstrap.group == "full") & (bootstrap.metric == "ndcg@10") & (bootstrap.left == "sasrec_ce") & (bootstrap.right == "fenrec")].iloc[0]
    findings.append({"id": "F4", "experiment": "E1", "contrast": "fenrec-sasrec_ce", "metric": "test_ndcg@10", "difference": -pair.difference,
                     "ci_low": -pair.ci_high, "ci_high": -pair.ci_low, "seeds": "7,42,2026",
                     "observation_vi": "FENRec cao hơn SASRec-CE trong E1 Beauty; kết quả E1 không thay đổi sau phiên cuối",
                     "limitation_vi": "Một dataset, ba seeds, kiến trúc/objective khác nhau; không quy kết phần tăng cho một thành phần FENRec vì E4 chưa chạy",
                     "inference_scope": "conditional pointwise interval; no causal component attribution"})
    groups = pd.read_csv(tables / "e8_error_groups.csv").groupby("category").fraction.agg(["mean", "std"])
    findings.append({"id": "F5", "experiment": "E8", "contrast": "all_six_incorrect", "metric": "fraction_at10", "difference": None,
                     "ci_low": None, "ci_high": None, "seeds": "7,42,2026",
                     "observation_vi": f"Trung bình {groups.loc['all_six_incorrect', 'mean']*100:.2f}% users không có hit@10 ở cả sáu mô hình trong từng seed",
                     "limitation_vi": "Top-K miss không tương đương không thích mọi item; các nhóm errors chỉ mô tả, không giải thích nhân quả",
                     "inference_scope": "descriptive error analysis from unchanged E1 predictions"})
    findings_frame = pd.DataFrame(findings)
    findings_frame.to_csv(tables / "scientific_findings.csv", index=False)
    write_json(tables / "scientific_findings.json", records(findings_frame))
    status = [{"component": name, "status": state, "completed_runs": count} for name, state, count in [
        ("Hyperparameter Tuning", "COMPLETE", 15), ("E1", "COMPLETE", 16), ("E2", "COMPLETE", 12),
        ("E3", "DEFERRED", 0), ("E4", "DEFERRED", 0), ("E5", "DEFERRED", 0), ("TIGER", "THEORY_ONLY", None),
        ("CRAMER", "THEORY_ONLY", None), ("E7 recorded analysis", "COMPLETE", None),
        ("E7 dedicated profiling", "DEFERRED", 0), ("E8", "COMPLETE", None)]]
    pd.DataFrame(status).to_csv(tables / "course_status.csv", index=False)
    write_json(tables / "course_results.json", {"training_closed": True, "source_archive_sha256": ARCHIVE_SHA256,
               "status": status, "tuning_validation_only": records(tuning), "official_test_summary": records(summary),
               "paired_bootstrap": records(bootstrap), "scientific_findings": records(findings_frame)})
    efficiency_rows = []
    e2_runs = runs[runs.experiment == "E2"].copy()
    e2_runs["training_seconds_per_epoch"] = e2_runs.training_seconds/e2_runs.epochs_completed
    for variant in ("A1", "A2", "A3", "A4"):
        data = e2_runs[e2_runs.variant == variant]
        for metric in ("epochs_completed", "training_seconds", "training_seconds_per_epoch", "validation_seconds", "inference_mean_ms", "inference_p95_ms", "peak_gpu_memory_mib"):
            efficiency_rows.append({"variant": variant, "metric": metric, "n_runs": len(data), "mean": data[metric].mean(), "sd": data[metric].std(ddof=1), "source": "archived_Kaggle_records_not_new_profiling"})
    pd.DataFrame(efficiency_rows).to_csv(tables / "e2_efficiency_summary.csv", index=False)
    dynamics = e2_runs.groupby("variant", as_index=False).agg(
        n_runs=("seed", "size"), initial_loss_mean=("initial_train_loss", "mean"), initial_loss_sd=("initial_train_loss", "std"),
        final_loss_mean=("final_train_loss", "mean"), final_loss_sd=("final_train_loss", "std"),
        best_epoch_mean=("best_epoch", "mean"), completed_epoch_mean=("epochs_completed", "mean"),
        epoch_capped_runs=("training_limit_reached", "sum"), patience_stopped_runs=("stopped_for_patience", "sum"))
    dynamics.to_csv(tables / "e2_training_dynamics.csv", index=False)
    efficiency_table = []
    for variant in ("A1", "A2", "A3", "A4"):
        data = e2_runs[e2_runs.variant == variant]
        efficiency_table.append([variant, f"{data.epochs_completed.mean():.1f} ± {data.epochs_completed.std(ddof=1):.1f}",
                                f"{data.training_seconds.mean()/60:.2f}", f"{data.training_seconds_per_epoch.mean():.3f}",
                                f"{data.inference_mean_ms.mean():.3f}", f"{data.peak_gpu_memory_mib.mean():.2f}"])
    new = runs[runs.new_in_final_session]
    new_logged = new.training_seconds.sum()+new.validation_seconds.sum()+new.checkpoint_seconds.sum()+new.test_evaluation_seconds.sum()
    capped = runs[(runs.experiment == "E2") & runs.training_limit_reached]
    capped_names = ", ".join(f"{r.variant}/seed {r.seed}" for r in capped.itertuples())
    text = f'''# Báo cáo thực nghiệm cuối cùng — Đồ án Data Mining

Ngày chốt: **27/11/2026**. **Phạm vi thực nghiệm môn học đã đóng; không còn training, tuning hoặc GPU profiling được yêu cầu.** Đây là hoàn thành phạm vi môn học, không phải hoàn thành grid nghiên cứu gốc.

## 1. Trạng thái cuối cùng và nguồn bằng chứng

{markdown_table(['Thành phần', 'Trạng thái', 'Số run đã hoàn thành'], [[s['component'], s['status'], s['completed_runs'] if s['completed_runs'] is not None else 'N/A'] for s in status])}

Grid gốc chỉ được giữ làm tham khảo: **28/127**, còn 99 jobs không thuộc phạm vi môn học đã chốt. E3/E4/E5 hoãn vì tài nguyên và phạm vi môn học; không chọn phạm vi theo test score. Không biến E3 retention 1.0 có thể reuse thành E3 SUCCESS khi chưa tạo artifact tương ứng. TIGER và CRAMER chỉ nghiên cứu lý thuyết, không có kết quả thực nghiệm của đồ án.

Nguồn mới: `{audit['source_archive']}`, SHA256 `{audit['archive_sha256']}`. Archive có **354 tệp**, gồm **28 SUCCESS**, **55 checkpoint** (28 best + 27 latest) và **306 artifacts có hash trong status**. Đã xác minh config fingerprints, weights/resume state, split, tất cả predictions/metrics; rehash dataset Beauty theo manifest lưu trong archive và kiểm tra chronology.

Đối chiếu **262 tệp của 22 run cũ** với ZIP phiên trước: **0 thay đổi**, kể cả checkpoint, predictions, metrics, config và log. Phase 5C được kiểm tra lại hash ZIP và toàn bộ **182 raw files**; năm selections trong master plan khớp chính xác các cấu hình validation đã chốt. Không trộn validation tuning với final test scores.

ZIP mới được giữ tại `artifacts/e2_final/`; bản gốc ngoài repository vẫn giữ nguyên. Import tái sử dụng **{audit['deduplicated_prior_members']} tệp nội dung trùng** với phiên trước bằng hard link khi cùng filesystem, khoảng **{audit['reused_prior_logical_bytes']/1048576:.1f} MiB logical bytes**, thay vì giải nén lại toàn bộ. Các tệp raw được xem là immutable; không sửa tại chỗ vì hard links chia sẻ bytes. Nếu filesystem không hỗ trợ hard link, importer mới copy riêng các tệp cần thiết. Release mới chỉ đóng gói sáu run E2 bổ sung, không repack 22 run cũ.

## 2. Phương pháp và nguyên tắc thống kê

Beauty có 22.363 users, 12.101 items, 198.502 interactions; train 153.776 events, validation/test mỗi tập một target/user. Test history là train + validation; không đưa target test vào encoder. Full catalogue, seen-history mask, target exception, PAD/MASK exclusion và item-ID tie ordering được giữ nguyên. Tất cả 22.363 test users được giữ; warm cohort có 22.225 users. Split SHA: `{audit['split_version']}`.

Checkpoint chọn bằng validation NDCG@10, không chọn bằng test. Neural E1 và mỗi E2 variant có ba seeds 7/42/2026. Báo cáo mean và **sample SD, ddof=1**; MostPopular là predictor xác định chỉ chạy một lần, SD là N/A chứ không điền 0 giả. Recall/NDCG@5/10/20 và MRR@10 đều có full/warm trong CSV/JSON.

Paired bootstrap ghép đúng user/target/split, lấy trung bình contribution của mỗi user qua ba seeds trước khi resample users. 10.000 resamples, RNG seed 42, percentile 95% CI; implementation multinomial trên các vector difference trùng nhau có cùng phân phối với empirical row bootstrap, không dùng xấp xỉ normal. E2 có đủ ba contrast A2−A1, A3−A2, A4−A3 cho mọi metric ở cả full/warm cohorts. CI **pointwise, chưa điều chỉnh đa so sánh**, điều kiện trên các mô hình/seeds đã fit; không phải toàn bộ retraining uncertainty hoặc p-value confirmatory.

Tham khảo phương pháp, deviations so với papers, môi trường và giới hạn tại [tài liệu phương pháp/tái lập](EXPERIMENT_METHODS_REPRODUCIBILITY_VI.md) và [METHOD_REFERENCES.md](METHOD_REFERENCES.md).

## 3. Phase 5C và E1 đã chốt

Phase 5C gồm 15 trial validation-only, 3 trial/mô hình, seed 7; đây là bounded search hoàn thành, không phải exhaustive hyperparameter optimization. Các selections được giữ nguyên. Bảng `tuning_selected.csv` ghi rõ `evaluation_split=validation`; bảng E1/E2 là test. Không so điểm validation như final generalization performance.

{markdown_table(['Mô hình E1', 'Recall@10 mean ± SD', 'NDCG@10 mean ± SD', 'MRR@10 mean ± SD'], e1_rows)}

E1 không thay đổi sau phiên cuối, vì toàn bộ artifact của nó trùng byte với phiên trước. FENRec − SASRec-CE có test NDCG@10 difference **{-pair.difference:.6f}**, CI **[{-pair.ci_high:.6f}, {-pair.ci_low:.6f}]** có điều kiện như trên. Không quy kết mức tăng này cho riêng soft labels hoặc hard negatives vì E4 component ablation đã hoãn.

## 4. E2 đầy đủ: bốn variants × ba seeds

| Variant | Attention | Training objective |
|---|---|---|
| A1 | Causal | Next-item BCE |
| A2 | Causal | Next-item CE |
| A3 | Causal | Masked-item CE |
| A4 | Bidirectional | Masked-item CE |

Tất cả dùng matched post-norm ablation backbone/head, cùng số tham số và cùng các thiết lập không phải yếu tố đang xét: hidden/embedding dim 64, hai layers, hai heads, max length 50, batch 64, LR 0.0005, dropout 0.5, mask ratio 0.15; max 120 epochs, patience 10 validation checks, validation mỗi hai epochs cùng lần đầu/cuối. Không dùng E1 original SASRec/BERT4Rec làm alias cho E2.

{markdown_table(['Variant', 'Recall@10 mean ± SD', 'NDCG@10 mean ± SD', 'MRR@10 mean ± SD'], e2_rows)}

{markdown_table(['Contrast', 'Test NDCG@10 difference', 'SD paired differences qua seeds', 'Paired-user 95% CI', 'Seeds'], contrast_rows)}

SD của ba seed-level paired differences được báo riêng trong `e2_contrast_seed_summary.csv`; nó không tương đương user-bootstrap CI. Đặc biệt, CI hẹp trên 22.363 users không biến ba fitted seeds thành bằng chứng chắc chắn về toàn bộ retraining distribution.

- **A1 vs A2:** trong cấu hình đã chạy, next-item CE có kết quả cao hơn sampled BCE. Contrast bao gồm khác biệt coverage negatives/full softmax, gradient scaling và normalization; không cô lập một công thức loss scalar và không chứng minh CE luôn tốt hơn BCE ở mọi recommender.
- **A2 vs A3:** masked-item causal có kết quả thấp hơn next-item causal ở đây. Cả hai dùng CE nhưng khác corruption, vị trí/mật độ supervision và context của targets. Vì vậy không quy toàn bộ difference cho riêng masking hoặc nói masked objectives nói chung kém hơn.
- **A3 vs A4:** bidirectional masked-item cao hơn causal masked-item trong matched setup này. Đây là contrast attention hẹp hơn, nhưng vẫn điều kiện trên backbone/hyperparameters/validation stopping cụ thể; không suy ra causal attention không cần thiết hoặc bidirectional tốt hơn mọi next-item model. A2 causal vẫn có mean cao nhất trong bốn E2 variants, không có nghịch lý vì mục tiêu training khác nhau.

Các nhận xét trên mô tả kết quả thực tế có điều kiện, không phải causal law phổ quát. Không tuning lại hyperparameters sau khi xem test.

Báo cáo lịch sử `MASTER_SESSION1_ANALYSIS_VI.md` chỉ có sáu E2 run, trong đó A4−A3 dùng seed 7 và interval chứa 0. Phân tích hiện tại dùng đủ ba seeds cho cả ba contrasts; không sửa số cũ để làm nó giống bằng chứng đầy đủ. Đây là bản kết quả thay thế để viết báo cáo cuối cùng, còn bằng chứng từng phần vẫn giữ riêng.

![E2 test](../results/figures/course_final/e2_test_metrics.png)
![E2 paired contrasts](../results/figures/course_final/e2_paired_contrasts.png)

## 5. Training dynamics và convergence

E2 có **{len(capped)}/12** run chạm max epochs: **{capped_names}**; các run E2 còn lại dừng theo patience. Những run chạm cap vẫn là pipeline SUCCESS nhưng không được gọi là đã hội tụ toàn cục. Best checkpoint có thể ở epoch sớm hơn latest; cả hai được bảo toàn. Validation curves chỉ chứa các lần đánh giá thực sự, không dùng giá trị carry-forward ở epoch bỏ qua như measurement mới.

Loss BCE/next-CE/masked-CE không cùng thang đo; không xếp hạng model từ absolute training loss. So sánh learning curves chủ yếu trong từng variant qua seeds. Early stopping dẫn đến số epochs khác nhau; total training cost vừa phản ánh thời gian mỗi epoch vừa phản ánh stopping dynamics.

{markdown_table(['Variant', 'Loss đầu mean', 'Loss cuối mean', 'Best epoch mean', 'Epochs mean', 'Số run chạm cap'], [[r.variant, f'{r.initial_loss_mean:.5f}', f'{r.final_loss_mean:.5f}', f'{r.best_epoch_mean:.1f}', f'{r.completed_epoch_mean:.1f}', int(r.epoch_capped_runs)] for r in dynamics.itertuples()])}

![E2 dynamics](../results/figures/course_final/e2_training_validation.png)

## 6. Hiệu năng E7 hiện có, không đo thêm

{markdown_table(['Variant', 'Epochs mean ± SD', 'Training phút mean', 'Giây/epoch mean', 'Scoring ms mean', 'Peak allocated MiB mean'], efficiency_table)}

Số liệu lấy từ Kaggle `efficiency.json`/history, không đo lại trên máy cá nhân. Latency có 5 warm-up và 20 repeats, batch 1, history encoding + full-catalogue scoring, **không sorting/ranking/disk I/O**. SD giữa ba run của mean latency không phải SD của 20 repeats. Peak là allocated memory, không phải tổng VRAM. Neural dùng Tesla T4; MostPopular CPU fit/scoring không tương đương một GPU training workload.

Sáu run mới có training **{new.training_seconds.sum():.2f} giây**, validation **{new.validation_seconds.sum():.2f} giây**; tổng các components training/validation/checkpoint/test được ghi khoảng **{new_logged:.2f} giây**. Người thực hiện báo cáo phiên khoảng **5.667 giây (~94,45 phút)**; archive không lưu một wall-clock field đủ để xác minh độc lập số này. Component sum không gồm mọi setup/restore/export/verification overhead. Không cộng lại clustering FENRec vì nó đã nằm trong training time của E1.

**E7 phân tích số đo sẵn có COMPLETE; dedicated GPU profiling DEFERRED.** Không suy ra throughput chuẩn hóa, memory limit hoặc latency end-to-end từ những measurements không có trong artifacts.

![Recorded E2 efficiency](../results/figures/course_final/e2_recorded_efficiency.png)

## 7. E8 và bộ kết quả thống nhất

E8 được tính lại từ saved E1 predictions, giữ nguyên kết quả lịch sử: all-six-miss@10 khoảng **{groups.loc['all_six_incorrect', 'mean']*100:.2f}%** users trung bình qua seeds. Hit/miss pair groups có thể overlap. Sequence-length/popularity bins được định nghĩa cố định; target popularity chỉ đếm train events. Ví dụ chi tiết dùng quy tắc cố định ba user ID nhỏ nhất mỗi nhóm/seed và chỉ lưu riêng tư; không cherry-pick test-score extremes.

JSON `course_results.json` phân tách tuning-validation và official-test, kèm trạng thái phạm vi. `scientific_findings.csv/json` cung cấp phát hiện kèm limitation. [Danh mục assets](REPORT_ASSET_CATALOG.md) liệt kê hình/bảng, captions, PNG/PDF hoặc CSV để đưa vào báo cáo/slide. Kết quả E2 từng phần và mọi ZIP/report cũ vẫn còn trong namespace `master_session1`, không bị sửa thành bằng chứng giả của thiết kế hoàn chỉnh.

## 8. Threats to validity và những điều không được kết luận

1. **External validity:** một category Beauty, một quy tắc lọc/split và ba seeds; không có bằng chứng tương tự cho Sports/Toys, dữ liệu khác hoặc triển khai online. Next-item hit không đo causal user satisfaction.
2. **Construct/implementation validity:** GRU4Rec-CE, SASRec-CE và các backbone có adaptations được công bố trong method audit; không tuyên bố tái lập bitwise original papers. E2 matched backbone khác E1 originals. FENRec dùng source-code loss branch và sklearn Lloyd thay GPU FAISS.
3. **Optimization validity:** E2 dùng cùng hyperparameters, không tối ưu riêng từng objective; batch/mask ratio, target density và số epochs dừng khác nhau có thể ảnh hưởng ranking. A1/A4 có runs chạm cap; không tăng budget sau test.
4. **Statistical validity:** sample SD chỉ từ ba neural seeds; user bootstrap có điều kiện trên fitted seeds, không đầy đủ retraining uncertainty. Nhiều contrasts/metrics, pointwise CI chưa điều chỉnh; không báo p-value giả hoặc family-wide significance.
5. **Measurement validity:** runtime nhúng không thay dedicated profiling; phần cứng/clock/CPU load, caching và history được đo không được kiểm soát như một benchmark đầy đủ. Không so CPU baseline với GPU neural như phép đo kiến trúc công bằng.
6. **Evidence validity:** SHA256 xác minh bytes và tính nhất quán artifacts, không chứng thực mọi mã đã chạy. Archive có driver/plan nhưng không có complete training-source snapshot attestation.
7. **Missing experiments:** E3/E4/E5 DEFERRED nên không kết luận robustness theo retention/length hoặc attribution thành phần FENRec. TIGER/CRAMER THEORY_ONLY, không có score hoặc checkpoint thực nghiệm của chúng.

## 9. Đóng phạm vi và tái lập

`course_scope.yaml` và `course_jobs.json` ghi **training_closed=true**, 28 completed E1/E2 requests, **0 requested remaining runs**. Scope lịch sử phiên 1 được lưu trong `configs/experiments/history/`. Driver tiếp tục cũ từ chối `--run` ngay cả trên Kaggle; rerun historical analysis không được mở lại scope hiện tại. Không có kế hoạch hoặc lệnh để chạy thêm training/tuning/GPU profiling cho đồ án đã chốt.

```bash
python scripts/analyze_course_final.py
python scripts/gen_fig_course_final.py
python scripts/write_course_final_report.py
```

Raw audit cần ba ZIP riêng tư có đúng hash tại `artifacts/phase5c/`, `artifacts/master_session1/`, `artifacts/e2_final/` và đúng bundle Beauty; không có Downloads hard-code. Dữ liệu trùng được tái sử dụng, không ghi đè artifacts khác hash. Public notebook mode dùng tables đã versioned và chạy được từ clean clone không có raw ZIP/dataset. Bật FULL_AUDIT để tính lại từ bằng chứng gốc.

Checkpoint final phân phối bằng **composite manifest**: Release phiên 1 giữ nguyên 22 run; Release `e2-final-checkpoints-v1` chỉ chứa 6 run mới. Sau clone:

```bash
python scripts/fetch_course_final_checkpoints.py
python scripts/fetch_course_final_checkpoints.py --with-resume
```

Gói ghép cho 28 best checkpoints hoặc thêm 27 latest states, kiểm tra SHA256 từng ZIP và từng tệp. Không repack lại toàn bộ weights cũ; Git history chỉ chứa source/configs/notebooks/bảng/hình/manifests, không chứa ZIP gốc/dataset/raw predictions/checkpoint lớn/credentials. Không xóa bất kỳ artifact hợp lệ của các phiên trước.
'''
    (root / "docs/COURSE_FINAL_REPORT_VI.md").write_text(text, encoding="utf-8")
    build_catalog(root)
    return root / "docs/COURSE_FINAL_REPORT_VI.md"


def build_catalog(root):
    root = Path(root)
    figures = {"e1_test_metrics": "E1 test mean và sample SD", "e1_recall_ndcg_at_k": "E1 Recall/NDCG theo K",
               "e1_paired_bootstrap": "E1 paired-user intervals", "e1_training_validation": "E1 training/validation curves",
               "e2_complete_results": "E2 đủ 12/12 điểm theo seeds", "e2_test_metrics": "E2 mean/SD ba seeds",
               "e2_paired_contrasts": "Ba E2 contrasts với paired 95% CI", "e2_training_validation": "E2 training dynamics",
               "e2_recorded_efficiency": "E2 runtime/latency/memory đã ghi", "recorded_efficiency": "E1 runtime/latency/memory đã ghi",
               "e8_error_groups": "E8 error groups", "e8_history_popularity": "E8 phân đoạn history/popularity"}
    catalog = []
    for index, (name, caption) in enumerate(figures.items(), 1):
        base = f"results/figures/course_final/{name}"
        if not all((root / (base+extension)).is_file() for extension in (".png", ".pdf")):
            raise FileNotFoundError("Generate final figures first: " + base)
        catalog.append({"id": f"FIG{index:02d}", "type": "figure", "caption_vi": caption, "path": base+".png", "vector_pdf": base+".pdf", "scope": "official test/recorded dynamics; caption distinguishes SD and CI"})
    for name in ("validation_curves", "training_loss", "trial_validation_comparison", "selected_trial_resources"):
        catalog.append({"id": f"FIG{len(catalog)+1:02d}", "type": "figure", "caption_vi": "Phase 5C " + name,
                        "path": f"results/figures/phase5c/{name}.png", "vector_pdf": f"results/figures/phase5c/{name}.pdf", "scope": "validation tuning only; never final test"})
    table_names = {"course_status": "Trạng thái phạm vi đã đóng", "metric_summary": "Mean/SD E1 và E2 full/warm, mọi K",
                   "per_seed_metrics": "Mọi test metric từng seed", "paired_bootstrap": "Paired differences và CI",
                   "runs": "Provenance, stopping và runtime từng run", "e2_efficiency_summary": "E2 hiệu năng tính toán mean/SD",
                   "scientific_findings": "Phát hiện khoa học và limitation", "tuning_selected": "Hyperparameters chọn bằng validation",
                   "e2_training_dynamics": "E2 loss đầu/cuối, best epoch và stopping",
                   "e2_contrast_seed_summary": "SD của paired differences qua ba seeds",
                   "e8_error_groups": "E8 nhóm lỗi từng seed", "e8_segments": "E8 metrics theo history/popularity",
                   "e8_pair_outcomes": "E8 paired hit/miss", "prior_run_comparison": "262 tệp cũ không thay đổi"}
    for index, (name, caption) in enumerate(table_names.items(), 1):
        catalog.append({"id": f"TAB{index:02d}", "type": "table", "caption_vi": caption,
                        "path": f"results/tables/course_final/{name}.csv", "vector_pdf": None, "scope": "aggregate evidence; retain stated split/units"})
    data = pd.DataFrame(catalog)
    data.to_csv(root / "results/tables/course_final/asset_catalog.csv", index=False)
    body = "# Danh mục hình và bảng cho báo cáo/slide\n\nDùng PDF vector cho báo cáo học thuật, PNG 300 DPI cho slide. Hình/bảng Phase 5C chỉ là validation tuning; E1/E2 là official test; runtime là log Kaggle chứ không dedicated profiling. Không dùng hình E2 partial lịch sử thay hình complete.\n\n"
    body += markdown_table(["ID", "Caption đề xuất", "Dữ liệu / PNG", "PDF vector", "Phạm vi"],
                           [[r["id"], r["caption_vi"], f"[mở](../{r['path']})", f"[PDF](../{r['vector_pdf']})" if r["vector_pdf"] else "—", r["scope"]] for r in catalog])
    body += "\n\nCSV giữ số thực gốc; chỉ làm tròn khi dàn trang. Không tự thay SD N/A của MostPopular thành 0. CI là percentile paired-user, không phải sample SD. Các raw case examples/history/predictions không thuộc publication catalog. Các statements causal/global phải theo giới hạn trong COURSE_FINAL_REPORT_VI.md.\n"
    (root / "docs/REPORT_ASSET_CATALOG.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    print(write_report(parser.parse_args().root))
