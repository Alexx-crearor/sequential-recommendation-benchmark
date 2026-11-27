# Phương pháp, tái lập và giới hạn — phạm vi đồ án đã đóng

Đây là tài liệu phương pháp cho kết quả cuối cùng của đồ án Data Mining. Training cuối cùng hoàn thành ngày 27/11/2026; không yêu cầu chạy thêm training, tuning hoặc GPU profiling. Hoàn thành môn học không tương đương hoàn thành toàn bộ master training plan 127 jobs ban đầu.

Con số 127 là training plan trong master export trước khi đóng scope; các Full/Core configuration grids lớn hơn trong source vẫn được giữ làm thiết kế nghiên cứu tham khảo, không phải các run đã thực hiện hoặc nghĩa vụ còn lại của đồ án.

## 1. Bộ dữ liệu và giao thức

Amazon Beauty sau preprocessing: 22.363 users, 12.101 items, 198.502 events; 153.776 train events và 22.363 targets cho mỗi validation/test. Dùng implicit feedback, iterative 5-core, giữ lần user–item đầu tiên, chronology và tie-break được khai báo trong manifest. Per-user leave-two-out: mục kế cuối validation, cuối cùng test. Test input history gồm train + validation, không chứa target test. Các kiểm tra chronology, user coverage, event separation và SHA256 chạy lại trên bundle dùng trong analysis.

Bundle identity: `298d55546282277e0e2ba77a752e8be10e9067b1a05fee87ab09b6a5bffed6da`.

Split fingerprint: `2228499f23f01d419863d0f09b3671f098672471887c4fb37b85bae26d42f91e`.

Evaluation dùng **toàn catalogue**, loại PAD/MASK khỏi output, loại items đã thấy nhưng giữ target exception, tie scores theo item ID. Không thay bằng sampled evaluation hoặc approximate retrieval. Full metrics bao gồm mọi test target; warm-only cohort gồm 22.225 users, 138 cold targets vẫn ở full metrics. Recall/NDCG@5/10/20 và MRR@10 là one-positive-target metrics; no-hit không có nghĩa user không thích các item gợi ý.

## 2. Model, tuning và checkpoint selection

Phase 5C gồm 15 trials, ba trials/mô hình (GRU4Rec-CE, SASRec-BCE/CE, BERT4Rec, FENRec), seed 7, validation-only. Các winner được chọn bằng validation NDCG@10 với earliest-trial tie break; không dùng test metrics. Đây là bounded search, không exhaustive optimization. Các trial chạm budget vẫn được giữ; không tự gia hạn sau khi xem test.

E1 dùng năm neural models × seeds 7/42/2026 và MostPopular một lần (16 runs). Các hyperparameters được giữ theo selections Phase 5C. MostPopular không có stochastic training nên không nhân bản ba seeds và không gán sample SD bằng 0 giả. Neural checkpoint chọn theo validation NDCG@10, latest weights/optimizer/scaler/RNG giữ riêng để bảo toàn provenance; best epoch có thể sớm hơn epoch cuối.

E2 có 12 runs, cùng matched post-norm attention-ablation backbone/head, hidden dim 64, hai layers/heads, sequence length 50, batch 64, LR 0.0005, dropout 0.5 và mask ratio 0.15. Chỉ thay attention/objective theo A1 causal next-BCE, A2 causal next-CE, A3 causal masked-CE, A4 bidirectional masked-CE. Các config ngoài factor/seed và số parameters được kiểm tra giống nhau. Max 120 epochs; validation mỗi hai epochs cùng lần đầu/cuối; patience 10 validation checks; minimum-epoch gate được giữ. Không tối ưu lại từng objective sau khi xem test.

Không dùng E1 original SASRec/BERT4Rec như alias cho E2. GRU4Rec-CE là user-history CE adaptation, SASRec-CE đổi objective, BERT4Rec dùng dynamic corruption và các adaptations đã công bố. FENRec bám source-code loss branch, dùng sklearn Lloyd thay GPU FAISS. [METHOD_REFERENCES.md](METHOD_REFERENCES.md) ghi nguồn pinned và implementation deviations; không tuyên bố bitwise reproduction của original papers.

## 3. Tổng hợp và uncertainty

Mỗi neural setting có ba training seeds. Bảng dùng arithmetic mean và sample SD `ddof=1`, không dùng population SD hoặc gọi SD là confidence interval. MostPopular n=1, SD undefined/null. Không dùng trials có hyperparameters khác nhau như repeated seeds để tính variance.

Paired bootstrap ghép cùng user/target/split, lấy trung bình per-user contributions qua ba fitted seeds rồi resample users với replacement. 10.000 resamples, RNG seed 42, percentile 2.5/97.5. Multinomial multiplicities trên các vector contribution-difference trùng nhau có cùng phân phối empirical row bootstrap; các metric columns chia sẻ replicate multiplicities. Không coi user×seed là độc lập. Baseline predictor dùng cố định một lần.

E1 giữ 15 model pairs × 7 metrics trên full cohort; E2 có ba contrasts A2−A1, A3−A2, A4−A3 × 7 metrics trên full và warm cohorts. Intervals là pointwise/unadjusted và conditional on fitted seeds, không toàn bộ retraining uncertainty. Không báo confirmatory p-values, không diễn giải cả họ comparisons có FWER 5%, không suy ra statistical significance phổ quát từ việc một interval không chứa 0. Sample SD ba seeds vẫn được báo riêng.

## 4. Ba contrasts E2 được phép diễn giải thế nào

- **A2−A1:** khác biệt next-item CE và sampled BCE dưới cùng cấu hình. Bao gồm full-vs-sampled negatives, normalization và gradient noise; không cô lập riêng công thức scalar của loss.
- **A3−A2:** khác biệt gói next-item/masked-item supervision khi cùng causal attention. Input corruption, vị trí/mật độ targets và số updates hữu hiệu khác nhau; không quy toàn bộ difference cho riêng mask token.
- **A4−A3:** so sánh attention visibility khi cùng masked-item procedure và backbone. Đây là ablation hẹp hơn nhưng chỉ có giá trị trong setting/hyperparameters/budget đã chạy. Không suy ra bidirectional là nguyên nhân tốt hơn mọi causal model, nhất là khi A2 có objective khác.

Early stopping và cap khiến completed epochs khác nhau. Cùng max-epoch cap không đồng nghĩa cùng optimizer work hoặc computational cost. Một số A1/A4 chạm cap; pipeline SUCCESS không chứng minh hội tụ tối ưu. Chỉ một dataset, ba seeds và một capacity setting; không ngoại suy online user satisfaction hoặc category khác.

## 5. Hiệu năng và E8

Hiệu năng chỉ từ history/efficiency records Kaggle: training/validation/test seconds, parameter counts, peak allocated GPU memory và embedded batch-1 latency. Latency gồm encoding/scoring toàn catalogue, không sorting/disk I/O, 5 warm-up/20 repeats. Không có repeat-level raw timing distribution đủ để dựng SD mới từ repeats. SD giữa run means phải được ghi đúng là between-run SD. Baseline CPU fit/scoring không tương đương neural GPU workload; T4 clock/background CPU load và power không được kiểm soát như một dedicated profiling study. FENRec clustering đã nằm trong training time, không cộng hai lần. Component sum không phải session elapsed time.

E7 recorded-data analysis COMPLETE, dedicated GPU profiling DEFERRED. Không benchmark hoặc profile lại để lấp dữ liệu thiếu. Dữ liệu Phase 5A trên T1000 là lịch sử tối ưu/pilot, không đưa vào bảng T4 chính thức.

E8 dùng saved E1 predictions, không gọi model inference. Hit@10 groups và bins history/popularity được định nghĩa cố định; target popularity chỉ từ train interactions. Pairwise error groups có thể overlap. Per-user examples chọn theo quy tắc cố định, giữ riêng tư; không cherry-pick cases theo extreme score differences. Những kết quả mô tả không chứng minh cơ chế nhân quả. Do E1 files không thay đổi, E8 được giữ nhất quán qua snapshot mới.

## 6. Nguồn và lưu trữ

| Bằng chứng | SHA256 |
|---|---|
| Phase 5C ZIP | `7861c626561fc1d7914abb8ee5a7eb4d552a932aca5dfbc54419648efb0176fa` |
| Master session 1 ZIP | `9f6b30dd3d29750b21ef64ddd7c8fea88c66d1c3ca52ea44a9809b3cc323cfab` |
| E2 final ZIP | `4bc3da79817ad1463f1c4ff16b2e9aa13efbad8e9caad20f9368cfb690acdc39` |

Hash là kiểm tra bytes/provenance, không phải attestation đầy đủ của mọi code đã chạy. Archives có driver/plan và config, nhưng không complete training-source snapshot. Không sửa source checkpoint để “chuẩn hóa” hash.

Raw imports content-addressed, preflight xung đột trước khi ghi. Dữ liệu giống hệt dùng hard links khi cùng filesystem; không sửa raw files tại chỗ. Ngoài ZIP canonical duy nhất trong repository, không tạo bản repack toàn bộ 28 models. Old checkpoint Release giữ nguyên; Release mới chỉ sáu E2 runs bổ sung. Git chỉ chứa source, configs, tables, figures, clean notebooks và hash manifests. Raw ZIP/dataset/predictions/checkpoints/logs/credentials bị ignore. Bằng chứng E2 partial lịch sử vẫn nằm riêng dưới master_session1 và Git revision trước, không bị đổi số để giống thiết kế complete.

## 7. Tái lập local analysis từ Git

Python 3.10/3.11; môi trường PyTorch đã xác minh là 2.5.1+cu121, Kaggle Python 3.11/Tesla T4. Dependencies/optional notebook dependencies được pin trong `requirements-lock.txt` và `requirements-analysis.txt`. CPU đủ cho audit/statistics/plotting; không cần GPU mới cho phân tích.

```bash
python -m pip install -r requirements-analysis.txt --extra-index-url https://download.pytorch.org/whl/cu121
python scripts/analyze_course_final.py
python scripts/gen_fig_course_final.py
python scripts/write_course_final_report.py
python -m pytest -q
```

Đặt original ZIPs tại `artifacts/phase5c/`, `artifacts/master_session1/`, `artifacts/e2_final/`, và đúng dataset bundle tại `data/processed/beauty/<bundle-id>/`. CLI có `--root`, `--archive`, `--bundle`; không hard-code Downloads. Nếu prior raw chưa có, audit hydrate từ original ZIP đúng hash; không cần rerun training/tuning hoặc historical experiment jobs. Toàn bộ bước chạy trên máy cá nhân, không Kaggle.

Public summary notebooks chạy được từ clean clone không có private ZIP/dataset; bật FULL_AUDIT để tái tính từ originals. Source/archive hashes và scope được kiểm tra; aggregate-only mode không được gọi là một fresh raw re-audit. `gen_fig_course_final.py` tạo PNG 300 DPI và PDF vector; `REPORT_ASSET_CATALOG.md` ghi captions và paths. Nội dung numerical reproducible; font/software/platform khác có thể làm bytes hình khác nhau, không thay đổi metric values.

```bash
python scripts/fetch_course_final_checkpoints.py
python scripts/fetch_course_final_checkpoints.py --with-resume
```

Downloader xác minh ZIP/member SHA256, tái sử dụng old Release và delta mới; 28 best + 27 latest states, không tải raw predictions/dataset. MostPopular `.pt` là NPZ dùng `MostPopular.load`; neural dùng `SequentialModel.load(..., device='cpu')`. Chỉ load/inspect checkpoint, không chạy thêm evaluation/profiling cho phạm vi đã đóng. Git clone không tự fetch Release assets; lệnh trên là bước tải weights tường minh.

## 8. Đóng scope

Tuning COMPLETE, E1 COMPLETE 16/16, E2 COMPLETE 12/12; E3/E4/E5 DEFERRED, TIGER/CRAMER THEORY_ONLY; E7 recorded analysis COMPLETE nhưng dedicated GPU profiling DEFERRED; E8 COMPLETE. Lý do thu gọn là tài nguyên và yêu cầu môn học, không test ranking. Giữ full grid 127 jobs để tham khảo; không có requested remaining training jobs. Driver course cũ chặn --run ngay cả trên Kaggle. Unit/integration tests dùng fixtures tổng hợp tạm thời để kiểm tra correctness; không phải lượt training thực nghiệm mới và không đi vào result tables.
