"""Generate Vietnamese interpretation from audited Phase 5C aggregates."""
import argparse
from pathlib import Path
import pandas as pd
import _bootstrap
from src.analysis.phase5c import LABELS, MODELS, read_json


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers),
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def write_report(root):
    root = Path(root)
    tables = root / "results/tables/phase5c"
    report = read_json(tables / "artifact_audit.json")
    frame = pd.read_csv(tables / "tuning_trials.csv")
    selected = frame[frame.selected].set_index("model").loc[list(MODELS)]
    choices, losses = [], []
    for model, row in selected.iterrows():
        extra = f"mask_ratio={row.mask_ratio:g}" if model == "bert4rec" else (f"gamma={row.soft_label_gamma:g}; lambda_0={row.lambda_0:g}; beta_0={row.beta_0:g}" if model == "fenrec" else "—")
        choices.append([LABELS[model], f"T{int(row.trial)}", f"{row.learning_rate:g}", int(row.batch_size),
                        f"{row.dropout:g}", extra, f"{row['validation_ndcg@10']:.8f}", int(row.best_epoch),
                        f"{int(row.epochs_completed)}/{int(row.max_epochs)}", "Patience" if row.stopped_for_patience else "Trần epochs"])
        losses.append([LABELS[model], f"{row.initial_train_loss:.5f}", f"{row.final_train_loss:.5f}",
                       f"{row['final_validation_ndcg@10']:.8f}", f"{row.training_seconds/60:.2f}", f"{row.peak_gpu_memory_mib:.2f}"])
    capped = ", ".join(f"{LABELS[r.model]} T{r.trial}" for r in frame[frame.training_limit_reached].itertuples())
    fenrec = frame[frame.model == "fenrec"]
    fraction = fenrec.training_seconds.sum()/frame.training_seconds.sum()*100
    clustering_fraction = fenrec.clustering_seconds.sum()/fenrec.training_seconds.sum()*100
    text = f'''# Phase 5C — Báo cáo xác minh và phân tích tuning

Ngày phân tích: 26/11/2026. Trạng thái: **hoàn thành 15/15 trial của đợt tuning giới hạn đã xuất từ Kaggle**. Đây không phải kết quả final test hay toàn bộ không gian tìm kiếm hyperparameters. Không chạy lại tuning hoặc huấn luyện trên máy cá nhân trong quá trình phân tích.

## Nguồn duy nhất và phạm vi xác minh

Nguồn thực nghiệm duy nhất: `{report['source_archive']}`.

SHA256: `{report['archive_sha256']}`.

Archive có {report['archive_files']} tệp, tổng dung lượng giải nén {report['archive_uncompressed_bytes']:,} byte. Đã kiểm tra CRC toàn bộ ZIP, hash từng tệp, {report['hashed_run_artifacts_verified']} artifacts có hash trong status, 30 checkpoint best/latest, hash của 5 tuning plan và 15 configuration. Tất cả 15 run đều `VALIDATED`, seed 7, Beauty, không phải pilot, không giới hạn batch hoặc người dùng validation. Kiểm tra CPU chỉ đọc checkpoint, cấu trúc trọng số và trạng thái resume; không forward, backward hoặc optimizer step trên dữ liệu thực nghiệm.

Mỗi trial chứa đầy đủ predictions cho **22.363 người dùng validation**, trên catalogue **12.101 items**. Đã tính lại Recall/NDCG/MRR từ target ranks, kiểm tra top-K, loại lịch sử đã thấy, ngoại lệ target, coverage full/warm và sự đồng nhất user/history/target giữa cả 15 trial. Không chấm lại điểm mô hình hoặc chạy evaluation GPU tại máy cá nhân.

Split chung: `{report['split_version']}`. Báo cáo kiểm tra dữ liệu trong ZIP ghi `PASSED`, gồm chronology, event/target separation và training-only FENRec windows; các plan, config và checkpoint khớp split này. **ZIP không chứa dataset gốc hoặc manifest đầy đủ của bundle, nên phân tích này không thể tái băm độc lập các byte dataset hay tự chứng minh lại chronology từ dữ liệu thô.** Không dùng dataset hoặc kết quả cục bộ khác làm nguồn số liệu Phase 5C.

ZIP cũng không có source snapshot hoặc Git commit attestation của phiên training. Do đó xác minh chứng minh tính nhất quán của artifacts cung cấp, không chứng thực toàn bộ mã đã chạy ngoài những bằng chứng được lưu.

## Lựa chọn chỉ dùng validation

`evaluate_test=false` trong cả 5 plan, 15 config và efficiency record. Không có `test_metrics.json` hoặc raw test predictions. Cả 5 winner trùng argmax **validation NDCG@10** của 3 trial; khi hòa điểm, quy tắc chọn trial có chỉ số nhỏ hơn được giữ nguyên. Tính lại argmax từ `validation_metrics.csv`, đối chiếu predictions của best checkpoint, `trial_results.json` và `selected_config.yaml`. Không dùng test performance để chọn hoặc sắp xếp cấu hình.

Chỉ số trial T0–T2 giữ nguyên hệ đánh số 0-based của tuner; epoch trong báo cáo và biểu đồ đổi sang **1-based**, còn cột `epoch` gốc trong bảng lịch sử giữ nguyên 0-based. Các đường validation chỉ chứa lần đánh giá thực sự; các giá trị carry-forward ở epoch bỏ qua không được coi là phép đo mới.

{markdown_table(['Mô hình', 'Trial', 'LR', 'Batch', 'Dropout', 'Tham số riêng', 'Best val NDCG@10', 'Best epoch', 'Đã chạy/trần', 'Dừng'], choices)}

FENRec đạt validation NDCG@10 cao nhất trong các cấu hình được thử ({selected.loc['fenrec', 'validation_ndcg@10']:.8f}), tiếp theo SASRec-CE ({selected.loc['sasrec_ce', 'validation_ndcg@10']:.8f}). Chênh lệch mô tả là {selected.loc['fenrec', 'validation_ndcg@10']-selected.loc['sasrec_ce', 'validation_ndcg@10']:.8f} theo thang NDCG, **không phải bằng chứng về final-test superiority hay ý nghĩa thống kê**. Chỉ có một seed, ba trial/mô hình; các tham số thay đổi đồng thời nên không thể quy kết nguyên nhân cho riêng LR, batch hoặc dropout. Không tính error bar, khoảng tin cậy hoặc sample SD từ ba trial có hyperparameters khác nhau.

## Loss, stopping và giới hạn convergence

{markdown_table(['Mô hình được chọn', 'Loss đầu', 'Loss cuối', 'Val cuối', 'Training phút', 'Peak allocated MiB'], losses)}

Loss giảm từ đầu đến cuối ở cả năm trial được chọn. Tuy nhiên BCE, shifted full-softmax CE, masked-item CE và tổng objective FENRec có thang đo/đơn vị khác nhau; **không dùng loss tuyệt đối để xếp hạng giữa mô hình**. FENRec còn có lịch warm-up và trọng số contrastive thay đổi theo epoch, nên bước thay đổi loss không tự chứng minh mất ổn định hoặc overfitting.

Có **{report['patience_stopped_trials']} trial dừng theo patience** và **{report['limit_reached_trials']} trial chạm trần epochs**. Đã mô phỏng lại counter patience từ các validation checks, có minimum-epoch/warm-up gate, và đối chiếu với latest checkpoint. Patience là tiêu chí dừng vận hành, không chứng minh hội tụ toàn cục.

Các trial chạm trần: {capped}. Cả ba FENRec và cả ba GRU4Rec-CE đều hết ngân sách; không được ghi là đã hội tụ. FENRec giữ nguyên warm-up 20 epochs, chạy tổng 60 epochs. Các trial SASRec-CE dừng ở 34/42/52 epochs; BERT4Rec T1/T2 dừng ở 78/76, SASRec-BCE T1 dừng ở 56. Best checkpoint có thể sớm hơn epoch cuối; phase tiếp theo phải giữ lựa chọn checkpoint theo validation, không tự thay bằng latest weights.

## Chi phí quan sát trên Kaggle

Tổng thời gian epoch training được ghi: **{frame.training_seconds.sum()/3600:.3f} giờ**; validation checks: **{frame.validation_seconds.sum()/3600:.3f} giờ**; checkpoint writes: **{frame.checkpoint_seconds.sum():.2f} giây**. Tổng các thành phần này khoảng **{(frame.training_seconds.sum()+frame.validation_seconds.sum()+frame.checkpoint_seconds.sum())/3600:.3f} giờ**. Đây là tổng thời gian thành phần ghi nhận, không phải thời lượng session hoặc chi phí GPU billing; không bao gồm chuẩn bị dữ liệu, toàn bộ overhead hoặc final validation pass của selected checkpoint.

FENRec chiếm khoảng {fraction:.1f}% tổng training time của 15 trial. Clustering FENRec chiếm {clustering_fraction:.1f}% training time của chính FENRec và **đã nằm trong** training time, không cộng hai lần. Phép đo là peak **allocated** GPU memory, không phải tổng VRAM mà hệ điều hành/các tiến trình chiếm. Các run ghi Tesla T4, `cuda:0`; báo cáo môi trường phát hiện hai T4 nhưng artifacts không chứng minh distributed/multi-GPU training.

![Validation curves](../results/figures/phase5c/validation_curves.png)

![Training objectives](../results/figures/phase5c/training_loss.png)

![Trial comparison](../results/figures/phase5c/trial_validation_comparison.png)

![Resource measurements](../results/figures/phase5c/selected_trial_resources.png)

Mỗi hình có PNG 300 DPI và PDF vector cùng tên trong `results/figures/phase5c/`. Màu và kiểu nét phân biệt trial; không có dải sai số giả. Workflow dữ liệu của skill `academic-plotting` được áp dụng; script tạo hình được lưu trong `scripts/gen_fig_phase5c.py`.

## Tổ chức và tái lập tại máy cá nhân

```text
results/raw/phase5c/<archive-sha>/results/...   # nguyên byte, ignored
results/derived/phase5c/<archive-sha>/         # provenance cục bộ, ignored
results/tables/phase5c/                       # CSV/JSON tổng hợp để publication
results/figures/phase5c/                      # PNG 300 DPI + PDF vector
configs/selected/phase5c/beauty/               # 5 selection YAML nguyên byte + manifest
notebooks/Phase5C_Analysis.ipynb                # audit → tables → plots, không training
```

ZIP gốc được sao chép vào `artifacts/phase5c/` trong repository; bản bên ngoài vẫn giữ nguyên. CLI và notebook mặc định dùng đường dẫn tương đối này, không giả định thư mục Downloads của người dùng khác. Giải nén không ghi đè: tệp đã có phải có hash giống hệt, khác biệt sẽ báo lỗi trước khi ghi thêm bất kỳ tệp nào. Selected YAML sao chép nguyên byte từ archive; `run_path` vẫn là provenance đường dẫn gốc của Kaggle. Checkpoint được tổ chức riêng dưới `checkpoints/phase5c/beauty/<model>/trials/<trial>/`; đường dẫn portable của từng winner nằm trong `configs/artifacts/phase5c_checkpoints.json`.

Tái lập đầy đủ, chỉ tại máy cá nhân:

```powershell
python scripts/analyze_phase5c.py
```

Lệnh trên dùng `artifacts/phase5c/{report['source_archive']}`. Có thể override `--archive PATH` hoặc `SEQREC_PHASE5C_ARCHIVE`; notebook dùng cùng resolver. Chỉ dựng lại hình từ bảng đã xuất:

```bash
python scripts/gen_fig_phase5c.py
python scripts/write_phase5c_report.py
```

Git history chứa inventory/hash, bảng tổng hợp, cấu hình và mã; không chứa binary lớn. Theo yêu cầu cung cấp checkpoint, GitHub Release `phase5c-checkpoints-v1` phân phối riêng các checkpoint nguyên byte cùng cấu hình, không có raw predictions, per-user histories, dataset, logs hoặc credentials. Chạy `python scripts/fetch_phase5c_checkpoints.py` để lấy năm best checkpoint đã chọn; thêm `--all-trials` để lấy đủ 30 best/latest checkpoint. Downloader xác minh SHA256 của ZIP và từng tệp rồi đặt vào cây repository. Gói checkpoint không thay thế ZIP gốc riêng tư để audit toàn bộ predictions. `artifact_audit.json` mô tả phạm vi xác minh; `archive_inventory.csv` ghi tên, kích thước và SHA256 của cả 182 tệp.

## Công việc tiếp theo, chưa thực hiện

Đợt tuning Phase 5C này hoàn thành theo ngân sách 3 trial/mô hình đã xuất. E1 final với seeds 7/42/2026, E2–E5 ablations/robustness, E7/E8 từ final checkpoints, TIGER/E6 và báo cáo thống kê final test **chưa được thực hiện bởi đợt này**. Trial hết ngân sách là hạn chế cần trình bày; không tự tăng epoch hoặc mở lại tuning trong phân tích hiện tại.

Cấu hình có thể dùng qua `--selection configs/selected/phase5c/beauty/MODEL.yaml`; final runs phải tạo hash mới, bật giao thức test cuối cùng và giữ seed/selection provenance. Không resume trial validation-only thành run final khác cấu hình. Ví dụ lệnh chuẩn bị cho Kaggle (không chạy ở máy cá nhân):

```bash
python scripts/run_experiment.py --profile core_research --final-seeds --experiment E1 --dataset beauty --model sasrec_ce --seed 7 --bundle data/processed/beauty/HASH --device cuda:0 --selection configs/selected/phase5c/beauty/sasrec_ce.yaml --results-root results/core --resume
```

Kaggle chỉ dùng cho training/testing/evaluation/đo GPU; phân tích, trực quan hóa, tổng hợp và publication thực hiện tại máy cá nhân. Không suy diễn trạng thái Phase 5B từ ZIP này. Final test chưa có thì không công bố bảng xếp hạng final, p-value hoặc khẳng định cải thiện generalization.
'''
    path = root / "docs/PHASE5C_ANALYSIS_VI.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_bootstrap.ROOT)
    print(write_report(parser.parse_args().root))
