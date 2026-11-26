"""Build the UTF-8 local Phase 5C analysis notebook without embedded results."""
import json
from pathlib import Path
import _bootstrap


def cell(kind, text):
    value = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def create(root):
    cells = [cell("markdown", """# Phase 5C — Phân tích tuning có thể tái lập

Chạy tại máy cá nhân. ZIP gốc là nguồn thực nghiệm duy nhất; notebook chỉ kiểm tra và phân tích CPU, không chạy tuning, training hoặc evaluation mô hình. Các điểm số dưới đây là **validation**, không phải final test.

Cài trước: `python -m pip install -r requirements-analysis.txt`. ZIP riêng tư không có trên GitHub. Đặt `SEQREC_PHASE5C_ARCHIVE` nếu ZIP không nằm trong Downloads. Notebook đã thực thi được lưu riêng dưới `results/derived/`, không commit per-user artifacts.
"""), cell("code", '''import os, sys
from pathlib import Path
ROOT = Path.cwd().resolve()
if not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
assert (ROOT / "src/analysis/phase5c.py").is_file(), "Open from repository root or notebooks/"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.analysis.phase5c import analyze_archive, ARCHIVE_NAME, ARCHIVE_SHA256
ARCHIVE = Path(os.environ.get("SEQREC_PHASE5C_ARCHIVE", str(Path.home() / "Downloads" / ARCHIVE_NAME)))
assert ARCHIVE.is_file(), f"Original private ZIP required: {ARCHIVE}"
print("Expected source SHA256:", ARCHIVE_SHA256)
'''), cell("markdown", """## 1. Xác minh tính toàn vẹn và provenance

Kiểm tra CRC, toàn bộ member hashes, 150 status-hashed artifacts, 30 best/latest checkpoints, plan/config fingerprints, validation population, contributions, top-K, selection và stopping counters. Raw extraction không ghi đè tệp khác hash.

Dataset bytes không nằm trong ZIP: chỉ xác minh tính nhất quán của split IDs và báo cáo data audit được lưu. ZIP không có Git/source attestation của phiên training.
"""), cell("code", '''trials, audit = analyze_archive(ARCHIVE, ROOT)
assert audit["state"] == "VERIFIED" and audit["trials"] == 15
assert audit["test_result_artifacts"] == 0
from IPython.display import display, Image, Markdown
print({k: audit[k] for k in ("archive_files", "trials", "checkpoint_files_verified", "limit_reached_trials", "patience_stopped_trials")})
print(audit["dataset_verification_scope"])
'''), cell("markdown", """## 2. Toàn bộ trials và cấu hình được chọn

Trial giữ nguyên T0–T2; best epoch là 1-based. Những trial có hyperparameters khác nhau không phải các seed lặp để tính standard deviation hoặc confidence interval.
"""), cell("code", '''columns = ["model", "trial", "selected", "learning_rate", "batch_size", "dropout", "mask_ratio", "soft_label_gamma", "lambda_0", "beta_0", "validation_ndcg@10", "best_epoch", "epochs_completed", "max_epochs", "stopping_reason", "training_seconds", "peak_gpu_memory_mib"]
display(trials[columns])
display(trials.loc[trials.selected, columns])
'''), cell("markdown", """## 3. Convergence, stopping và tài nguyên

Chỉ vẽ validation checks thực sự; giá trị carry-forward không phải quan sát mới. Loss khác objective không so sánh trực tiếp giữa mô hình. Thời gian clustering đã nằm trong training time. Component logs không phải session wall time. Patience không phải chứng minh hội tụ toàn cục.
"""), cell("code", '''from gen_fig_phase5c import generate_figures
figures = generate_figures(ROOT)
for path in figures:
    if path.suffix == ".png":
        display(Image(filename=str(path)))
print("Exported", len(figures), "files: PNG 300 DPI and vector PDF")
'''), cell("markdown", """## 4. Báo cáo tiếng Việt và bước tiếp theo

Không thay đổi giao thức hoặc mở lại tuning cho trial chạm giới hạn. Final seeds, E1–E5, E7/E8 từ final checkpoints và TIGER/E6 vẫn là công việc tiếp theo. Training/testing/evaluation/đo GPU chạy trên Kaggle; phân tích/publication ở máy cá nhân.
"""), cell("code", '''from write_phase5c_report import write_report
report_path = write_report(ROOT)
display(Markdown(report_path.read_text(encoding="utf-8")))
''')]
    for index, value in enumerate(cells):
        value["id"] = f"phase5c-{index:02d}"
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.10"}}, "nbformat": 4, "nbformat_minor": 5}
    output = Path(root) / "notebooks/Phase5C_Analysis.ipynb"
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(create(_bootstrap.ROOT))
