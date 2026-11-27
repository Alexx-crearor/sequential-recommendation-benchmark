"""Create clean local-analysis notebooks, including a public-summary-only mode."""
import json
from pathlib import Path
import _bootstrap


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True), "execution_count": None, "outputs": []}


SETUP = '''import os, sys, json
from pathlib import Path
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd().resolve()
if not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
assert (ROOT / "src/analysis/master_session1.py").is_file()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.analysis.master_session1 import ARCHIVE_NAME, ARCHIVE_SHA256, BUNDLE_HASH, analyze
TABLES = ROOT / "results/tables/master_session1"
FULL_AUDIT = False  # True needs the private original ZIP and matching dataset; never trains.
REBUILD_FIGURES = False
if FULL_AUDIT:
    archive = Path(os.environ.get("SEQREC_MASTER_ARCHIVE", str(ROOT / "artifacts/master_session1" / ARCHIVE_NAME)))
    bundle = Path(os.environ.get("SEQREC_BUNDLE", str(ROOT / "data/processed/beauty" / BUNDLE_HASH)))
    analyze(archive, ROOT, bundle, resamples=10000)
audit = json.loads((TABLES / "artifact_audit.json").read_text())
assert audit["state"] == "VERIFIED" and audit["archive_sha256"] == ARCHIVE_SHA256
assert audit["E1_success"] == 16 and audit["E2_success"] == 6 and audit["E7_gpu_state"] == "DEFERRED"
print("Source SHA256:", ARCHIVE_SHA256)
print("Mode:", "raw ZIP re-audited" if FULL_AUDIT else "versioned audited aggregate tables; not a fresh raw re-audit")
'''


def figures(names):
    return code('''from gen_fig_master_session1 import generate
if REBUILD_FIGURES:
    generate(ROOT)
''' + "\n".join(f'display(Image(filename=str(ROOT / "results/figures/master_session1/{name}.png")))' for name in names)+"\n")


def create(root):
    definitions = {
        "E1_Official_Analysis": [md("""# E1 — Phân tích test chính thức ba seeds

Chạy tại máy cá nhân. Nguồn là phiên master session 1; không dùng pilot hoặc validation tuning làm test results. Mặc định dùng bảng đã kiểm chứng trong GitHub; bật FULL_AUDIT để tái kiểm tra ZIP riêng tư/dataset và bootstrap từ raw predictions. Không training/tuning/inference GPU.
"""), code(SETUP), md("""## Mean, sample SD và đầy đủ Recall/NDCG/MRR

Neural models có ba seeds; MostPopular chỉ một run xác định, SD không xác định. Bảng bao gồm full/warm và K=5/10/20 (MRR@10).
"""), code('''summary = pd.read_csv(TABLES / "metric_summary.csv")
display(summary[summary.experiment == "E1"])
per_seed = pd.read_csv(TABLES / "per_seed_metrics.csv")
display(per_seed[per_seed.experiment == "E1"])
'''), md("""## Paired bootstrap

Lấy trung bình contributions qua seeds trong mỗi user trước khi lấy mẫu lại 22.363 users. 10.000 resamples, percentile 95%, seed 42; baseline dùng cố định. Multinomial trên nhóm vector trùng nhau giữ đúng phân phối empirical bootstrap. CI điều kiện trên checkpoint đã học, không bao trùm đầy đủ retraining uncertainty, chưa điều chỉnh đa so sánh.
"""), code('''paired = pd.read_csv(TABLES / "paired_bootstrap.csv")
display(paired[paired.experiment == "E1"])
'''), figures(["e1_test_metrics", "e1_recall_ndcg_at_k", "e1_paired_bootstrap", "e1_training_validation"])],
        "E2_Partial_Analysis": [md("""# E2 — Kết quả từng phần, chưa hoàn thành thiết kế

Chỉ 6/12 run SUCCESS. A1/A2 có seeds 7/42; A3/A4 chỉ seed 7. Không điền điểm giả hoặc coi missing runs đã hoàn thành.
"""), code(SETUP), code('''progress = pd.read_csv(TABLES / "original_progress.csv")
display(progress[progress.experiment == "E2"])
metrics = pd.read_csv(TABLES / "per_seed_metrics.csv")
display(metrics[metrics.experiment == "E2"])
summary = pd.read_csv(TABLES / "metric_summary.csv")
display(summary[summary.experiment == "E2"])
'''), md("""## Chỉ so sánh matched completed seeds

A2−A1 dùng seeds 7/42; A4−A3 chỉ seed 7. CI không thay thế sáu run thiếu. Giữ nguyên matched-capacity backbone, hyperparameters và quy tắc checkpoint validation.
"""), code('''paired = pd.read_csv(TABLES / "paired_bootstrap.csv")
display(paired[paired.experiment == "E2"])
display(pd.read_csv(TABLES / "course_manifest.csv").query("experiment == 'E2'"))
'''), figures(["e2_partial_results"])],
        "E7_Recorded_Efficiency": [md("""# E7 — Phân tích số đo đã lưu; dedicated GPU measurements DEFERRED

Không đo GPU hoặc chạy model inference tại máy cá nhân. Chỉ đọc Kaggle efficiency/history logs. Không nâng trạng thái E7 thành COMPLETE vì có latency nhúng trong trainer.
"""), code(SETUP), code('''runs = pd.read_csv(TABLES / "runs.csv")
display(runs[["experiment", "model", "variant", "seed", "training_seconds", "clustering_seconds", "validation_seconds", "checkpoint_seconds", "test_evaluation_seconds", "inference_mean_ms", "inference_p95_ms", "inference_scope", "peak_gpu_memory_mib", "measurement_device", "gpu_name"]])
'''), md("""Clustering nằm trong training time. Latency batch 1 gồm encoding/scoring toàn catalogue, không sorting/disk I/O; 5 warm-up và 20 repeats. Baseline CPU fit không trực tiếp tương đương neural GPU training. Memory là peak allocated, không phải tổng VRAM. Component sums không phải session elapsed time.
"""), code('''budget = pd.read_csv(TABLES / "remaining_budget_scenarios.csv")
display(budget)
print("Planning scenario hours (not measured remaining runs):", budget.max_epoch_mean_rate_scenario_seconds.sum()/3600)
assert not budget.remaining_run_actually_executed.any()
'''), figures(["recorded_efficiency"])],
        "E8_Error_Analysis": [md("""# E8 — Error analysis tại máy cá nhân

Chỉ từ predictions E1 đã xác minh; không inference. Các nhóm pairwise có thể giao nhau; all-six-miss dùng đủ sáu predictors. Không cherry-pick test-score extremes để thay đổi scope hoặc hyperparameters.
"""), code(SETUP), code('''groups = pd.read_csv(TABLES / "e8_error_groups.csv")
display(groups)
display(groups.groupby("category").agg(mean_fraction=("fraction", "mean"), sd_fraction=("fraction", "std")))
display(pd.read_csv(TABLES / "e8_pair_outcomes.csv"))
'''), md("""## History length và target popularity

Bins được định nghĩa cố định, không fit dựa trên test performance. Target popularity chỉ đếm training interactions; giữ cold targets. Per-user examples được lưu riêng tư tại results/derived, không commit raw histories/predictions. Examples lấy ba user ID nhỏ nhất mỗi nhóm/seed theo quy tắc cố định, không chọn chênh lệch lớn nhất.
"""), code('''segments = pd.read_csv(TABLES / "e8_segments.csv")
display(segments)
private_cases = ROOT / "results/derived/master_session1" / ARCHIVE_SHA256 / "e8_private_examples.json"
print("Private cases available locally:", private_cases.is_file())
# Do not display or commit identifying raw histories in this public notebook.
'''), figures(["e8_error_groups", "e8_history_popularity"])]}
    output = []
    for name, cells in definitions.items():
        cells.append(md("""## Báo cáo và trạng thái

Đọc [báo cáo tiếng Việt](../docs/MASTER_SESSION1_ANALYSIS_VI.md). Phạm vi thu gọn do tài nguyên GPU và yêu cầu môn học, không theo mô hình có test score cao. E1 complete, E2 partial, E3 pending/reuse available, E4/E5 deferred; TIGER/CRAMER theory only. Không tuyên bố các phase chưa chạy đã hoàn tất.
"""))
        for index, value in enumerate(cells):
            value["id"] = f"master-{index:02d}"
        notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        path = Path(root) / "notebooks" / f"{name}.ipynb"
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
        output.append(path)
    return output


if __name__ == "__main__":
    for path in create(_bootstrap.ROOT):
        print(path)
