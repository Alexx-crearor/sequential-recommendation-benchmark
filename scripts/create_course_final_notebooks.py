"""Create clean final-course analysis notebooks; public mode needs no private ZIP."""
import json
from pathlib import Path
import _bootstrap
from create_master_analysis_notebooks import md, code

SETUP = '''import os, sys, json
from pathlib import Path
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd().resolve()
if not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
assert (ROOT / "src/analysis/course_final.py").is_file()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.analysis.course_final import analyze, ARCHIVE_NAME, ARCHIVE_SHA256
from src.analysis.master_session1 import BUNDLE_HASH
TABLES = ROOT / "results/tables/course_final"
FULL_AUDIT = False  # True requires three private source ZIPs and the matching dataset.
REBUILD_PUBLICATION = False
if FULL_AUDIT:
    archive = Path(os.environ.get("SEQREC_FINAL_ARCHIVE", str(ROOT / "artifacts/e2_final" / ARCHIVE_NAME)))
    bundle = Path(os.environ.get("SEQREC_BUNDLE", str(ROOT / "data/processed/beauty" / BUNDLE_HASH)))
    analyze(archive, ROOT, bundle, 10000)
if REBUILD_PUBLICATION:
    from gen_fig_course_final import generate
    from write_course_final_report import write_report
    generate(ROOT)
    write_report(ROOT)
audit = json.loads((TABLES / "artifact_audit.json").read_text())
assert audit["state"] == "VERIFIED" and audit["archive_sha256"] == ARCHIVE_SHA256
assert audit["runs_verified"] == 28 and audit["prior_changed_files"] == 0
assert audit["training_closed"] and audit["remaining_requested_training_runs"] == 0
print("Raw re-audit" if FULL_AUDIT else "Public audited-table mode, not a fresh raw audit")
print("Source:", ARCHIVE_SHA256)
'''


def figures(*names):
    return code("\n".join(f'display(Image(filename=str(ROOT / "results/figures/course_final/{name}.png")))' for name in names)+"\n")


def create(root):
    notebooks = {
        "Course_Final_Analysis": [md("""# Bộ kết quả cuối cùng — Data Mining

Phạm vi đã đóng: tuning COMPLETE, E1 16/16, E2 12/12; E3–E5 DEFERRED; TIGER/CRAMER THEORY_ONLY; E7 recorded analysis COMPLETE, dedicated profiling DEFERRED; E8 COMPLETE. Không training/tuning/inference/profiling trong notebook. Public mode đọc tables đã versioned, chạy được từ clean clone.
"""), code(SETUP), code('''display(pd.read_csv(TABLES / "course_status.csv"))
display(pd.read_csv(TABLES / "scientific_findings.csv"))
'''), md("""## Validation tuning và official test phải tách biệt

Ba trial/mô hình seed 7 là bounded search, không phải exhaustive optimum. Tuning selection chỉ dựa validation; E1/E2 là held-out test, không dùng chọn lại hyperparameters hoặc scope. MostPopular n=1 và SD undefined.
"""), code('''tuning = pd.read_csv(TABLES / "tuning_selected.csv")
assert (tuning.evaluation_split == "validation").all()
display(tuning)
summary = pd.read_csv(TABLES / "metric_summary.csv")
assert (summary.evaluation_split == "test").all()
display(summary)
'''), figures("e1_test_metrics", "e2_test_metrics"), md("""## E7 và E8 từ artifacts

Không đo mới GPU/latency. Timing scope và giới hạn phải được giữ khi viết báo cáo. E8 chỉ đọc saved E1 predictions đã được chứng minh không thay đổi.
"""), code('''display(pd.read_csv(TABLES / "e2_efficiency_summary.csv"))
display(pd.read_csv(TABLES / "e8_error_groups.csv"))
display(pd.read_csv(TABLES / "asset_catalog.csv"))
'''), figures("recorded_efficiency", "e8_error_groups")],
        "E2_Final_Analysis": [md("""# E2 cuối cùng — đủ bốn variants × ba seeds

A1 causal next-BCE; A2 causal next-CE; A3 causal masked-CE; A4 bidirectional masked-CE. Thay thế bản partial để viết kết luận cuối; bản partial và raw source cũ vẫn giữ nguyên. Không kết luận causal/global vượt quá matched procedure.
"""), code(SETUP), code('''summary = pd.read_csv(TABLES / "metric_summary.csv")
e2 = summary[summary.experiment == "E2"]
assert (e2.n_runs == 3).all() and e2.design_complete.all()
display(e2)
display(pd.read_csv(TABLES / "per_seed_metrics.csv").query("experiment == 'E2'"))
'''), md("""## Ba contrasts và uncertainty

10.000 paired-user resamples, sau khi lấy mean contribution trong user qua ba fitted seeds. Percentile 95%, pointwise/unadjusted; không full retraining uncertainty hoặc confirmatory p-value. BCE-vs-CE và next-vs-mask là objective packages có nhiều thay đổi về negative coverage/target density, không cô lập một scalar hoặc riêng masking.
"""), code('''paired = pd.read_csv(TABLES / "paired_bootstrap.csv")
display(paired[paired.experiment == "E2"])
display(pd.read_csv(TABLES / "scientific_findings.csv").query("experiment == 'E2'"))
'''), figures("e2_complete_results", "e2_test_metrics", "e2_paired_contrasts"), md("""## Dynamics và recorded efficiency

Loss khác thang đo giữa objectives; dùng within-variant curves. Pipeline SUCCESS không phải chứng minh global convergence; ghi rõ cap/patience. Latency là encoded-history/full-catalog scoring không sorting/disk I/O, không đại diện dedicated GPU profiling.
"""), code('''display(pd.read_csv(TABLES / "e2_training_dynamics.csv"))
display(pd.read_csv(TABLES / "runs.csv").query("experiment == 'E2'"))
display(pd.read_csv(TABLES / "e2_efficiency_summary.csv"))
'''), figures("e2_training_validation", "e2_recorded_efficiency")],
        "Course_Reproducibility_Audit": [md("""# Audit và reproducibility của phạm vi đã đóng

Không phải notebook training. Public-summary mode trình bày bằng chứng versioned; FULL_AUDIT cần originals riêng tư và dataset đúng hash, chạy audit/statistics CPU. Không tự tải raw predictions hoặc tạo missing experiments.
"""), code(SETUP), code('''display(pd.Series(audit))
comparison = pd.read_csv(TABLES / "prior_run_comparison.csv")
assert len(comparison) == 262 and comparison.unchanged.all()
assert (comparison.old_sha256 == comparison.new_sha256).all()
display(comparison)
inventory = pd.read_csv(TABLES / "archive_inventory.csv")
print("Archive members:", len(inventory))
print("Prior logical bytes reused:", inventory.loc[inventory.deduplicated_prior_content, "bytes"].sum())
'''), md("""## Checkpoint index ghép, không nhân bản lại old weights

Release phiên 1 có 22 run; delta mới chỉ sáu run. CLI `python scripts/fetch_course_final_checkpoints.py` tải 28 best; thêm `--with-resume` lấy 27 latest states. Hash ZIP và từng tệp được kiểm tra. MostPopular là NPZ, không dùng PyTorch loader cho nó. Gói weights không thay thế ZIP nguyên gốc để re-audit per-user metrics.
"""), code('''manifest = json.loads((ROOT / "configs/artifacts/course_final_checkpoints.json").read_text())
assert len(manifest["models"]) == manifest["best_checkpoints"] == 28
assert manifest["latest_states"] == 27
display(pd.DataFrame(manifest["models"]).T)
scope = json.loads((ROOT / "configs/experiments/course_jobs.json").read_text())
assert scope["training_closed"] and len(scope["jobs"]) == 28
print("No requested remaining training runs. Full 127-job grid is reference only.")
'''), md("""## Báo cáo và hình/bảng

Đọc [COURSE_FINAL_REPORT_VI.md](../docs/COURSE_FINAL_REPORT_VI.md), [phương pháp/limitations](../docs/EXPERIMENT_METHODS_REPRODUCIBILITY_VI.md) và [asset catalog](../docs/REPORT_ASSET_CATALOG.md). PNG 300 DPI cho slides, PDF vector cho report. Không đổi SD N/A thành 0, không nhầm SD với CI và không gán dedicated E7 COMPLETE.
""")]
    }
    outputs = []
    for name, cells in notebooks.items():
        for index, value in enumerate(cells):
            value["id"] = f"final-{index:02d}"
        notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        path = Path(root) / "notebooks" / f"{name}.ipynb"
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
        outputs.append(path)
    return outputs


if __name__ == "__main__":
    for path in create(_bootstrap.ROOT):
        print(path)
