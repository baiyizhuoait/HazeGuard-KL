"""Compare archived and QC-refitted M5 results without fitting or prediction."""
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/qc_fix/pre_refit_stage5"
METRICS = ("Macro_F1", "Balanced_Accuracy", "High_Risk_Recall", "MAOE",
           "Severe_Jumping_Error_Rate", "Avg_Asymmetric_Cost")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    output = ROOT / "models/metrics_stage5_qc_comparison.json"
    table = ROOT / "docs/qc_fix/stage5_refit_comparison.md"
    rows_path = ROOT / "docs/qc_fix/stage5_benchmark_rows.csv"
    if any(p.exists() for p in (output, table, rows_path)):
        raise FileExistsError("QC comparison already exists; inspect it instead of replacing it")
    old, new = (read(p / "metrics_stage5.json") for p in (ARCHIVE, ROOT / "models"))
    records = []
    for model in ("M5_fixed", "M5_val_tuned"):
        for split, key in (("val_seed42", "validation_metrics_seed42"), ("test_seed42", "test_metrics_seed42")):
            for metric in METRICS:
                a, b = old["models"][model][key][metric], new["models"][model][key][metric]
                records.append({"model": model, "split": split, "metric": metric, "old": a, "new": b, "delta": b-a})
        for metric in METRICS:
            a = old["validation_mean_std"][model][metric]["mean"]
            b = new["validation_mean_std"][model][metric]["mean"]
            records.append({"model": model, "split": "val_three_seed_mean", "metric": metric,
                            "old": a, "new": b, "delta": b-a})
    intervals = []
    for filename, method in (("metrics_stage5_interval_adaptive.json", "adaptive"),
                              ("metrics_stage5_interval_calibrated.json", "fixed")):
        a, b = read(ARCHIVE / filename), read(ROOT / "models" / filename)
        for split in ("val", "test"):
            for group in ("overall", "KL_City", "Port_Klang", "Sepang"):
                if method == "adaptive":
                    sa = a[split]["overall"] if group == "overall" else a[split]["by_node"][group]
                    sb = b[split]["overall"] if group == "overall" else b[split]["by_node"][group]
                else:
                    sa = a[split]["calibrated"] if group == "overall" else a[split]["by_node"][group]["calibrated"]
                    sb = b[split]["calibrated"] if group == "overall" else b[split]["by_node"][group]["calibrated"]
                intervals.append({"method": method, "split": split, "group": group,
                                  "old": sa, "new": sb, "delta": {k: sb[k]-sa[k] for k in ("PICP", "MPIW", "n_samples")}})
    adaptive = read(ROOT / "models/metrics_stage5_interval_adaptive.json")
    fixed = read(ROOT / "models/metrics_stage5_interval_calibrated.json")
    gain = (adaptive["test"]["overall"]["PICP"] - fixed["test"]["calibrated"]["PICP"]) * 100
    width_change = (adaptive["test"]["overall"]["MPIW"] / fixed["test"]["calibrated"]["MPIW"] - 1) * 100
    result = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "input_sha256": new["input_sha256"], "old_input_sha256": old["input_sha256"],
              "same_final_model_parameters": True, "same_level_thresholds": True,
              "new_model_or_threshold_search": False, "model_comparison": records,
              "interval_comparison": intervals, "adaptive_minus_fixed_test_coverage_pp": gain,
              "adaptive_vs_fixed_test_width_change_percent": width_change,
              "M6_refit": False, "new_model_fits": 0, "new_model_test_predictions": 0,
              "model_sha256": {name: hashlib.sha256((ROOT / "models" / name).read_bytes()).hexdigest()
                               for name in ("rf_regressor.joblib", "ordinal_rf.joblib")}}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    lines = ["# M5 QC 重评：同配方新旧对照", "",
             "原模型参数、分档切点与三个种子不变；新表含 77,744 / 16,647 / 16,647 行。",
             "M6 未重训，原模型与原预测保留为历史附录；本表不把它当作 QC 重训结果。", "",
             "## M5 分类指标", "", "| 模型 | 切分 | 指标 | 旧 | QC | 变化 |", "|---|---|---|---:|---:|---:|"]
    for row in records:
        lines.append(f"| {row['model']} | {row['split']} | {row['metric']} | {row['old']:.4f} | {row['new']:.4f} | {row['delta']:+.4f} |")
    lines += ["", "## 总体区间指标", "", "| 方法 | 切分 | 旧 PICP | QC PICP | 旧 MPIW | QC MPIW |", "|---|---|---:|---:|---:|---:|"]
    for row in intervals:
        if row["group"] == "overall":
            lines.append(f"| {row['method']} | {row['split']} | {row['old']['PICP']:.8f} | {row['new']['PICP']:.8f} | {row['old']['MPIW']:.8f} | {row['new']['MPIW']:.8f} |")
    lines += ["", "## 绝对变化超过 0.005 的逐项说明", "",
              "下列说明界定原因范围，不把联合数据变更的增益单独归因于某一列。", ""]
    for row in records:
        if abs(row["delta"]) > .005:
            lines.append(f"- {row['model']} / {row['split']} / {row['metric']}：{row['delta']:+.4f}。新表更换一个选择特征并冲刷边界行，随后按原超参重新拟合；未新增调参或按测试结果选择方案。")
    for row in intervals:
        for metric in ("PICP", "MPIW"):
            delta = row["delta"][metric]
            if abs(delta) > .005:
                lines.append(f"- {row['method']} / {row['split']} / {row['group']} / {metric}：{delta:+.6f}。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。")
    lines += ["", "## 指纹", ""]
    lines += [f"- `{name}`：`{digest}`。" for name, digest in result["model_sha256"].items()]
    lines += ["", "## 给 6/7 号", "",
              "`stage5_benchmark_rows.csv` 给出两条 M5 测试记录和特征表指纹，供 6 号更新 benchmark；它不是共享 CSV 的替代品。",
              "M6/M7 仍使用历史概率：新 NPZ 只是按精确键对齐，不能宣称它们已经在 QC 特征表上重训。",
              "旧 ordinal_rf.joblib 的 selected_features_ 仍是旧顺序，不能直接传入新版 Top-25 矩阵。",
              "看板区间继续读同名自适应 JSON/NPZ，字段与键名不变。", ""]
    table.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    with rows_path.open("w", encoding="utf-8", newline="") as stream:
        fields = ["model", "feature_sha256", "split", "seed", *METRICS, "n_samples"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for model in ("M5_fixed", "M5_val_tuned"):
            scores = new["models"][model]["test_metrics_seed42"]
            writer.writerow({"model": model, "feature_sha256": new["input_sha256"]["features_final"],
                             "split": "test", "seed": 42, **{k: scores[k] for k in (*METRICS, "n_samples")}})
    print("Saved QC comparison, per-change explanations, model fingerprints and M5 benchmark handover rows.")


if __name__ == "__main__":
    main()
