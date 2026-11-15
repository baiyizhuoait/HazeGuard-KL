"""
1 号 · 修复验收证据（对账 + 逐行预测比对）

回答两个问题：
  1. 净减少的 271 行，能不能拆成互斥的原因，且与「旧剔除 315 / 新剔除 586」对得上；
  2. 修复后的 M1 是重新 fit 的，还是在加载旧模型——用同一批测试键逐行比预测。

用法：
    python tools/verify_qc_fix.py \
        --old data/haze_panel_clean.parquet \
        --new .scratch/qc_fix/fixed/data/haze_panel_clean.parquet \
        --drops .scratch/qc_fix/fixed/models/drops.parquet \
        --out-dir docs/qc_fix
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from importlib import util

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import config as C  # noqa: E402
import utils_metrics as UM  # noqa: E402

KEY = ["station_id", "timestamp"]


def _load_pipeline_module():
    spec = util.spec_from_file_location("pl", os.path.join(ROOT, "src", "01_data_pipeline.py"))
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_baseline_module():
    spec = util.spec_from_file_location("bl", os.path.join(ROOT, "src", "01_baseline_dt.py"))
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _full_grid(df: pd.DataFrame) -> pd.DataFrame:
    """每个节点的完整小时网格（首端空块已经在面板里被裁掉，这里以面板的起止为界）。"""
    parts = []
    for sid, sub in df.groupby("station_id"):
        # 面板尾部 3 小时没有 t+3h 目标，已经从面板里掉出去了，网格要补回来
        rng = pd.date_range(sub["timestamp"].min(),
                            sub["timestamp"].max() + pd.Timedelta(hours=C.HORIZON_H),
                            freq="h")
        parts.append(pd.DataFrame({"station_id": sid, "timestamp": rng}))
    return pd.concat(parts, ignore_index=True)


def _unrounded(y: np.ndarray, p: np.ndarray) -> dict:
    cm = np.asarray(UM.DEFAULT_COST_MATRIX, dtype=float)
    hr = (y >= 2)
    return {
        "Macro_F1": float(f1_score(y, p, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y, p)),
        "High_Risk_Recall": float((p[hr] >= 2).mean()),
        "MAOE": float(np.abs(p - y).mean()),
        "Severe_Jumping_Error_Rate": float((np.abs(p - y) >= 2).mean()),
        "Avg_Asymmetric_Cost": float(cm[y, p].mean()),
        "n": int(len(y)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=os.path.join(C.DATA_DIR, "haze_panel_clean.parquet"))
    ap.add_argument("--new", required=True)
    ap.add_argument("--drops", required=True)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "docs", "qc_fix"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    old = pd.read_parquet(args.old)
    new = pd.read_parquet(args.new)
    for df in (old, new):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    drops = pd.read_parquet(args.drops)
    drops["timestamp"] = pd.to_datetime(drops["timestamp"])

    # ---- 1. 行数对账 -----------------------------------------------------
    full = _full_grid(old)
    old_keys = old[KEY].copy()
    new_keys = new[KEY].copy()
    def _left_only(left: pd.DataFrame, right: pd.DataFrame, tag: str) -> pd.DataFrame:
        m = left.merge(right, on=KEY, how="left", indicator=tag)
        return m[m[tag] == "left_only"].drop(columns=tag)

    old_dropped = _left_only(full, old_keys, "_in_old")
    new_dropped = _left_only(full, new_keys, "_in_new")
    gone = _left_only(old_keys, new_keys, "_in_new")
    # 净减少 = 旧面板保留、新面板不再保留的行
    only_new = gone.copy()

    only_new = only_new.merge(drops[KEY + ["reason", "split"]], on=KEY, how="left")
    assert only_new["reason"].notna().all(), "有行没有原因"
    reasons = only_new["reason"].value_counts().to_dict()

    rec = {
        "fetched_station_hours": 969072,
        "leading_block_trimmed": 2080,
        "trimmed_grid_rows": int(len(full)),
        "old": {
            "rows_final": int(len(old)),
            "dropped_total": int(len(old_dropped)),
            "dropped_no_target": 226,
            "dropped_no_anchor": 89,
        },
        "new": {
            "rows_final": int(len(new)),
            "dropped_total": int(len(new_dropped)),
            "dropped_split_purge": int((drops["reason"] == "split_purge").sum()),
            "dropped_no_target": int((drops["reason"] == "no_target").sum()),
            "dropped_no_anchor": int((drops["reason"] == "no_anchor").sum()),
        },
        "net_row_change": int(len(new) - len(old)),
        "rows_kept_before_dropped_now": int(len(gone)),
        "partition_of_net_decrease": reasons,
        "partition_sums": int(sum(reasons.values())),
        "identity_check": {
            "purge": int(reasons.get("split_purge", 0)),
            "extra_no_target": int((drops["reason"] == "no_target").sum()) - 226,
            "extra_no_anchor": int((drops["reason"] == "no_anchor").sum()) - 89,
            "sum": (int((drops["reason"] == "split_purge").sum())
                    + int((drops["reason"] == "no_target").sum()) - 226
                    + int((drops["reason"] == "no_anchor").sum()) - 89),
        },
        "dropped_by_reason_and_split": only_new.groupby(["reason", "split"]).size()
        .rename("rows").reset_index().to_dict(orient="records"),
    }
    assert rec["partition_sums"] == abs(rec["net_row_change"]) == len(gone), "271 行对不上"
    assert rec["identity_check"]["sum"] == abs(rec["net_row_change"]), "156/328/102 与 271 对不上"

    # 271 行明细
    only_new[KEY + ["reason", "split"]].sort_values(KEY).to_csv(
        os.path.join(args.out_dir, "rows_dropped_271.csv"), index=False)

    # ---- 2. M1 重训与逐行预测 -------------------------------------------
    bl = _load_baseline_module()
    feat = pd.read_parquet(os.path.join(C.DATA_DIR, "feat_spatiotemporal.parquet"),
                           columns=KEY + ["audit_model_eligible"])
    feat["timestamp"] = pd.to_datetime(feat["timestamp"])

    def _fit(panel: pd.DataFrame):
        X, y, d = bl.build_xy(panel)
        merged = d.merge(feat, on=KEY, how="left")
        keep = merged["audit_model_eligible"].fillna(False).to_numpy()
        d2, X2, y2 = d[keep].reset_index(drop=True), X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
        tr = (d2["split"] == "train").to_numpy()
        model = DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=20,
                                       random_state=C.RANDOM_STATE).fit(X2[tr].astype(float), y2[tr])
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(X2[tr].to_numpy(dtype="float64")).tobytes())
        h.update(np.ascontiguousarray(y2[tr].to_numpy(dtype="int64")).tobytes())
        return model, d2, X2, y2, tr, h.hexdigest(), int(tr.sum())

    m_old, d_old, X_old, y_old, tr_old, fp_old, n_old = _fit(old)
    m_new, d_new, X_new, y_new, tr_new, fp_new, n_new = _fit(new)

    te_old = (d_old["split"] == "test").to_numpy()
    te_new = (d_new["split"] == "test").to_numpy()
    keys_old = d_old.loc[te_old, KEY].reset_index(drop=True)
    keys_new = d_new.loc[te_new, KEY].reset_index(drop=True)
    assert keys_old.equals(keys_new), "两次测试键顺序不一致"
    assert len(keys_old) == 16647, f"测试键应为 16,647，实际 {len(keys_old)}"

    p_old = m_old.predict(X_old[te_old].astype(float))
    p_new = m_new.predict(X_new[te_new].astype(float))
    y_te = y_old[te_old].to_numpy()
    diff = p_old != p_new

    pred_cmp = {
        "test_keys": int(len(y_te)),
        "identical_predictions": int((~diff).sum()),
        "different_predictions": int(diff.sum()),
        "max_abs_level_shift": int(np.abs(p_old[diff] - p_new[diff]).max()) if diff.any() else 0,
        "shift_distribution": {},
        "old_unrounded": _unrounded(y_te, p_old),
        "new_unrounded": _unrounded(y_te, p_new),
        "old_fit_rows": int(n_old),
        "new_fit_rows": int(n_new),
        "old_fit_input_sha256": fp_old,
        "new_fit_input_sha256": fp_new,
        "features": list(X_old.columns),
        "hyperparameters": {"criterion": "gini", "max_depth": 9, "min_samples_leaf": 20},
    }
    if diff.any():
        shifts, counts = np.unique(p_new[diff] - p_old[diff], return_counts=True)
        for k, v in zip(shifts, counts):
            pred_cmp["shift_distribution"][str(int(k))] = int(v)

    if diff.any():
        pd.DataFrame({
            "station_id": keys_old.loc[diff, "station_id"].to_numpy(),
            "timestamp": keys_old.loc[diff, "timestamp"].to_numpy(),
            "y_true": y_te[diff],
            "pred_old": p_old[diff],
            "pred_new": p_new[diff],
        }).to_csv(os.path.join(args.out_dir, "predictions_diff.csv"), index=False)

    out = {"reconciliation": rec, "prediction_comparison": pred_cmp}
    with open(os.path.join(args.out_dir, "evidence.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("== 行数对账")
    print(f"  网格 {len(full):,} | 旧保留 {len(old):,} 剔除 {len(old_dropped)} | "
          f"新保留 {len(new):,} 剔除 {len(new_dropped)}")
    print(f"  净减少 {rec['net_row_change']:+d} = " +
          " + ".join(f"{k} {v}" for k, v in reasons.items()))
    print(f"  恒等式：156 + (328-226) + (102-89) = {rec['identity_check']['sum']}")
    print("== 预测比对")
    print(f"  同一批测试键 {pred_cmp['test_keys']:,}，预测不同 {pred_cmp['different_predictions']} 行，"
          f"最大等级偏移 {pred_cmp['max_abs_level_shift']}")
    print(f"  旧未舍入 Macro-F1 {pred_cmp['old_unrounded']['Macro_F1']:.10f}")
    print(f"  新未舍入 Macro-F1 {pred_cmp['new_unrounded']['Macro_F1']:.10f}")
    print(f"  旧/新 训练行 {n_old:,} / {n_new:,}")
    print(f"✓ {os.path.join(args.out_dir, 'evidence.json')}")


if __name__ == "__main__":
    main()
