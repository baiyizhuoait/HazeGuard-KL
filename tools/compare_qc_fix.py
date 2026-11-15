"""
1 号 · 修复前后对比（插值规则 + 切分冲刷）

把两份面板对齐到 (station_id, timestamp)，逐项统计：
  行数、三段行数、被剔除的行、发生变化的单元格、质量标记、目标等级
再在「同一套可建模行」上重算 M1，避免把数据变化和评估口径变化混在一起。

用法：
    python tools/compare_qc_fix.py \
        --old data/haze_panel_clean.parquet \
        --new .scratch/qc_fix/fixed/data/haze_panel_clean.parquet \
        --model .scratch/qc_fix/fixed/models/baseline_dt.joblib \
        --out .scratch/qc_fix/comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import config as C  # noqa: E402
import utils_metrics as UM  # noqa: E402

VALUE_COLS = ["pm2_5", "pm10", "no2", "so2", "o3", "co", "temperature_2m",
              "relative_humidity_2m", "wind_speed_ms", "wind_direction_10m", "precipitation"]
KEY = ["station_id", "timestamp"]


def _read(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _m1_on(model_path: str, panel: pd.DataFrame, keys: pd.DataFrame) -> dict:
    """在给定行集上复算 M1。keys 为空表示用全部受体行。"""
    from importlib import util
    spec = util.spec_from_file_location("bl", os.path.join(ROOT, "src", "01_baseline_dt.py"))
    bl = util.module_from_spec(spec)
    spec.loader.exec_module(bl)
    X, y, d = bl.build_xy(panel)
    keep = np.ones(len(d), dtype=bool)
    if keys is not None:
        merged = d.merge(keys, on=KEY, how="left")
        keep = merged["audit_model_eligible"].fillna(False).to_numpy()
        d = d[keep].reset_index(drop=True)
        X = X[keep].reset_index(drop=True)
        y = y[keep].reset_index(drop=True)
    mask = (d["split"] == "test").to_numpy()
    payload = joblib.load(model_path)
    pred = payload["model"].predict(X[mask].astype(float))
    return UM.evaluate_predictions(y[mask], pred)


def _m1_refit_eligible(panel: pd.DataFrame, keys: pd.DataFrame) -> dict:
    """与主表同口径：只用可建模行，train 重训、test 评估。"""
    from importlib import util
    from sklearn.tree import DecisionTreeClassifier
    spec = util.spec_from_file_location("bl", os.path.join(ROOT, "src", "01_baseline_dt.py"))
    bl = util.module_from_spec(spec)
    spec.loader.exec_module(bl)
    X, y, d = bl.build_xy(panel)
    merged = d.merge(keys, on=KEY, how="left")
    keep = merged["audit_model_eligible"].fillna(False).to_numpy()
    d = d[keep].reset_index(drop=True)
    X = X[keep].reset_index(drop=True)
    y = y[keep].reset_index(drop=True)
    tr = (d["split"] == "train").to_numpy()
    te = (d["split"] == "test").to_numpy()
    model = DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=20,
                                   random_state=C.RANDOM_STATE).fit(X[tr].astype(float), y[tr])
    return UM.evaluate_predictions(y[te], model.predict(X[te].astype(float)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=os.path.join(C.DATA_DIR, "haze_panel_clean.parquet"))
    ap.add_argument("--new", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    old, new = _read(args.old), _read(args.new)
    out: dict = {"rows_old": int(len(old)), "rows_new": int(len(new)),
                 "rows_delta": int(len(new) - len(old))}
    out["split_old"] = {k: int(v) for k, v in old["split"].value_counts().items()}
    out["split_new"] = {k: int(v) for k, v in new["split"].value_counts().items()}

    m = old.merge(new, on=KEY, how="outer", suffixes=("_o", "_n"), indicator=True)
    both = m[m["_merge"] == "both"]
    dropped = m[m["_merge"] == "left_only"]
    out["shared_rows"] = int(len(both))
    out["dropped_rows"] = int(len(dropped))

    cell = {c: int((~np.isclose(both[c + "_o"], both[c + "_n"], equal_nan=True)).sum())
            for c in VALUE_COLS}
    out["changed_cells"] = cell
    out["changed_cells_total"] = int(sum(cell.values()))

    any_change = np.zeros(len(both), dtype=bool)
    for c in VALUE_COLS:
        any_change |= ~np.isclose(both[c + "_o"], both[c + "_n"], equal_nan=True)
    out["changed_rows"] = int(any_change.sum())

    out["qc"] = {}
    for c in ("qc_interpolated", "is_missing", "qc_calm_wind", "qc_pm25_gt_pm10"):
        out["qc"][c] = {"old": int(both[c + "_o"].sum()), "new": int(both[c + "_n"].sum())}

    tl = both[both["target_level_3h_o"].notna() & both["target_level_3h_n"].notna()]
    out["target_level_changed"] = int((tl["target_level_3h_o"] != tl["target_level_3h_n"]).sum())
    out["target_value_changed"] = int(
        (~np.isclose(tl["target_pm25_3h_o"], tl["target_pm25_3h_n"], equal_nan=True)).sum())

    out["dropped_by_split"] = {k: int(v) for k, v in dropped["split_o"].value_counts().items()}
    out["dropped_by_role"] = {k: int(v) for k, v in dropped["role_o"].value_counts().items()}
    out["dropped_by_node"] = {k: int(v) for k, v in dropped["station_id"].value_counts().items()}

    # 同一套可建模行（旧特征表的 audit 标记）上比较 M1
    feat = pd.read_parquet(os.path.join(C.DATA_DIR, "feat_spatiotemporal.parquet"),
                           columns=["station_id", "timestamp", "audit_model_eligible"])
    feat["timestamp"] = pd.to_datetime(feat["timestamp"])
    out["m1_all_rows"] = _m1_on(args.model, new, None)
    out["m1_eligible_rows"] = _m1_on(args.model, new, feat)
    out["m1_eligible_refit"] = _m1_refit_eligible(new, feat)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"✓ {args.out}")
    print(f"  行数 {out['rows_old']:,} → {out['rows_new']:,}（{out['rows_delta']:+,}）")
    print(f"  剔除 {out['dropped_rows']} 行：{out['dropped_by_split']}  按角色 {out['dropped_by_role']}")
    print(f"  变化单元格 {out['changed_cells_total']}（{cell}）")
    print(f"  目标等级变化 {out['target_level_changed']} 行，目标浓度变化 {out['target_value_changed']} 行")
    print(f"  M1（全部受体行） Macro-F1 {out['m1_all_rows']['Macro_F1']:.4f}  "
          f"召回 {out['m1_all_rows']['High_Risk_Recall']:.4f}")
    print(f"  M1（可建模行）   Macro-F1 {out['m1_eligible_rows']['Macro_F1']:.4f}  "
          f"召回 {out['m1_eligible_rows']['High_Risk_Recall']:.4f}")
    print(f"  M1（可建模行重训）Macro-F1 {out['m1_eligible_refit']['Macro_F1']:.4f}  "
          f"召回 {out['m1_eligible_refit']['High_Risk_Recall']:.4f}")


if __name__ == "__main__":
    main()
