"""
1 号 · 第 1 / 2.1 / 2.2 节的数字核算（写报告前先跑这个）

为什么单独一支脚本：报告里的数分散在清洗报告、门槛报告、基线指标、面板本身
四个地方，其中一部分是派生量（切分日期、月度占比、混淆矩阵单元格、逐节点
等级占比），任何一处改了流程都不会自动同步。这个脚本把它们一次算出来，
落到 models/sec1_audit.json，报告只从这份 json 取数。

用法：
    python tools/audit_section1.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import config as C  # noqa: E402

PANEL = os.path.join(C.DATA_DIR, "haze_panel_clean.parquet")
FEAT = os.path.join(C.DATA_DIR, "feat_spatiotemporal.parquet")
OUT = os.path.join(C.MODEL_DIR, "sec1_audit.json")

FEATURES = [
    "pm2_5", "pm10", "no2", "so2", "o3", "co",
    "temperature_2m", "relative_humidity_2m", "wind_speed_ms",
    "wind_direction_10m", "precipitation",
    "pm2_5_anomaly", "temp_anomaly",
    "hour", "month", "dayofweek",
]


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def _load() -> pd.DataFrame:
    df = pd.read_parquet(PANEL)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _eligible_keys() -> pd.DataFrame:
    f = pd.read_parquet(FEAT, columns=["station_id", "timestamp", "audit_model_eligible"])
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    return f


def main() -> None:
    df = _load()
    clean = json.load(open(os.path.join(C.MODEL_DIR, "cleaning_report.json"), encoding="utf-8"))
    gate0 = json.load(open(os.path.join(C.MODEL_DIR, "gate0_report.json"), encoding="utf-8"))
    stage1 = json.load(open(os.path.join(C.MODEL_DIR, "metrics_stage1.json"), encoding="utf-8"))
    stage4 = json.load(open(os.path.join(C.MODEL_DIR, "metrics_stage4.json"), encoding="utf-8"))

    rec = df[df["role"] == "receptor"].copy()
    out: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _commit(),
        "scope": "sections 1, 2.1, 2.2 of the group report",
    }

    # ---- 面板与清洗对账 -------------------------------------------------
    out["panel"] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "nodes": int(df["station_id"].nunique()),
        "months": int(df["timestamp"].dt.to_period("M").nunique()),
        "first_timestamp": str(df["timestamp"].min()),
        "last_timestamp": str(df["timestamp"].max()),
    }
    out["cleaning_reconciliation"] = {
        "rows_fetched": clean["rows_fetched"],
        "leading_block_trimmed": clean["leading_block_trimmed"],
        "rows_after_trim": clean["rows_total"],
        "rows_dropped_no_target": clean["rows_dropped_no_target"],
        "rows_dropped_no_anchor": clean["rows_dropped_no_anchor"],
        "rows_final": clean["rows_final"],
        "retention_pct": round(clean["rows_final"] / clean["rows_fetched"] * 100, 2),
        "interpolated_cells": clean["interpolated_cells"],
        "rows_interpolated": clean["rows_interpolated"],
        "rows_still_missing": clean["rows_with_missing"],
        "duplicates_removed": clean["duplicates_removed"],
        "grid_rows_added": clean["grid_rows_added"],
    }

    # ---- 切分（面板级与受体级各一份）-------------------------------------
    splits = {}
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        r = rec[rec["split"] == s]
        splits[s] = {
            "rows": int(len(d)),
            "receptor_rows": int(len(r)),
            "start": str(d["timestamp"].min()),
            "end": str(d["timestamp"].max()),
        }
    out["splits"] = splits
    out["receptor_split"] = {
        "train": int(len(rec[rec.split == "train"])),
        "val": int(len(rec[rec.split == "val"])),
        "test": int(len(rec[rec.split == "test"])),
        "test_eligible": int(C.PANEL_CONTRACT["receptor_test_eligible_rows"]),
        "val_eligible": int(C.PANEL_CONTRACT["receptor_val_eligible_rows"]),
        "train_eligible": int(C.PANEL_CONTRACT["receptor_train_eligible_rows"]),
    }

    # ---- 逐节点 ---------------------------------------------------------
    key = _eligible_keys()
    rec2 = rec.merge(key, on=["station_id", "timestamp"], how="left")
    rows = []
    for node in C.NODES:
        d = rec2[rec2.station_id == node["id"]]
        if d.empty:
            d = df[df.station_id == node["id"]]
        rows.append({
            "id": node["id"],
            "role": node["role"],
            "region": node["region"],
            "lat": node["lat"],
            "lon": node["lon"],
            "rows": int(len(d)),
            "mean_pm25": round(float(d["pm2_5"].mean()), 2),
            "level3_share_pct": round(float((d["target_level_3h"] == 3).mean() * 100), 2)
            if d["target_level_3h"].notna().any() else None,
        })
    out["nodes"] = rows

    # ---- 测试窗口 -------------------------------------------------------
    kl = rec[(rec.station_id == "KL_City") & (rec.split == "test")].copy()
    kl["ym"] = kl["timestamp"].dt.to_period("M")
    monthly = {}
    for per, g in kl.groupby("ym"):
        monthly[str(per)] = {
            "level3_share_pct": round(float((g["target_level_3h"] == 3).mean() * 100), 2),
            "p95_pm25": round(float(np.percentile(g["pm2_5"], 95)), 2),
            "rows": int(len(g)),
        }
    ep = kl[(kl.timestamp >= "2026-08-01") & (kl.timestamp <= "2026-10-31")].sort_values("timestamp")
    out["test_window"] = {
        "kl_city_monthly": monthly,
        "episode_max_3h_rise": round(float(ep["pm2_5"].diff().rolling(3).sum().max()), 1),
        "episode_hours_above_100": int((ep["pm2_5"] > 100).sum()),
        "episode_max_pm25": round(float(ep["pm2_5"].max()), 1),
        "level_distribution_train": gate0["level_distribution_receptor_train"],
        "level_distribution_test": gate0["level_distribution_receptor_test"],
    }

    # ---- M0 / M1（可建模行）---------------------------------------------
    d = rec.copy()
    rad = np.deg2rad(d["wind_direction_10m"])
    d["wd_sin"] = np.sin(rad)
    d["wd_cos"] = np.cos(rad)
    d = d.dropna(subset=FEATURES + ["target_level_3h"]).merge(
        key, on=["station_id", "timestamp"], how="left")
    d = d[d.audit_model_eligible.fillna(False)]
    te = d[d.split == "test"]
    tr = d[d.split == "train"].sort_values("timestamp")
    cols = FEATURES + ["wd_sin", "wd_cos"]
    model = DecisionTreeClassifier(
        criterion="gini", max_depth=9, min_samples_leaf=20,
        random_state=C.RANDOM_STATE).fit(tr[cols].astype(float), tr["target_level_3h"].astype(int))
    X = te[cols].astype(float)
    y = te["target_level_3h"].astype(int)
    pred = model.predict(X)
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y.to_numpy(), pred):
        cm[t, p] += 1
    refit = {
        "Macro_F1": round(float(f1_score(y, pred, average="macro")), 4),
        "Balanced_Accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "High_Risk_Recall": round(float(((pred >= 2) & (y >= 2)).sum() / max(1, (y >= 2).sum())), 4),
    }
    stored = stage1["metrics_test_eligible"]
    check = {k: abs(refit[k] - stored[k]) < 1e-4 for k in refit}
    out["m0_m1_eligible"] = {
        "n": int(len(te)),
        "n_high_risk": int(y.isin([2, 3]).sum()),
        "confusion": cm.tolist(),
        "m0": stage4["metrics_persistence_M0_test"],
        "m1": stored,
        "m1_refit": refit,
        "refit_matches_stored": check,
        "m1_best_params": stage1["best_params"],
        "m1_cv_macro_f1": stage1["cv_macro_f1"],
        "m1_train_macro_f1": stage1["metrics_train"]["Macro_F1"],
        "m1_val_macro_f1": stage1["metrics_val"]["Macro_F1"],
        "m1_leaves": stage1["tree_structure"]["n_leaves"],
    }
    out["per_node_test"] = stage1["per_node_test"]

    imp = pd.Series(model.feature_importances_, index=list(cols)).sort_values(
        ascending=False)
    out["m1_importance_top5"] = {k: round(float(v), 3) for k, v in imp.head(5).items()}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"✓ {OUT}")
    print(f"  面板 {out['panel']['rows']:,} 行 / {out['panel']['nodes']} 节点 / "
          f"{out['panel']['months']} 个月")
    print(f"  M0/M1 可建模测 {out['m0_m1_eligible']['n']:,} 行，"
          f"混淆矩阵对角 {np.trace(cm):,}")


if __name__ == "__main__":
    main()
