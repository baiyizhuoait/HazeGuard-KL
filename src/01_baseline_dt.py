#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01b 基线模型：Decision Tree（组长负责）

用途
----
给出「地板性能」：只用当前时刻的基础观测，不做任何滞后、滚动、空间传导特征。
后续 M2–M6 的所有增益都必须相对这条地板线解释，否则无法证明复杂度是必要的。

训练域
------
只在**受体节点**（KL_City / Port_Klang / Sepang 三个真实 CAMS 格点）上训练与评估。
源区节点不参与本基线——它们是 探索负责人 的空间特征来源，不是预测目标。

产出
----
models/baseline_dt.joblib
models/metrics_stage1.json
figures/fig_baseline_dt_confusion.png
figures/fig_baseline_dt_importance.png

用法
----
python src/01_baseline_dt.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import List

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import utils_metrics as UM  # noqa: E402

PANEL = os.path.join(C.DATA_DIR, "haze_panel_clean.parquet")

# 基础特征：当前时刻的污染物 + 气象 + 去基线残差 + 时间位置
BASE_FEATURES = [
    "pm2_5", "pm10", "no2", "so2", "o3", "co",
    "temperature_2m", "relative_humidity_2m", "wind_speed_ms",
    "wind_direction_10m", "precipitation",
    "pm2_5_anomaly", "temp_anomaly",
    "hour", "month", "dayofweek",
]


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL)
    df["split"] = df["split"].astype(str)
    return df


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    取受体节点、构造特征矩阵。
    风向是循环量：359° 与 1° 只差 2°，直接当数值会让树在 0/360 处劈出假边界，
    因此额外派生 sin/cos 两列（仍属于「基础气象特征」，不算滞后/传导特征）。
    """
    d = df[df["role"] == "receptor"].copy()
    rad = np.deg2rad(d["wind_direction_10m"])
    d["wd_sin"] = np.sin(rad)
    d["wd_cos"] = np.cos(rad)

    feats = BASE_FEATURES + ["wd_sin", "wd_cos"]
    d = d.dropna(subset=feats + ["target_level_3h"])
    X = d[feats].astype(float)
    y = d["target_level_3h"].astype(int)
    return X, y, d


def _baseline_lookup(train_df: pd.DataFrame) -> pd.DataFrame:
    """用给定训练段重算 (station, month, hour) 气候态。

    折内重算的原因：pm2_5_anomaly / temp_anomaly 是学习出来的特征。若在交叉验证里
    沿用「全训练段」算好的气候态，折的训练部分其实已经看过折验证期的数据（未来信息
    通过组均值渗进特征）。标签隔离只解决一半问题，预处理必须一起跟着折走。
    """
    g = (train_df.groupby(["station_id", "month", "hour"], observed=True)
         .agg(p=("pm2_5", "mean"), t=("temperature_2m", "mean"), n=("pm2_5", "size"))
         .reset_index())
    low = g["n"] < C.MIN_BASELINE_SUPPORT
    if low.any():
        sm = (train_df.groupby(["station_id", "month"], observed=True)
              .agg(fp=("pm2_5", "mean"), ft=("temperature_2m", "mean")).reset_index())
        g = g.merge(sm, on=["station_id", "month"], how="left")
        g.loc[low, "p"] = g.loc[low, "fp"]
        g.loc[low, "t"] = g.loc[low, "ft"]
        g = g[["station_id", "month", "hour", "p", "t"]]
    return g


def _anomaly_matrix(frame: pd.DataFrame, lookup: pd.DataFrame, feats: list) -> pd.DataFrame:
    """把 frame 的 anomaly 两列换成 lookup 口径，其余特征保持不变。"""
    merged = frame.merge(lookup, on=["station_id", "month", "hour"], how="left")
    X = merged[feats].astype(float).copy()
    X["pm2_5_anomaly"] = merged["pm2_5"] - merged["p"]
    X["temp_anomaly"] = merged["temperature_2m"] - merged["t"]
    return X


def _folds_by_timestamp(stamps: np.ndarray, n_splits: int, horizon: int = C.HORIZON_H):
    """按唯一时间戳切折；训练行的标签时间必须严格早于验证起点。

    不用 sklearn 的 TimeSeriesSplit：它按行切片，不认识 t+3h 的标签时间，也不认识
    多节点面板，训练集末尾 3 小时的行其实把标签写进了验证折。
    """
    uniq = np.unique(stamps)
    if len(uniq) < n_splits + 1:
        raise ValueError("时间戳太少，无法切折")
    block = len(uniq) // (n_splits + 1)
    label = stamps + np.timedelta64(horizon, "h")
    for i in range(1, n_splits + 1):
        start = uniq[i * block]
        stop = uniq[min((i + 1) * block, len(uniq) - 1)]
        train_mask = label < start
        val_mask = (stamps >= start) & (stamps < stop)
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        yield np.flatnonzero(train_mask), np.flatnonzero(val_mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=PANEL, help="面板路径；修复前后对比时指向临时面板")
    ap.add_argument("--out-dir", default=None, help="产物写到指定目录，默认写 models/ 与 figures/")
    ap.add_argument("--fixed-params", action="store_true",
                    help="不重新搜索超参数，沿用修复前选出的 gini / depth 9 / leaf 20")
    args = ap.parse_args()

    model_dir = C.MODEL_DIR if args.out_dir is None else os.path.join(args.out_dir, "models")
    fig_dir = C.FIGURE_DIR if args.out_dir is None else os.path.join(args.out_dir, "figures")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    t_start = time.time()
    print("[1/5] 读取面板")
    df = pd.read_parquet(args.panel)
    df["split"] = df["split"].astype(str)
    X, y, d = build_xy(df)
    split = d["split"].to_numpy()
    n_tr = int((split == "train").sum())
    n_va = int((split == "val").sum())
    n_te = int((split == "test").sum())
    print(f"      受体样本 {len(d):,}  train/val/test = {n_tr:,}/{n_va:,}/{n_te:,}")

    feats = list(X.columns)
    d_tr = d.loc[split == "train"].reset_index(drop=True)
    y_tr = d_tr["target_level_3h"].astype(int).to_numpy()
    X_va, y_va = X[split == "val"], y[split == "val"]
    X_te, y_te = X[split == "test"], y[split == "test"]

    # 外层边界自检：训练段任何一行的标签都必须早于验证段起点
    for a, b in (("train", "val"), ("val", "test")):
        prev = d[d["split"] == a]
        nxt_start = d.loc[d["split"] == b, "timestamp"].min()
        bad = int((prev["timestamp"] + pd.Timedelta(hours=C.HORIZON_H) >= nxt_start).sum())
        print(f"      边界 {a}->{b}：标签越界 {bad} 行  （应为 0）")

    grid = {
        "max_depth": [3, 5, 7, 9, 12, None],
        "min_samples_leaf": [5, 20, 50, 100],
        "criterion": ["gini", "entropy"],
    }
    fixed = dict(criterion="gini", max_depth=9, min_samples_leaf=20)

    print("[2/5] 折内重算气候态 + 按唯一时间戳切折的 5 折评估")
    fold_scores, fold_params = [], []
    stamps = d_tr["timestamp"].to_numpy()
    for k, (tr_idx, va_idx) in enumerate(_folds_by_timestamp(stamps, 5), 1):
        look = _baseline_lookup(d_tr.iloc[tr_idx])
        Xf = _anomaly_matrix(d_tr.iloc[tr_idx], look, feats)
        Xv = _anomaly_matrix(d_tr.iloc[va_idx], look, feats)
        yf, yv = y_tr[tr_idx], y_tr[va_idx]
        if args.fixed_params:
            est = DecisionTreeClassifier(random_state=C.RANDOM_STATE, **fixed)
        else:
            est = GridSearchCV(
                DecisionTreeClassifier(random_state=C.RANDOM_STATE), grid,
                cv=TimeSeriesSplit(n_splits=3), scoring="f1_macro", n_jobs=-1, refit=True)
        est.fit(Xf, yf)
        score = UM.evaluate_predictions(yv, est.predict(Xv))["Macro_F1"]
        fold_scores.append(score)
        if not args.fixed_params:
            fold_params.append(est.best_params_)
        print(f"      fold {k}: train={tr_idx.size:,}  val={va_idx.size:,}  macro-F1={score:.4f}")

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    print(f"      5 折均值 {cv_mean:.4f}  std {cv_std:.4f}")

    if args.fixed_params:
        best_params = dict(fixed)
        params_source = "carried over from the pre-fix run, no re-tuning"
    else:
        keys = ["criterion", "max_depth", "min_samples_leaf"]
        best_params = {k: pd.Series([p[k] for p in fold_params]).mode().iloc[0] for k in keys}
        params_source = "mode of the fold-level grid search"

    print("[3/5] 全训练段拟合（气候态用 train 段重算）")
    look_all = _baseline_lookup(d_tr)
    X_tr_refit = _anomaly_matrix(d_tr, look_all, feats)
    best = DecisionTreeClassifier(random_state=C.RANDOM_STATE, **best_params).fit(X_tr_refit, y_tr)
    print(f"      params={best_params}  ({params_source})")

    m_tr = UM.evaluate_predictions(y_tr, best.predict(X_tr_refit))
    m_va = UM.evaluate_predictions(y_va, best.predict(X_va))
    gap = m_tr["Macro_F1"] - m_va["Macro_F1"]
    print(f"      train Macro-F1={m_tr['Macro_F1']:.4f}  val={m_va['Macro_F1']:.4f}  gap={gap:.4f}")

    print("[4/5] test 段评估（修复后重评；测试集此前已看过，不能再说从未查看）")
    y_te_pred = best.predict(X_te)
    m_te = UM.evaluate_predictions(y_te, y_te_pred)
    UM.print_metrics(m_te, "Baseline DecisionTree · test")

    per_node = {}
    test_rows = d.loc[split == "test"]
    for sid in C.RECEPTORS:
        mask = (test_rows["station_id"] == sid).to_numpy()
        if mask.sum() == 0:
            continue
        per_node[sid] = UM.evaluate_predictions(y_te[mask], y_te_pred[mask])
    print("      分节点 Macro-F1：" +
          "  ".join(f"{k}={v['Macro_F1']:.3f}" for k, v in per_node.items()))

    m0_te_pred = _level_from_pm25(test_rows["pm2_5"].to_numpy())
    m0_va_pred = _level_from_pm25(d.loc[split == "val", "pm2_5"].to_numpy())
    m0_te = UM.evaluate_predictions(y_te, m0_te_pred)
    m0_va = UM.evaluate_predictions(y_va, m0_va_pred)
    UM.print_metrics(m0_te, "M0 Persistence · test")

    print("")
    print("      对照（test）      M0 持续性   M1 决策树   差值")
    for k in ("Macro_F1", "Balanced_Accuracy", "High_Risk_Recall",
              "MAOE", "Severe_Jumping_Error_Rate", "Avg_Asymmetric_Cost"):
        print(f"      {k:24s} {m0_te[k]:8.4f}   {m_te[k]:8.4f}   {m_te[k] - m0_te[k]:+.4f}")

    print("[5/5] 保存模型与指标")
    model_path = os.path.join(model_dir, "baseline_dt.joblib")
    joblib.dump({"model": best, "features": feats}, model_path, compress=3)

    def _sha_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _sha_fit(Xf: pd.DataFrame, yf: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(Xf.to_numpy(dtype="float64")).tobytes())
        h.update(np.ascontiguousarray(np.asarray(yf, dtype="int64")).tobytes())
        h.update("|".join(Xf.columns).encode("utf-8"))
        return h.hexdigest()

    payload = {
        "module": "01 数据工程与基线",
        "owner": "1号（组长）",
        "model": "DecisionTreeClassifier (baseline)",
        "stage": "test",
        "metrics": {k: m_te[k] for k in (
            "Macro_F1", "Balanced_Accuracy", "High_Risk_Recall",
            "MAOE", "Severe_Jumping_Error_Rate", "Avg_Asymmetric_Cost",
            "n_samples", "n_high_risk")},
        "metrics_persistence_M0_test": m0_te,
        "metrics_persistence_M0_val": m0_va,
        "metrics_train": m_tr,
        "metrics_val": m_va,
        "per_node_test": per_node,
        "best_params": best_params,
        "params_source": params_source,
        "cv_macro_f1": round(cv_mean, 4),
        "cv_macro_f1_std": round(cv_std, 4),
        "cv_fold_scores": [round(v, 4) for v in fold_scores],
        "cv_scheme": "unique-timestamp blocks, train label_time < val start, 3h horizon",
        "baseline_refit_in_fold": True,
        "split_protocol": "outer split assigned by label timestamp (t+3h)",
        "overfit_gap_train_minus_val": round(float(gap), 4),
        "features": feats,
        "n_features": len(feats),
        "n_train": int(len(d_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "class_distribution_train": {str(k): int(v) for k, v in pd.Series(y_tr).value_counts().sort_index().items()},
        "class_distribution_test": {str(k): int(v) for k, v in pd.Series(y_te).value_counts().sort_index().items()},
        "tree_structure": {"n_leaves": int(best.get_n_leaves()), "depth": int(best.get_depth())},
        "fit_rows": int(len(X_tr_refit)),
        "fit_features": feats,
        "fit_input_sha256": _sha_fit(X_tr_refit, y_tr),
        "panel_path": os.path.abspath(args.panel),
        "panel_sha256": _sha_file(args.panel),
        "model_file_sha256": _sha_file(model_path),
        "trained_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "model_loaded_from_disk": False,
        "train_universe": "receptor nodes only (KL_City, Port_Klang, Sepang)",
        "train_universe_note": "主实验统一只在 3 个受体节点上训练与验证；23 个走廊/源区节点只作特征输入与泛化实验对照",
        "metric_version": "High_Risk_Recall = P(ŷ∈{2,3} | y∈{2,3})，诊断项 diag_* 不参与横向排序",
        "evaluation_status": "post-fix re-evaluation",
        "test_previously_viewed": True,
        "evaluation_note": ("插值规则与切分修复后重评。测试集此前已查看过一次，"
                            "本次结果不构成「从未看过测试集」的声明；"
                            "超参数沿用修复前选择，未按测试表现调整。"),
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    UM.save_metrics(payload, os.path.join(model_dir, "metrics_stage1.json"))

    _plot_confusion(y_te, y_te_pred, fig_dir)
    _plot_importance(best, feats, fig_dir)

    print("")
    print(f"✓ {model_path}")
    print(f"✓ {model_dir}/metrics_stage1.json")
    print(f"✓ {fig_dir}/fig_baseline_dt_confusion.png")
    print(f"✓ {fig_dir}/fig_baseline_dt_importance.png")
    print(f"  叶子 {best.get_n_leaves()} 个，深度 {best.get_depth()}，耗时 {time.time() - t_start:.0f}s")


def _level_from_pm25(arr) -> np.ndarray:
    """按 config 阈值把浓度离散成四级。用于构造持续性基线。"""
    a = np.asarray(arr, dtype=float)
    lo, mid, hi = C.LEVEL_THRESHOLDS
    return np.select([a < lo, a < mid, a < hi], [0, 1, 2], default=3).astype(int)


def _plot_confusion(y_true, y_pred, fig_dir: str) -> None:
    cm = UM.confusion(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=200)
    im = ax.imshow(cm, cmap="Blues")
    names = [f"L{i}" for i in range(4)]
    ax.set_xticks(range(4)); ax.set_xticklabels(names)
    ax.set_yticks(range(4)); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted level"); ax.set_ylabel("True level")
    ax.set_title("Baseline Decision Tree — Test Confusion Matrix", fontsize=11)
    thr = cm.max() / 2.0
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thr else "#222222", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_baseline_dt_confusion.png"))
    plt.close(fig)


def _plot_importance(model, cols: List[str], fig_dir: str) -> None:
    imp = pd.Series(model.feature_importances_, index=list(cols)).sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=200)
    ax.barh(imp.index, imp.values, color="#2E5496")
    ax.set_xlabel("Gini importance")
    ax.set_title("Baseline Decision Tree — Feature Importance", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_baseline_dt_importance.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
