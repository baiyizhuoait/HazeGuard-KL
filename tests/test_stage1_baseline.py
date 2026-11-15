"""M1 基线的集成测试（1 号，2026-11-15 复核后补）

对应外部复核提的三类问题：
  1. 默认入口不能跑进未修复的调参路径（内层 TimeSeriesSplit 会跨折泄漏）；
  2. 最终训练与验证/测试必须用同一份气候态预处理；
  3. 主表指标对应的模型必须落盘，重新加载后能复现同样的预测与指标。

Run with: python -m pytest tests/test_stage1_baseline.py -q
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = load("baseline01_tests", "01_baseline_dt.py")
UM = load("metrics_tests", "utils_metrics.py")


def _synthetic(n=240, start="2026-01-01", base=10.0):
    ts = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({
        "station_id": "KL_City",
        "timestamp": ts,
        "month": ts.month,
        "hour": ts.hour,
        "pm2_5": base + (np.arange(n) % 7),
        "temperature_2m": 27.0 + (np.arange(n) % 3),
        "pm2_5_anomaly": 0.0,
        "temp_anomaly": 0.0,
    })


def test_search_flag_is_disabled():
    """默认入口不能进未修复的搜索路径：--search 必须直接拒绝，而不是悄悄跑。"""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "src" / "01_baseline_dt.py"), "--search"],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "--search 已禁用" in (proc.stderr + proc.stdout)


def test_folds_cover_the_tail():
    """末折必须覆盖到最后一个时间戳，不能把尾巴三小时漏掉。"""
    stamps = pd.date_range("2026-01-01", periods=600, freq="h").to_numpy()
    folds = list(B._folds_by_timestamp(stamps, 5, horizon=3))
    assert len(folds) == 5
    covered = np.zeros(len(stamps), dtype=bool)
    for tr_idx, va_idx in folds:
        covered[va_idx] = True
        assert (stamps[tr_idx] + np.timedelta64(3, "h")).max() < stamps[va_idx].min()
    # 第一个 block 留给最初的训练，其余每个唯一时间戳都应进入某一折的验证段
    assert covered[600 // 6:].all()
    assert covered[-1], "最后一个时间戳没有进任何验证折"


def test_boundary_violation_raises():
    """外层边界有越界就该报错，而不是打印一行继续训练。"""
    ts = pd.date_range("2026-01-01", periods=36, freq="h")
    # 干净切分：train 到 16:00，val 从 20:00 起（中间留 3 小时空档）
    clean = pd.DataFrame({
        "timestamp": ts,
        "split": ["train"] * 17 + ["skip"] * 3 + ["val"] * 6 + ["skip"] * 4 + ["test"] * 6,
    })
    B.check_outer_boundaries(clean)       # 干净数据不报

    # 把 train 往后延两小时，标签就会跨进 val
    dirty = clean.copy()
    dirty.loc[17:18, "split"] = "train"
    with pytest.raises(RuntimeError):
        B.check_outer_boundaries(dirty)


def test_one_preprocess_applied_to_all_splits():
    """train 拟合的气候态必须同时作用在 train/val/test 上。"""
    tr = _synthetic(120, "2026-01-01", 10.0)
    va = _synthetic(48, "2026-01-06", 40.0)
    te = _synthetic(48, "2026-01-08", 80.0)
    feats = ["pm2_5", "temperature_2m", "pm2_5_anomaly", "temp_anomaly"]

    look = B._baseline_lookup(tr)
    for frame in (tr, va, te):
        X = B._anomaly_matrix(frame, look, feats)
        merged = frame.merge(look, on=["station_id", "month", "hour"], how="left")
        assert np.allclose(X["pm2_5_anomaly"], merged["pm2_5"] - merged["p"])
        assert np.allclose(X["temp_anomaly"], merged["temperature_2m"] - merged["t"])


def test_metrics_are_fixed_params_and_saved_model_reproduces_them():
    """产物自洽：指标跑的是固定参数，且保存的模型重新加载后能复现同一批预测。"""
    metrics_path = ROOT / "models" / "metrics_stage1.json"
    model_path = ROOT / "models" / "baseline_dt.joblib"
    if not (metrics_path.exists() and model_path.exists()):
        pytest.skip("尚未生成 M1 产物")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["search_disabled"] is True
    assert "fixed" in payload["params_source"]
    assert payload["preprocess"]["applied_to"] == "train, val, test"

    import joblib
    bundle = joblib.load(model_path)
    assert bundle["schema"] == "hazeguard-baseline-v2"
    assert "preprocess" in bundle and "lookup" in bundle["preprocess"]

    panel = pd.read_parquet(ROOT / "data" / "haze_panel_clean.parquet")
    X, y, d = B.build_xy(panel)
    d = d.reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X_all = B._anomaly_matrix(d, bundle["preprocess"]["lookup"], bundle["features"])

    keep = np.ones(len(d), dtype=bool)
    if payload.get("eligibility"):
        keys = pd.read_parquet(payload["eligibility"]["source"],
                               columns=["station_id", "timestamp", "audit_model_eligible"])
        keys["timestamp"] = pd.to_datetime(keys["timestamp"])
        merged = d.merge(keys, on=["station_id", "timestamp"], how="left")
        keep = merged["audit_model_eligible"].fillna(False).to_numpy()

    mask = keep & (d["split"] == "test").to_numpy()
    pred = bundle["model"].predict(X_all[mask])
    got = UM.evaluate_predictions(y[mask], pred)

    for k in ("Macro_F1", "Balanced_Accuracy", "High_Risk_Recall", "MAOE",
              "Severe_Jumping_Error_Rate", "Avg_Asymmetric_Cost"):
        assert abs(got[k] - payload["metrics"][k]) < 1e-9, (k, got[k], payload["metrics"][k])
    assert got["n_samples"] == payload["metrics"]["n_samples"]


def test_model_transform_matches_panel_anomaly_columns():
    """模型用的气候态必须和面板里 pm2_5_anomaly / temp_anomaly 完全一致。

    不一致就意味着「训练用的特征」和「交付的面板列」是两套数，下游任何人
    拿面板复现模型都会对不上。
    """
    metrics_path = ROOT / "models" / "metrics_stage1.json"
    model_path = ROOT / "models" / "baseline_dt.joblib"
    if not (metrics_path.exists() and model_path.exists()):
        pytest.skip("尚未生成 M1 产物")

    import joblib
    bundle = joblib.load(model_path)
    panel = pd.read_parquet(ROOT / "data" / "haze_panel_clean.parquet")
    panel["split"] = panel["split"].astype(str)
    X, y, d = B.build_xy(panel)
    d = d.reset_index(drop=True)
    X = X.reset_index(drop=True)

    Xa = B._anomaly_matrix(d, bundle["preprocess"]["lookup"], bundle["features"])
    for col in ("pm2_5_anomaly", "temp_anomaly"):
        diff = np.abs(Xa[col].to_numpy() - X[col].to_numpy())
        assert np.nanmax(diff) < 1e-9, (col, np.nanmax(diff))
