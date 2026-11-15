"""质控与切分规则的回归测试（1 号，2026-11-15）

这两条规则之前都是「写在注释里、代码没做到」：
  1. 缺口超过 3 小时不插值 —— 实际用 .interpolate(limit=3) 把长缺口的前 3 个值补上了
  2. 切分不许把标签漏到下一段 —— 实际按特征时间切，边界前 3 小时的标签进了验证段

Run with: python -m pytest tests/test_qc_rules.py
"""

import importlib.util
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


P = load("pipeline01", "01_data_pipeline.py")
B = load("baseline01", "01_baseline_dt.py")


def test_short_gap_is_filled():
    s = pd.Series([1.0, 2.0, np.nan, np.nan, np.nan, 10.0])
    out = P._interp_short_gaps(s, 3)
    assert out.isna().sum() == 0
    assert out.iloc[2] == pytest.approx(4.0)


def test_long_gap_is_left_entirely_nan():
    s = pd.Series([1.0, 2.0] + [np.nan] * 5 + [10.0])
    out = P._interp_short_gaps(s, 3)
    assert out.isna().sum() == 5
    assert out.iloc[0] == 1.0 and out.iloc[-1] == 10.0


def test_boundary_between_three_and_four():
    three = pd.Series([1.0, np.nan] + [np.nan] * 2 + [4.0])
    four = pd.Series([1.0, np.nan] + [np.nan] * 3 + [5.0])
    assert P._interp_short_gaps(three, 3).isna().sum() == 0
    assert P._interp_short_gaps(four, 3).isna().sum() == 4


def test_head_and_tail_gaps_are_not_filled():
    s = pd.Series([np.nan, np.nan, 3.0, 4.0, np.nan])
    out = P._interp_short_gaps(s, 3)
    assert out.iloc[0:2].isna().all()
    assert out.iloc[-1] != out.iloc[-1]


def test_split_uses_label_time_not_feature_time():
    n = 200
    df = pd.DataFrame({
        "station_id": "KL_City",
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
    })
    out = P.assign_split(df)
    label = out["timestamp"] + pd.Timedelta(hours=P.C.HORIZON_H)
    assert (out.split == "purged").sum() > 0

    out = out[out.split != "purged"]
    tr = out.loc[out.split == "train"]
    va = out.loc[out.split == "val"]
    te = out.loc[out.split == "test"]
    assert len(tr) and len(va) and len(te)
    assert label[tr.index].max() < va["timestamp"].min()
    assert label[va.index].max() < te["timestamp"].min()


def test_purged_folds_never_see_validation_labels():
    stamps = pd.date_range("2026-01-01", periods=600, freq="h").to_numpy()
    folds = list(B._folds_by_timestamp(stamps, 5, horizon=3))
    assert len(folds) == 5
    label = stamps + np.timedelta64(3, "h")
    for tr_idx, va_idx in folds:
        assert label[tr_idx].max() < stamps[va_idx].min()


def _panel_frame(n_rows, start, pm_base):
    ts = pd.date_range(start, periods=n_rows, freq="h")
    return pd.DataFrame({
        "station_id": "KL_City",
        "timestamp": ts,
        "month": ts.month,
        "hour": ts.hour,
        "pm2_5": pm_base + np.arange(n_rows) % 5,
        "temperature_2m": 27.0 + (np.arange(n_rows) % 3),
        # 真实面板里这两列已存在，折内重算时会被覆盖
        "pm2_5_anomaly": np.nan,
        "temp_anomaly": np.nan,
    })


def test_infold_baseline_ignores_validation_values():
    """只改折内验证段的数据，折内训练所得气候态与训练特征必须一动不动。"""
    feats = ["pm2_5", "temperature_2m", "pm2_5_anomaly", "temp_anomaly"]
    tr = _panel_frame(240, "2026-01-01", 10.0)
    va = _panel_frame(48, "2026-01-11", 80.0)

    stamps = np.concatenate([tr["timestamp"].to_numpy(), va["timestamp"].to_numpy()])
    combined = pd.concat([tr, va], ignore_index=True)
    tr_idx, va_idx = list(B._folds_by_timestamp(stamps, 3, horizon=3))[-1]

    look_before = B._baseline_lookup(combined.iloc[tr_idx])
    X_before = B._anomaly_matrix(combined.iloc[tr_idx], look_before, feats)

    # 只动验证段：把浓度改成不可能的值
    mutated = combined.copy()
    mutated.loc[va_idx, "pm2_5"] = -999.0
    mutated.loc[va_idx, "temperature_2m"] = 99.0

    look_after = B._baseline_lookup(mutated.iloc[tr_idx])
    X_after = B._anomaly_matrix(mutated.iloc[tr_idx], look_after, feats)

    pd.testing.assert_frame_equal(X_before, X_after)
    pd.testing.assert_frame_equal(look_before, look_after)

    # 反向对照：如果误把验证段并进训练段，气候态必须变，否则这条测试是空的
    look_wrong = B._baseline_lookup(pd.concat([mutated.iloc[tr_idx], mutated.iloc[va_idx]]))
    assert not look_before.equals(look_wrong)
