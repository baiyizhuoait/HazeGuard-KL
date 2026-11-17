"""4 号的交叉验证必须做三小时净空：训练行的标签不能落进验证窗口。

原始实现只保证训练「时间戳」早于验证段，而标签比时间戳晚 3 小时，所以每折训练集末尾
几小时的标签仍然在验证期里。这里同时验证两件事：净空生效，以及净空确实删掉了行
（否则测试是空的，永远通过）。
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HORIZON = 3


@pytest.fixture(scope="module")
def stage4():
    spec = importlib.util.spec_from_file_location("stage4", ROOT / "src" / "04_rf_and_boosting.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _panel(n_hours: int = 300, nodes: int = 3):
    stamps = pd.date_range("2026-01-01", periods=n_hours, freq="h")
    frame = pd.DataFrame({
        "timestamp": np.repeat(stamps.to_numpy(), nodes),
        "station_id": np.tile([f"S{i}" for i in range(nodes)], n_hours),
    })
    return frame


def test_every_fold_purges_the_label_horizon(stage4):
    frame = _panel()
    folds = stage4.chronological_splits(frame["timestamp"], n_splits=5)
    assert len(folds) == 5

    stamps = pd.to_datetime(frame["timestamp"]).to_numpy()
    for train_rows, valid_rows in folds:
        train_label = stamps[train_rows] + np.timedelta64(HORIZON, "h")
        assert train_label.max() < stamps[valid_rows].min()


def test_purge_actually_removes_rows(stage4):
    """对照：不净空时每折会有标签越界的行，净空后为 0。"""
    frame = _panel()
    stamps = pd.to_datetime(frame["timestamp"]).to_numpy()
    folds = stage4.chronological_splits(frame["timestamp"], n_splits=5)

    removed_total = 0
    for train_rows, valid_rows in folds:
        valid_start = stamps[valid_rows].min()
        # 重建未净空的训练集：所有时间戳早于验证起点的行
        naive = np.flatnonzero(stamps < valid_start)
        leaking = naive[(stamps[naive] + np.timedelta64(HORIZON, "h")) >= valid_start]
        removed_total += leaking.size
        assert not set(leaking).intersection(train_rows)
    assert removed_total > 0, "构造的数据无法体现净空效果，测试需要重做"


def test_too_few_timestamps_is_rejected(stage4):
    with pytest.raises(ValueError):
        stage4.chronological_splits(pd.date_range("2026-01-01", periods=3, freq="h"), n_splits=5)
