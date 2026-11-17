"""区间产物的两套嵌套结构都要能读，缺文件时也不能抛异常。

回归点：`utils` 里原来直接取 ``test["by_node"][node]["PICP"]``，在 adaptive 文件缺失、
回退到 calibrated 文件时抛 ``KeyError: 'PICP'``，因为 calibrated 的 by_node 多一层。
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import interval_view as iv  # noqa: E402

ADAPTIVE = ROOT / "models" / "metrics_stage5_interval_adaptive.json"
CALIBRATED = ROOT / "models" / "metrics_stage5_interval_calibrated.json"


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} 尚未生成")
    return json.loads(path.read_text(encoding="utf-8"))


def test_adaptive_record_reads_overall():
    info = iv.summary(_load(ADAPTIVE))
    assert info is not None
    assert 0.0 <= info["test"]["PICP"] <= 1.0
    assert 0.0 <= info["val"]["PICP"] <= 1.0
    assert set(info["nodes"]) == {"KL_City", "Port_Klang", "Sepang"}
    for block in info["nodes"].values():
        assert "PICP" in block and "MPIW" in block


def test_calibrated_record_reads_nested_layer():
    """这是原来炸掉的那条路径：没有 overall，by_node 多一层。"""
    info = iv.summary(_load(CALIBRATED))
    assert info is not None
    assert 0.0 <= info["test"]["PICP"] <= 1.0
    assert 0.0 <= info["val"]["PICP"] <= 1.0
    for name, block in info["nodes"].items():
        assert "PICP" in block, name


def test_missing_sections_return_none_instead_of_raising():
    assert iv.summary(None) is None
    assert iv.summary({}) is None
    assert iv.summary({"test": {"overall": {}}}) is None
    assert iv.summary({"test": {"by_node": {}}}) is None
    # by_node 里混进一个没有 PICP 的块，不能连累整体
    info = iv.summary({
        "nominal_coverage": 0.9,
        "test": {"overall": {"PICP": 0.88, "MPIW": 30.0}, "by_node": {"X": {"foo": 1}}},
        "val": {"overall": {"PICP": 0.90, "MPIW": 25.0}},
    })
    assert info["test"]["PICP"] == 0.88
    assert info["nodes"] == {}


def test_pick_block_prefers_the_first_layer_with_numbers():
    container = {
        "overall": {"PICP": 0.1, "MPIW": 1.0},
        "calibrated": {"PICP": 0.2, "MPIW": 2.0},
    }
    assert iv.pick_block(container)["PICP"] == 0.1
    assert iv.pick_block({"calibrated": {"PICP": 0.2}})["PICP"] == 0.2
    assert iv.pick_block({"raw_tree_spread": {"PICP": 0.3}})["PICP"] == 0.3
