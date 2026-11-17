"""派生产物的指纹校验：上游重算后，旧分析必须被判为过期。"""

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cache_freshness as cf  # noqa: E402

MODELS = ROOT / "models"
DATA = ROOT / "data"
STAGE7 = MODELS / "metrics_stage7.json"


def _record():
    if not STAGE7.exists():
        pytest.skip("尚未生成 stage7")
    return json.loads(STAGE7.read_text(encoding="utf-8"))


def test_current_stage7_matches_its_inputs():
    """仓库里的 stage7 必须与当前缓存同源，否则页面会隐藏它。"""
    assert cf.stale_inputs(_record(), MODELS, DATA) == []


def test_missing_or_empty_record_is_stale():
    assert cf.stale_inputs(None, MODELS, DATA) == ["record"]
    assert cf.stale_inputs({}, MODELS, DATA) == []


def test_wrong_fingerprint_is_flagged():
    record = _record()
    record = json.loads(json.dumps(record))
    record["input_sha256"]["stage5_predictions"] = "0" * 64
    assert "stage5_predictions" in cf.stale_inputs(record, MODELS, DATA)


def test_missing_file_is_not_treated_as_stale():
    """指纹有记录但文件不在（例如浅克隆未拉 LFS）时不该误报。"""
    record = _record()
    record = json.loads(json.dumps(record))
    record["input_sha256"]["seam_report"] = "0" * 64
    tmp = MODELS.parent / ".scratch" / "freshness_probe"
    tmp.mkdir(parents=True, exist_ok=True)
    empty = tmp / "models"
    empty.mkdir(exist_ok=True)
    assert cf.stale_inputs(record, empty, tmp) == []
