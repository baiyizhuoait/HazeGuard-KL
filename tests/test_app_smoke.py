"""看板必须能在默认数据下跑通，并且在自适应区间文件缺席时安全回退。

第二条是 7 号复现的那个 `KeyError: 'PICP'`：adaptive 文件不在时，页面会退回读
calibrated 文件，而两份文件的 by_node 嵌套层级不同。这里把 adaptive 文件临时改名，
让回退路径真正被执行，然后恢复。
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
ADAPTIVE = ROOT / "models" / "metrics_stage5_interval_adaptive.json"


def _run() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    return at


def test_app_loads_with_default_data():
    at = _run()
    assert not at.exception, at.exception
    assert len(at.tabs) >= 1


@pytest.mark.skipif(not ADAPTIVE.exists(), reason="自适应区间文件不存在，无法构造回退场景")
def test_app_survives_without_the_adaptive_interval_file():
    backup = ADAPTIVE.with_suffix(".json.pytest-bak")
    ADAPTIVE.rename(backup)
    try:
        at = _run()
        assert not at.exception, at.exception
    finally:
        backup.rename(ADAPTIVE)
    assert ADAPTIVE.exists()
