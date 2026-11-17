"""派生分析的新鲜度检查。

`metrics_stage7.json` 是从缓存预测、冻结阈值和接缝报告算出来的。上游任何一项重算之后，
这个 JSON 仍然能渲染，页面就会把旧结论配在新数字旁边，而且看不出异常。比一次 sha256
很便宜，可以把这种静默错配变成一句显式的“请重跑”。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# key in the artefact's input_sha256 → file under models/
STAGE7_SOURCES = {
    "ordinal_probs_val_test": "ordinal_probs_val_test.npz",
    "stage5_predictions": "stage5_predictions.npz",
    "cost_thresholds": "cost_thresholds.json",
    "seam_report": "seam_report.json",
}

STAGE7_TABULAR = {"features_final": ("features_final.parquet", "data")}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stale_inputs(record: dict | None, models_dir: Path, data_dir: Path) -> list[str]:
    """Keys whose recorded fingerprint no longer matches the file on disk."""
    if not isinstance(record, dict):
        return ["record"]
    recorded = record.get("input_sha256") or {}
    stale: list[str] = []
    for key, filename in STAGE7_SOURCES.items():
        expected = recorded.get(key)
        path = Path(models_dir) / filename
        if not expected or not path.exists():
            continue
        if sha256_file(path) != expected:
            stale.append(key)
    for key, (filename, folder) in STAGE7_TABULAR.items():
        expected = recorded.get(key)
        base = Path(data_dir) if folder == "data" else Path(models_dir)
        path = base / filename
        if not expected or not path.exists():
            continue
        if sha256_file(path) != expected:
            stale.append(key)
    return stale
