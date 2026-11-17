"""用修复后的（带 3 小时净空的）折重算 M2/M3/M4 的交叉验证。

参数全部沿用 `metrics_stage4.json` 里已经选定的组合，不重新搜索、不改种子，因此这不是
一次调参，而是把同一组配置在一个更严格的折上重新打分。旧搜索记录（rf_search_trace /
boosting_search_trace）保持原样，结果另存。

用法：
    python tools/recompute_stage4_cv.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402


def load_stage4():
    spec = importlib.util.spec_from_file_location("stage4", ROOT / "src" / "04_rf_and_boosting.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(model: str, params: dict, n_jobs: int):
    if model == "M2_Nominal_RF":
        return RandomForestClassifier(**params, class_weight=None,
                                      random_state=C.RANDOM_STATE, n_jobs=n_jobs)
    if model == "M3_Balanced_RF":
        return RandomForestClassifier(**params, class_weight="balanced_subsample",
                                      random_state=C.RANDOM_STATE, n_jobs=n_jobs)
    if model == "M4_Boosting":
        return HistGradientBoostingClassifier(**params, random_state=C.RANDOM_STATE)
    raise ValueError(model)


def main() -> None:
    started = time.perf_counter()
    stage4 = load_stage4()
    metrics_path = ROOT / "models" / "metrics_stage4.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    frame, features = stage4.load_and_validate(stage4.FEATURE_PATH, stage4.SELECTION_PATH)
    train = stage4.split_frame(frame, "train")
    x_all = train[features].to_numpy(dtype=float)
    y_all = train["target_level_3h"].to_numpy()
    stamps = train["timestamp"].to_numpy()

    folds = stage4.chronological_splits(stamps)
    sizes = [(int(t.size), int(v.size)) for t, v in folds]
    print("folds (train, valid) rows:", sizes)

    results: dict[str, dict] = {}
    for model in ("M2_Nominal_RF", "M3_Balanced_RF", "M4_Boosting"):
        block = metrics["models"][model]
        params = block["hyperparameters"]
        scores = []
        for number, (train_index, valid_index) in enumerate(folds, start=1):
            estimator = build(model, params, n_jobs=-1)
            estimator.fit(x_all[train_index], y_all[train_index])
            score = stage4.macro_f1(y_all[valid_index], estimator.predict(x_all[valid_index]))
            scores.append(score)
            print(f"  {model} fold {number}: {score:.4f}", flush=True)
        results[model] = {
            "hyperparameters": params,
            "class_weight": block.get("class_weight"),
            "cv_fold_macro_f1": [round(s, 6) for s in scores],
            "cv_mean_macro_f1": round(float(np.mean(scores)), 6),
            "cv_std_macro_f1": round(float(np.std(scores)), 6),
        }
        old = next((r for r in metrics.get("rf_search_trace", [])
                    if r.get("params") == params), None)
        if old is None:
            old = next((r for r in metrics.get("boosting_search_trace", [])
                        if r.get("params") == params), None)
        if old is not None:
            results[model]["cv_mean_macro_f1_before_purge"] = old["cv_mean_macro_f1"]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "tools/recompute_stage4_cv.py",
        "why": ("The original folds split on timestamps only, so the last training hours "
                "carried labels inside the validation window. Parameters are unchanged; "
                "only the folds are stricter."),
        "cv_scheme": "TimeSeriesSplit(5) on unique train timestamps, 3 h horizon purge",
        "n_splits": len(folds),
        "fold_sizes": [{"train": t, "valid": v} for t, v in sizes],
        "rows_by_split": {k: int(v) for k, v in train["split"].value_counts().items()},
        "models": results,
        "search_trace_untouched": True,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    out = ROOT / "models" / "metrics_stage4_cv_fixed.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics["cv_fixed"] = {
        "artifact": "models/metrics_stage4_cv_fixed.json",
        "cv_scheme": payload["cv_scheme"],
        "note": ("用固定参数、带 3 小时净空的折重算；旧的 rf_search_trace 与 "
                 "boosting_search_trace 保留为当时的搜索记录，未改动。"),
        "models": {k: {"cv_mean_macro_f1": v["cv_mean_macro_f1"],
                       "cv_std_macro_f1": v["cv_std_macro_f1"]} for k, v in results.items()},
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    for model, block in results.items():
        before = block.get("cv_mean_macro_f1_before_purge")
        delta = "" if before is None else f" (was {before:.4f})"
        print(f"{model}: {block['cv_mean_macro_f1']:.4f} ± {block['cv_std_macro_f1']:.4f}{delta}")
    print(f"✓ {out}")


if __name__ == "__main__":
    main()
