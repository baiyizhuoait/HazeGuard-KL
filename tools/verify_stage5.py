"""Verify stage-05 artifacts from saved predictions; never predict test again."""

import hashlib
import ast
import importlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interval_view import summary as interval_summary
from ordinal_forest import cumulative_to_proba
from utils_metrics import evaluate_predictions
R = importlib.import_module("05_reg_to_cls")


def text_hash_matches(path, expected):
    """Git can convert LF/CRLF on checkout; accept only that byte difference."""
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return expected in {hashlib.sha256(b).hexdigest()
                        for b in (raw, lf, lf.replace(b"\n", b"\r\n"))}


def verify():
    md = ROOT / "models"
    m = json.loads((md / "metrics_stage5.json").read_text(encoding="utf-8"))
    parts, names, hashes = R.load_data(ROOT)
    assert m["input_sha256"]["features_final"] == hashes["features_final"]
    assert text_hash_matches(md / "selected_features.json", m["input_sha256"]["selected_features"])
    assert m["feature_names"] == names
    assert m["test_evaluation_count"] == 1
    # 原训练摘要保留；重构后的代码使用单独登记的摘要。
    sources = m.get("refactor", {}).get("code_sha256", m["code_sha256"])
    for name, expected in sources.items():
        assert text_hash_matches(ROOT / "src" / name, expected), name
    for name, expected in m["artifacts"].items():
        if name.endswith(".json"):
            assert text_hash_matches(md / name, expected), name
        else:
            assert R.sha256(md / name) == expected, name
    frozen = json.loads((md / "stage5_selection.json").read_text(encoding="utf-8"))
    assert frozen["frozen_at"] <= m["generated_at"]
    cuts = m["m5_threshold_search"]["selected_thresholds"]
    assert cuts == frozen["m5_val_tuned_thresholds"]
    with np.load(md / "ordinal_probs_val_test.npz", allow_pickle=False) as p, \
         np.load(md / "stage5_predictions.npz", allow_pickle=False) as d:
        for split, suffix in (("val", "validation_metrics_seed42"), ("test", "test_metrics_seed42")):
            part = parts[split]
            n = len(part)
            prob, y = p[f"proba_{split}"], p[f"y_{split}"]
            assert prob.shape == (n,4) and y.shape == (n,)
            assert np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all()
            np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-12)
            np.testing.assert_array_equal(p[f"ts_{split}"], part.timestamp.to_numpy())
            np.testing.assert_array_equal(p[f"station_id_{split}"], part.station_id.to_numpy(dtype=str))
            np.testing.assert_array_equal(y, part.target_level_3h)
            np.testing.assert_allclose(prob, cumulative_to_proba(d[f"raw_cumulative_{split}"]))
            predictions = {"M5_fixed": R.levels(d[f"reg_{split}"]),
                           "M5_val_tuned": R.levels(d[f"reg_{split}"], cuts),
                           "M6_Ordinal_RF": prob.argmax(axis=1)}
            for name, pred in predictions.items():
                actual = evaluate_predictions(y, pred, prob if name.startswith("M6") else None)
                assert actual == m["models"][name][suffix], (split, name)
            interval = R.interval_metrics(part.target_pm25_3h, d[f"quantiles_{split}"])
            assert interval == m["interval"]["validation" if split == "val" else "test"]
    # Load independently of a training __main__ module; only inspect metadata here.
    for filename in ("rf_regressor.joblib", "ordinal_rf.joblib"):
        model = joblib.load(md / filename)
        expected_features = (m["models"]["M6_Ordinal_RF"].get("feature_names", names)
                             if filename == "ordinal_rf.joblib" else names)
        assert model.selected_features_ == expected_features
        assert model.random_state == 42 and model.n_features_in_ == 25
        assert (md / filename).stat().st_size < 20 * 1024**2
    if m["models"]["M6_Ordinal_RF"].get("qc_refit") is False:
        # Preserved M6 results are historical, even though their row keys now match the QC cache.
        archive = ROOT / m["models"]["M6_Ordinal_RF"]["cache_source"]
        original = json.loads((archive / "fingerprints.json").read_text(encoding="utf-8"))
        assert R.sha256(md / "ordinal_rf.joblib") == original["sha256"]["ordinal_rf.joblib"]
        refit = importlib.import_module("05_refit_m5_qc")
        with np.load(archive / "ordinal_probs_val_test.npz", allow_pickle=False) as before, \
                np.load(md / "ordinal_probs_val_test.npz", allow_pickle=False) as after:
            for split in ("val", "test"):
                index = refit.align_legacy_rows(parts[split], before, split)
                np.testing.assert_array_equal(after[f"proba_{split}"], before[f"proba_{split}"][index])
        print("NOTE: M6 is a historical cache aligned by keys; it was not QC-refitted.")
    if (md / "stage5_qc_refit_plan.json").exists():
        adaptive = importlib.import_module("05_interval_adaptive")
        fixed = importlib.import_module("05_interval_calibration")
        plan = json.loads((md / "stage5_qc_refit_plan.json").read_text(encoding="utf-8"))
        assert plan["new_candidate_search"] is False and plan["new_threshold_search"] is False
        assert plan["M6_refit"] is False and plan["random_seeds"] == [42, 43, 44]
        with np.load(md / "stage5_predictions.npz", allow_pickle=False) as saved:
            for module, suffix, cal_name in (
                (adaptive, "adaptive", "interval_calibration_adaptive.json"),
                (fixed, "calibrated", "interval_calibration.json"),
            ):
                cal = json.loads((md / cal_name).read_text(encoding="utf-8"))
                scores = json.loads((md / f"metrics_stage5_interval_{suffix}.json").read_text(encoding="utf-8"))
                assert cal["n_calibration"] == len(parts["val"])
                assert text_hash_matches(md / cal_name, scores["calibration_sha256"])
                assert scores["new_model_fits"] == scores["new_model_test_predictions"] == 0
                qv = saved["quantiles_val"]
                fitted = (adaptive.fit_calibration(parts["val"].target_pm25_3h, qv,
                                                   coverage=cal["nominal_coverage"], floor=cal["scale_floor"])
                          if suffix == "adaptive" else fixed.fit_calibration(
                              parts["val"].target_pm25_3h, qv[:, 1], coverage=cal["nominal_coverage"]))
                for key, value in fitted.items():
                    assert cal[key] == value, (suffix, key)
                band_path = md / f"stage5_intervals_{suffix}.npz"
                assert R.sha256(band_path) == scores["intervals_sha256"]
                with np.load(band_path, allow_pickle=False) as bands:
                    assert set(bands.files) == {f"{key}_{s}" for key in ("interval", "ts", "station_id") for s in ("val", "test")}
                    for split in ("val", "test"):
                        frame, q = parts[split], saved[f"quantiles_{split}"]
                        band = module.apply_calibration(q if suffix == "adaptive" else q[:, 1], cal)
                        np.testing.assert_array_equal(bands[f"interval_{split}"], band)
                        np.testing.assert_array_equal(bands[f"ts_{split}"], frame.timestamp.to_numpy())
                        np.testing.assert_array_equal(bands[f"station_id_{split}"], frame.station_id.to_numpy(dtype=str))
                        y = frame.target_pm25_3h.to_numpy()
                        overall = scores[split]["overall"] if suffix == "adaptive" else scores[split]["calibrated"]
                        assert fixed.interval_scores(y, band) == overall
                        for node in frame.station_id.unique():
                            mask = frame.station_id.eq(node).to_numpy()
                            node_scores = scores[split]["by_node"][node]
                            if suffix != "adaptive":
                                node_scores = node_scores["calibrated"]
                            assert fixed.interval_scores(y[mask], band[mask]) == node_scores
        print("PASS: frozen QC recipe; validation-only calibration rank; cached adaptive/fixed intervals and node scores.")
        # Execute the app's actual loading functions without installing or rendering Streamlit.
        needed = {"read_json", "apply_thresholds", "load_thresholds", "load_predictions",
                  "interval_label", "load_json_or_none", "self_check"}
        functions = []
        for node in ast.parse((ROOT / "app.py").read_text(encoding="utf-8")).body:
            if isinstance(node, ast.FunctionDef) and node.name in needed:
                node.decorator_list = []
                functions.append(node)
        assert {f.name for f in functions} == needed
        scope = {"np": np, "pd": pd, "json": json, "Path": Path, "MODELS": md, "HORIZON_H": 3,
                 "evaluate_predictions": evaluate_predictions,
                 "interval_summary": interval_summary}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "app.py loading functions", "exec"), scope)
        displayed = scope["load_predictions"]()
        assert displayed.groupby("split").size().to_dict() == {"val": len(parts["val"]), "test": len(parts["test"])}
        assert not displayed.duplicated(["station_id", "timestamp"]).any()
        assert displayed[["cal_lo", "cal_mid", "cal_hi"]].notna().all().all()
        label, note = scope["interval_label"]()
        current = json.loads((md / "metrics_stage5_interval_adaptive.json").read_text(encoding="utf-8"))
        assert f"{current['test']['overall']['PICP']:.1%}" in note
        assert scope["self_check"](displayed) is None, "M7's historical validation-cost record needs owner review"
        print("PASS: dashboard cache loader and interval label; 33,294 aligned rows. Streamlit UI was not rendered.")
    for filename in ("fig_ordinal_error_distribution.png", "fig_interval_band.png"):
        with Image.open(ROOT / "figures" / filename) as im:
            assert im.width >= 3000 and im.height >= 1000
            assert min(im.info.get("dpi", (0,0))) >= 299
    print("PASS: inputs/code/artifact hashes; probability and row contracts; saved metrics; "
          "intervals; model loading; 300dpi figures. No new test prediction.")


if __name__ == "__main__":
    verify()
