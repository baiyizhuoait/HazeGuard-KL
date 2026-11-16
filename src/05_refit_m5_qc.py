"""Refit M5 after the QC repair, with its original final recipe frozen.

Run once: python src/05_refit_m5_qc.py --n-jobs 8
Old records are archived before replacement. M6 is not fitted or predicted.
"""

import argparse
import importlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from utils_metrics import evaluate_predictions

R = importlib.import_module("05_reg_to_cls")
G = importlib.import_module("05_ordinal_rf")
A = importlib.import_module("05_interval_adaptive")
I = importlib.import_module("05_interval_calibration")
APP = importlib.import_module("05_apply_adaptive_interval")
ROOT = R.ROOT
ARCHIVE = "docs/qc_fix/pre_refit_stage5"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def align_legacy_rows(frame, saved, split):
    """Keep cached M6 predictions only for exactly matching, unchanged target keys."""
    old = pd.MultiIndex.from_arrays([saved[f"ts_{split}"], saved[f"station_id_{split}"].astype(str)])
    new = pd.MultiIndex.from_arrays([frame.timestamp, frame.station_id.astype(str)])
    if not old.is_unique or not new.is_unique:
        raise ValueError("duplicate timestamp/station key")
    index = old.get_indexer(new)
    if (index < 0).any():
        raise ValueError(f"{split}: a current key has no historical M6 prediction")
    if not np.array_equal(saved[f"y_{split}"][index], frame.target_level_3h.to_numpy()):
        raise ValueError(f"{split}: historical and current class targets differ")
    return index


def main(n_jobs=8):
    md, archive = ROOT / "models", ROOT / ARCHIVE
    plan_path = md / "stage5_qc_refit_plan.json"
    if archive.exists() or plan_path.exists():
        raise FileExistsError("QC refit already has an audit record; inspect cached results, do not repeat test")
    parts, features, hashes = R.load_data()
    old = json.loads((md / "metrics_stage5.json").read_text(encoding="utf-8"))
    frozen = json.loads((md / "stage5_selection.json").read_text(encoding="utf-8"))
    old_cal = json.loads((md / "interval_calibration_adaptive.json").read_text(encoding="utf-8"))
    if hashes["features_final"] == old["input_sha256"]["features_final"]:
        raise ValueError("the feature table has not changed; this entry is only for a documented QC repair")
    params = frozen["selected_regressor_params"]
    cuts = frozen["m5_val_tuned_thresholds"]
    if frozen["final_seed"] != 42 or old["random_seeds"] != [42, 43, 44]:
        raise ValueError("unexpected frozen seeds")
    if old_cal["method"] != "tree-spread-normalized absolute residual":
        raise ValueError("unexpected calibration family")
    for name in ("rf_regressor.joblib", "ordinal_rf.joblib", "stage5_predictions.npz",
                 "ordinal_probs_val_test.npz"):
        if R.sha256(md / name) != old["artifacts"][name]:
            raise ValueError(f"old artifact does not match its audit record: {name}")
    with np.load(md / "ordinal_probs_val_test.npz", allow_pickle=False) as ids:
        indices = {s: align_legacy_rows(parts[s], ids, s) for s in ("val", "test")}
        legacy_probs = {s: ids[f"proba_{s}"][indices[s]] for s in indices}
    with np.load(md / "stage5_predictions.npz", allow_pickle=False) as cache:
        legacy_raw = {s: cache[f"raw_cumulative_{s}"][indices[s]] for s in indices}

    archive.mkdir(parents=True)
    names = ("metrics_stage5.json", "stage5_selection.json", "ordinal_probs_val_test.npz",
             "stage5_predictions.npz", "interval_calibration_adaptive.json",
             "metrics_stage5_interval_adaptive.json", "stage5_intervals_adaptive.npz",
             "interval_calibration.json", "metrics_stage5_interval_calibrated.json",
             "stage5_intervals_calibrated.npz", "m5_seam_mae.csv", "metrics_stage5_seam.json")
    original_hashes = {}
    for name in names:
        source = md / name
        if source.exists():
            shutil.copy2(source, archive / name)
            original_hashes[name] = R.sha256(source)
    original_hashes["rf_regressor.joblib"] = R.sha256(md / "rf_regressor.joblib")
    original_hashes["ordinal_rf.joblib"] = R.sha256(md / "ordinal_rf.joblib")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(archive / "fingerprints.json", {
        "source_commit": commit, "input_sha256": old["input_sha256"],
        "sha256": original_hashes, "model_recovery": "Original joblib files remain available at source_commit.",
    })
    plan = {"frozen_at": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "input_sha256": hashes, "feature_names": features,
            "selected_regressor_params": params, "m5_fixed_thresholds": frozen["m5_fixed_thresholds"],
            "m5_val_tuned_thresholds": cuts, "random_seeds": [42, 43, 44], "final_seed": 42,
            "adaptive_family": old_cal["method"], "nominal_coverage": old_cal["nominal_coverage"],
            "scale_floor": old_cal["scale_floor"], "new_candidate_search": False,
            "new_threshold_search": False, "M6_refit": False,
            "evaluation_status": "data-repair re-evaluation; test previously viewed",
            "phase": "recipe_frozen_before_training"}
    write_json(plan_path, plan)
    print("Frozen QC recipe:", json.dumps(plan, ensure_ascii=False), flush=True)

    train, val, test = (parts[s] for s in ("train", "val", "test"))
    X = train[features].to_numpy(dtype=np.float32)
    XV = val[features].to_numpy(dtype=np.float32)
    runs, model, q_val, reg_val = [], None, None, None
    for seed in (42, 43, 44):
        reg = RandomForestRegressor(**params, random_state=seed, n_jobs=n_jobs)
        reg.fit(X, train.target_pm25_3h.to_numpy())
        if seed == 42:
            model = reg
            trees = R.tree_predictions(reg, XV)
            reg_val = trees.mean(axis=1)
            q_val = np.quantile(trees, [.05, .5, .95], axis=1).T
            prediction = reg_val
        else:
            prediction = reg.predict(XV)
        scores = {"M5_fixed": evaluate_predictions(val.target_level_3h, R.levels(prediction)),
                  "M5_val_tuned": evaluate_predictions(val.target_level_3h, R.levels(prediction, cuts))}
        runs.append({"seed": seed, "validation_metrics": scores})
        print(f"M5 validation seed {seed}: {json.dumps(scores)}", flush=True)

    model.selected_features_ = features
    model.level_thresholds_fixed_ = frozen["m5_fixed_thresholds"]
    model.level_thresholds_val_tuned_ = cuts
    joblib.dump(model, md / "rf_regressor.joblib", compress=("xz", 6))
    adaptive = A.fit_calibration(val.target_pm25_3h.to_numpy(), q_val,
                                coverage=plan["nominal_coverage"], floor=plan["scale_floor"])
    fixed = I.fit_calibration(val.target_pm25_3h.to_numpy(), q_val[:, 1],
                              coverage=plan["nominal_coverage"])
    plan.update(phase="calibration_frozen_before_test_predictions",
                adaptive_scale_multiplier=adaptive["scale_multiplier"],
                fixed_half_width=fixed["half_width"], calibration_frozen_at=datetime.now(timezone.utc).isoformat())
    write_json(plan_path, plan)
    print("Calibration frozen; starting the single M5 QC test prediction batch.", flush=True)
    trees = R.tree_predictions(model, test[features].to_numpy(dtype=np.float32))
    reg_test = trees.mean(axis=1)
    q_test = np.quantile(trees, [.05, .5, .95], axis=1).T
    arrays, probabilities = {}, {}
    model_metrics = {name: {"validation_metrics_seed42": scores} for name, scores in runs[0]["validation_metrics"].items()}
    legacy_scores = {}
    for split, point, q in (("val", reg_val, q_val), ("test", reg_test, q_test)):
        frame = parts[split]
        arrays[f"reg_{split}"], arrays[f"quantiles_{split}"] = point, q
        arrays[f"raw_cumulative_{split}"] = legacy_raw[split]
        for label, thresholds in (("fixed", R.THRESHOLDS), ("tuned", cuts)):
            pred = R.levels(point, thresholds)
            arrays[f"pred_m5_{label}_{split}"] = pred
            name = "M5_fixed" if label == "fixed" else "M5_val_tuned"
            if split == "test":
                model_metrics[name]["test_metrics_seed42"] = evaluate_predictions(frame.target_level_3h, pred)
        arrays[f"pred_m6_{split}"] = legacy_probs[split].argmax(axis=1).astype(np.int8)
        legacy_scores[split] = evaluate_predictions(frame.target_level_3h, arrays[f"pred_m6_{split}"], legacy_probs[split])
        probabilities.update({f"proba_{split}": legacy_probs[split],
                              f"y_{split}": frame.target_level_3h.to_numpy(),
                              f"ts_{split}": frame.timestamp.to_numpy(),
                              f"station_id_{split}": frame.station_id.to_numpy(dtype=str)})
    np.savez_compressed(md / "stage5_predictions.npz", **arrays)
    np.savez_compressed(md / "ordinal_probs_val_test.npz", **probabilities)
    plan.update(phase="test_predictions_cached", test_evaluation_count=1)
    write_json(plan_path, plan)
    stability = G.summary([run["validation_metrics"] for run in runs])
    for name in model_metrics:
        model_metrics[name]["validation_mean_std"] = stability[name]
        model_metrics[name]["qc_refit"] = True
    model_metrics["M6_Ordinal_RF"] = {
        "qc_refit": False, "status": "historical predictions aligned by timestamp/station; not a QC-refitted model",
        "training_input_sha256": old["input_sha256"], "feature_names": old["feature_names"],
        "validation_metrics_seed42": legacy_scores["val"], "test_metrics_seed42": legacy_scores["test"],
        "original_validation_metrics_seed42": old["models"]["M6_Ordinal_RF"]["validation_metrics_seed42"],
        "original_validation_mean_std": old["models"]["M6_Ordinal_RF"]["validation_mean_std"],
        "cache_source": ARCHIVE, "original_rows_by_split": old["rows_by_split"],
        "selected_ordinal_params": frozen["selected_ordinal_params"]}
    frozen_new = {**plan, "frozen_at": plan["calibration_frozen_at"],
                  "test_evaluation_started": True, "note": "M5 refit only; final model parameters and risk cuts unchanged.",
                  "legacy_M6": {"qc_refit": False, "input_sha256": old["input_sha256"], "archive": ARCHIVE}}
    write_json(md / "stage5_selection.json", frozen_new)
    sources = {name: R.sha256(ROOT / "src" / name) for name in (
        "05_refit_m5_qc.py", "05_reg_to_cls.py", "05_ordinal_rf.py", "05_interval_adaptive.py", "05_interval_calibration.py",
        "05_apply_adaptive_interval.py", "ordinal_forest.py", "utils_metrics.py", "config.py")}
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "code_sha256": sources, "input_sha256": hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "training_domain": "role == receptor", "feature_names": features,
        "rows_by_split": {s: len(frame) for s, frame in parts.items()}, "random_seeds": [42, 43, 44],
        "selected_regressor_params": params, "parameter_candidates": old["parameter_candidates"],
        "selection_rule": "Original final M5 parameters and 17/33/55 risk cuts frozen; no new selection.",
        "cv": {"repeated": False, "reason": "No model selection in this fixed-recipe data-repair re-evaluation.", "original_record": ARCHIVE + "/metrics_stage5.json"},
        "m5_threshold_search": {"performed": False, "selected_thresholds": cuts,
                                "source": ARCHIVE + "/stage5_selection.json"},
        "seed_runs": runs, "validation_mean_std": stability, "models": model_metrics,
        "monotonicity": {s: G.clipping_diagnostics(legacy_raw[s]) for s in ("val", "test")},
        "interval": {"source_model": "QC-refitted M5 RandomForestRegressor",
                     "method": "5/50/95 percentiles of individual tree point predictions",
                     "warning": "Tree spread is uncalibrated; dashboard uses separate adaptive calibration.",
                     "validation": R.interval_metrics(val.target_pm25_3h, q_val),
                     "test": R.interval_metrics(test.target_pm25_3h, q_test)},
        "evaluation_status": plan["evaluation_status"], "test_previously_viewed": True,
        "test_evaluation_count": 1, "new_model_fits": {"M5": 3, "M6": 0},
        "test_prediction_batches": {"M5_all_tree_predictions": 1, "M6_three_cumulative_forests": 0},
        "legacy_M6_row_alignment": {s: {"old_rows": old["rows_by_split"][s], "current_rows": len(parts[s]),
                                            "dropped_old_rows": old["rows_by_split"][s] - len(parts[s]),
                                            "method": "exact timestamp/station join; unchanged class targets"} for s in indices},
        "artifacts": {name: R.sha256(md / name) for name in (
            "rf_regressor.joblib", "ordinal_rf.joblib", "ordinal_probs_val_test.npz",
            "stage5_predictions.npz", "stage5_selection.json", "stage5_qc_refit_plan.json")}}
    payload["runtime"] = payload["environment"].copy()
    write_json(md / "metrics_stage5.json", payload)
    for cal, filename in ((adaptive, "interval_calibration_adaptive.json"), (fixed, "interval_calibration.json")):
        cal.update(input_sha256=hashes, frozen_utc=plan["calibration_frozen_at"],
                   predictions_sha256=R.sha256(md / "stage5_predictions.npz"),
                   source_sha256=R.sha256(ROOT / "src" / (
                       "05_interval_adaptive.py" if filename == "interval_calibration_adaptive.json"
                       else "05_interval_calibration.py")),
                   selection_evidence={"family_fixed_from_previous_experiment": True,
                       "old_calibration": ARCHIVE + "/interval_calibration_adaptive.json",
                       "new_method_search": False, "test_used_for_selection": False},
                   limitations=["Data-repair re-evaluation: the original test outcomes were already known.",
                       "Original M5 model parameters and risk thresholds were retained without new selection.",
                       "Calibration multiplier was fitted only on the repaired validation split.",
                       "Time-series dependence and shift prevent a guarantee of 90% future or per-node coverage."])
        write_json(md / filename, cal)
    # These saved outputs belong to the old data; their exact bytes are in the archive.
    for name in ("metrics_stage5_interval_adaptive.json", "stage5_intervals_adaptive.npz",
                 "metrics_stage5_interval_calibrated.json", "stage5_intervals_calibrated.npz",
                 "m5_seam_mae.csv", "metrics_stage5_seam.json"):
        (md / name).unlink()
    adaptive_result = APP.main(ROOT)
    fixed_result = {key: fixed[key] for key in ("nominal_coverage", "half_width", "unit", "limitations")}
    fixed_result.update(method="validation absolute residual about frozen M5 tree median",
                        calibration_sha256=R.sha256(md / "interval_calibration.json"),
                        new_model_fits=0, new_model_test_predictions=0)
    fixed_arrays = {}
    for split, q in (("val", q_val), ("test", q_test)):
        frame = parts[split]
        band = I.apply_calibration(q[:, 1], fixed)
        y = frame.target_pm25_3h.to_numpy()
        fixed_result[split] = {"raw_tree_spread": I.interval_scores(y, q),
                              "calibrated": I.interval_scores(y, band), "by_node": {}}
        for node in sorted(frame.station_id.unique()):
            mask = frame.station_id.eq(node).to_numpy()
            fixed_result[split]["by_node"][node] = {"raw_tree_spread": I.interval_scores(y[mask], q[mask]),
                                                    "calibrated": I.interval_scores(y[mask], band[mask])}
        fixed_arrays.update({f"interval_{split}": band, f"ts_{split}": frame.timestamp.to_numpy(),
                             f"station_id_{split}": frame.station_id.to_numpy(dtype=str)})
    np.savez_compressed(md / "stage5_intervals_calibrated.npz", **fixed_arrays)
    fixed_result["intervals_sha256"] = R.sha256(md / "stage5_intervals_calibrated.npz")
    write_json(md / "metrics_stage5_interval_calibrated.json", fixed_result)
    G.error_figure(test.target_level_3h.to_numpy(), {name: arrays[f"pred_m5_{key}_test"] for name, key in (
        ("M5 fixed (QC)", "fixed"), ("M5 frozen cuts (QC)", "tuned"))},
        ROOT / "figures/fig_ordinal_error_distribution.png")
    G.interval_figure(test, reg_test, A.apply_calibration(q_test, adaptive), adaptive_result["test"]["overall"],
                      ROOT / "figures/fig_interval_band.png", calibrated=True)
    importlib.import_module("05_seam_error").main()
    print("QC M5 refit and cached interval application complete. M6 remains historical.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=8)
    main(parser.parse_args().n_jobs)
