"""Apply frozen adaptive calibration to saved M5 tree percentiles.

Run: python src/05_apply_adaptive_interval.py
The feature table supplies targets; ordinal_probs_val_test.npz supplies row IDs.
No fitting, parameter selection, or model prediction takes place here.
"""

import importlib
import json
from pathlib import Path

import numpy as np

R = importlib.import_module("05_reg_to_cls")
I = importlib.import_module("05_interval_calibration")
A = importlib.import_module("05_interval_adaptive")
ROOT = Path(__file__).resolve().parents[1]


def main(root=ROOT):
    md = Path(root) / "models"
    metrics_path = md / "metrics_stage5_interval_adaptive.json"
    intervals_path = md / "stage5_intervals_adaptive.npz"
    for path in (metrics_path, intervals_path):
        if path.exists():
            raise FileExistsError(f"output already exists; archive it before running: {path}")

    calibration_path = md / "interval_calibration_adaptive.json"
    cal = json.loads(calibration_path.read_text(encoding="utf-8"))
    original = json.loads((md / "metrics_stage5.json").read_text(encoding="utf-8"))
    parts, _, hashes = R.load_data(Path(root))
    if hashes != original["input_sha256"] or hashes != cal["input_sha256"]:
        raise ValueError("feature/calibration/prediction provenance mismatch; supply matching stage-05 artifacts")
    if cal["n_calibration"] != len(parts["val"]):
        raise ValueError("calibration sample count differs from the current validation contract")
    if cal["method"] != "tree-spread-normalized absolute residual" or cal["unit"] != "ug/m3":
        raise ValueError("expected frozen adaptive calibration in ug/m3")
    if not 0 < cal["nominal_coverage"] < 1:
        raise ValueError("nominal coverage must be between 0 and 1")
    if not isinstance(cal["limitations"], list) or not all(
        isinstance(item, str) for item in cal["limitations"]
    ):
        raise ValueError("calibration limitations must be a list of strings")

    for name in ("stage5_predictions.npz", "ordinal_probs_val_test.npz"):
        if R.sha256(md / name) != original["artifacts"][name]:
            raise ValueError(f"saved artifact fingerprint mismatch: {name}")
    if R.sha256(md / "stage5_predictions.npz") != cal["predictions_sha256"]:
        raise ValueError("calibration belongs to a different prediction cache")

    result = {key: cal[key] for key in (
        "method", "nominal_coverage", "scale_multiplier", "scale_floor", "limitations", "unit"
    )}
    result.update(calibration_sha256=R.sha256(calibration_path),
                  new_model_fits=0, new_model_test_predictions=0)
    arrays = {}
    with np.load(md / "stage5_predictions.npz", allow_pickle=False) as saved, \
            np.load(md / "ordinal_probs_val_test.npz", allow_pickle=False) as rows:
        for split in ("val", "test"):
            frame = parts[split]
            ts = rows[f"ts_{split}"]
            nodes = rows[f"station_id_{split}"]
            if not np.array_equal(ts, frame.timestamp.to_numpy()) or not np.array_equal(
                nodes, frame.station_id.to_numpy(dtype=str)
            ):
                raise ValueError(f"{split} timestamp/station row identity mismatch")
            quantiles = saved[f"quantiles_{split}"]
            if quantiles.shape != (len(frame), 3):
                raise ValueError(f"{split} percentiles must have shape ({len(frame)}, 3)")
            band = A.apply_calibration(quantiles, cal)
            y = frame.target_pm25_3h.to_numpy()
            result[split] = {"overall": I.interval_scores(y, band), "by_node": {}}
            for node in sorted(frame.station_id.unique()):
                mask = frame.station_id.eq(node).to_numpy()
                result[split]["by_node"][node] = I.interval_scores(y[mask], band[mask])
            arrays[f"interval_{split}"] = band
            arrays[f"ts_{split}"] = ts
            arrays[f"station_id_{split}"] = nodes

    # Exclusive creation also rejects a file created after the initial check.
    with intervals_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    result["intervals_sha256"] = R.sha256(intervals_path)
    with metrics_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"val.n_samples={result['val']['overall']['n_samples']} / "
          f"test.n_samples={result['test']['overall']['n_samples']}")
    print(f"interval_val.shape={arrays['interval_val'].shape} / "
          f"interval_test.shape={arrays['interval_test'].shape}")
    for split in ("val", "test"):
        scores = result[split]["overall"]
        print(f"{split}: PICP={scores['PICP']:.8f}, MPIW={scores['MPIW']:.8f} ug/m3")
    return result


if __name__ == "__main__":
    main()
