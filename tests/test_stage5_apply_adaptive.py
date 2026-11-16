"""Application checks: frozen schema, row provenance, and overwrite refusal."""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
APP = importlib.import_module("05_apply_adaptive_interval")


class ApplyAdaptiveChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.md = self.root / "models"
        self.md.mkdir()
        self.parts = {}
        quantiles, row_ids = {}, {}
        for split, day in (("val", "2025-01-01"), ("test", "2025-02-01")):
            frame = pd.DataFrame({
                "timestamp": pd.to_datetime([day] * 3),
                "station_id": ["KL_City", "Port_Klang", "Sepang"],
                "target_pm25_3h": [1., 10., 40.],
            })
            self.parts[split] = frame
            quantiles[f"quantiles_{split}"] = np.array([[1., 2., 3.], [6., 10., 14.], [6., 10., 14.]])
            row_ids[f"ts_{split}"] = frame.timestamp.to_numpy()
            row_ids[f"station_id_{split}"] = frame.station_id.to_numpy(dtype=str)
        np.savez_compressed(self.md / "stage5_predictions.npz", **quantiles)
        np.savez_compressed(self.md / "ordinal_probs_val_test.npz", **row_ids)
        self.hashes = {"features_final": "feature-hash", "selected_features": "selection-hash"}
        self.cal = {"method": "tree-spread-normalized absolute residual",
                    "nominal_coverage": .9, "scale_multiplier": 2., "scale_floor": 1.,
                    "unit": "ug/m3", "n_calibration": 3,
                    "input_sha256": self.hashes, "limitations": ["Synthetic fixture."],
                    "predictions_sha256": APP.R.sha256(self.md / "stage5_predictions.npz")}
        self.write_calibration()
        original = {"input_sha256": self.hashes,
                    "artifacts": {name: APP.R.sha256(self.md / name) for name in (
                        "stage5_predictions.npz", "ordinal_probs_val_test.npz")}}
        (self.md / "metrics_stage5.json").write_text(json.dumps(original), encoding="utf-8")
        self.loader = patch.object(APP.R, "load_data", return_value=(self.parts, [], self.hashes))
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def write_calibration(self):
        (self.md / "interval_calibration_adaptive.json").write_text(json.dumps(self.cal), encoding="utf-8")

    def test_frozen_application_schema_and_no_refit(self):
        with patch.object(APP.A, "fit_calibration", side_effect=AssertionError("must not refit")):
            result = APP.main(self.root)
        loaded = json.loads((self.md / "metrics_stage5_interval_adaptive.json").read_text(encoding="utf-8"))
        self.assertEqual(result, loaded)
        self.assertEqual(set(result), {"method", "nominal_coverage", "scale_multiplier", "scale_floor",
            "calibration_sha256", "limitations", "new_model_fits", "new_model_test_predictions",
            "unit", "val", "test", "intervals_sha256"})
        self.assertEqual(result["new_model_fits"], 0)
        self.assertEqual(result["new_model_test_predictions"], 0)
        for split in ("val", "test"):
            self.assertEqual(set(result[split]), {"overall", "by_node"})
            self.assertEqual(set(result[split]["by_node"]), {"KL_City", "Port_Klang", "Sepang"})
            self.assertEqual(set(result[split]["overall"]), {"PICP", "MPIW", "n_samples"})
            self.assertAlmostEqual(result[split]["overall"]["PICP"], 2 / 3)
            self.assertAlmostEqual(result[split]["overall"]["MPIW"], 12.)
        with np.load(self.md / "stage5_intervals_adaptive.npz", allow_pickle=False) as saved:
            self.assertEqual(set(saved.files), {f"{key}_{split}" for split in ("val", "test")
                                               for key in ("interval", "ts", "station_id")})
            for split in ("val", "test"):
                np.testing.assert_array_equal(saved[f"interval_{split}"], [[0., 2., 4.], [2., 10., 18.], [2., 10., 18.]])
                self.assertEqual(saved[f"interval_{split}"].shape, (3, 3))
                self.assertEqual(saved[f"ts_{split}"].shape, (3,))
                self.assertEqual(saved[f"station_id_{split}"].shape, (3,))

    def test_each_existing_output_is_preserved(self):
        for name in ("metrics_stage5_interval_adaptive.json", "stage5_intervals_adaptive.npz"):
            with self.subTest(name=name):
                path = self.md / name
                path.write_bytes(b"keep existing result")
                with self.assertRaises(FileExistsError):
                    APP.main(self.root)
                self.assertEqual(path.read_bytes(), b"keep existing result")
                path.unlink()

    def test_mismatched_inputs_fail_before_writing(self):
        mutations = (
            ("input_sha256", {"features_final": "old", "selected_features": "old"}, "provenance"),
            ("n_calibration", 16656, "sample count"),
            ("predictions_sha256", "old-cache", "different prediction cache"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key):
                previous = self.cal[key]
                self.cal[key] = value
                self.write_calibration()
                with self.assertRaisesRegex(ValueError, message):
                    APP.main(self.root)
                self.assertFalse((self.md / "stage5_intervals_adaptive.npz").exists())
                self.cal[key] = previous
        self.write_calibration()
        self.parts["test"].loc[0, "timestamp"] += pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError, "test timestamp/station"):
            APP.main(self.root)
        self.assertFalse((self.md / "metrics_stage5_interval_adaptive.json").exists())


if __name__ == "__main__":
    unittest.main()
