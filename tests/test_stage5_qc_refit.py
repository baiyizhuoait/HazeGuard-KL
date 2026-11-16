"""Check that cached historical probabilities cannot be silently misaligned."""
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
Q = importlib.import_module("05_refit_m5_qc")


class QcRefitChecks(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01", "2025-01-01"]),
                                   "station_id": ["KL_City", "Sepang"], "target_level_3h": [1, 2]})
        self.saved = {"ts_val": pd.to_datetime(["2025-01-01"] * 3).to_numpy(),
                      "station_id_val": np.array(["Sepang", "Port_Klang", "KL_City"]),
                      "y_val": np.array([2, 0, 1])}

    def test_join_is_by_both_keys_not_position(self):
        np.testing.assert_array_equal(Q.align_legacy_rows(self.frame, self.saved, "val"), [2, 0])

    def test_missing_duplicate_or_changed_targets_rejected(self):
        for bad in ("missing", "duplicate", "target"):
            with self.subTest(bad=bad):
                frame = self.frame.copy()
                if bad == "missing":
                    frame.loc[0, "timestamp"] += pd.Timedelta(hours=1)
                elif bad == "duplicate":
                    frame.loc[1] = frame.loc[0]
                else:
                    frame.loc[0, "target_level_3h"] = 3
                with self.assertRaises(ValueError):
                    Q.align_legacy_rows(frame, self.saved, "val")

    def test_existing_refit_record_blocks_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "models").mkdir()
            (root / "models/stage5_qc_refit_plan.json").write_text("{}", encoding="utf-8")
            with patch.object(Q, "ROOT", root), patch.object(Q.R, "load_data", side_effect=AssertionError("too late")):
                with self.assertRaises(FileExistsError):
                    Q.main()


if __name__ == "__main__":
    unittest.main()
