import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.calce_windows import observation_segments
from src.soh_data import observed_windows


class CalceWindowTests(unittest.TestCase):
    def rows(self):
        starts = pd.date_range("2010-01-01", periods=90, freq="6h")
        return pd.DataFrame({"dataset": "calce", "battery_id": "CS2_35",
            "cycle": np.arange(1, 91), "source_cycle": np.tile(np.arange(1, 46), 2),
            "segment_id": ["first"] * 45 + ["second"] * 45,
            "protocol_id": "same", "source_start_time": starts,
            "source_end_time": starts + pd.Timedelta(hours=2),
            "soh": np.linspace(1., .9, 90), "capacity_ah": 1.,
            "is_observed": True, "quality_flags": "", "capacity_kind": "full_discharge"})

    def test_files_can_join_but_invalid_measurements_do_not(self):
        rows = self.rows()
        self.assertEqual(len(observed_windows(rows)["target"]), 0)
        rows["segment_id"] = observation_segments(rows)
        self.assertEqual(len(observed_windows(rows)["target"]), 11)
        rows.loc[44, "capacity_kind"] = "partial_discharge"
        self.assertEqual(len(observed_windows(rows)["target"]), 0)

    def test_source_gaps_and_protocol_changes_still_break(self):
        for field, value in [("source_cycle", 47), ("protocol_id", "different")]:
            rows = self.rows()
            rows.loc[44, field] = value
            rows["segment_id"] = observation_segments(rows)
            self.assertEqual(len(observed_windows(rows)["target"]), 0)
        rows = self.rows().drop(index=44)
        rows["segment_id"] = observation_segments(rows)
        self.assertEqual(len(observed_windows(rows)["target"]), 0)

    def test_preprocessed_observation_segments_can_be_applied_twice(self):
        rows = self.rows()
        rows["source_segment_id"] = rows.segment_id
        rows["segment_id"] = observation_segments(rows)
        first = rows.segment_id.copy()
        rows["segment_id"] = observation_segments(rows)
        np.testing.assert_array_equal(first, rows.segment_id)
        self.assertEqual(len(observed_windows(rows)["target"]), 11)

    def test_overlapping_chronology_is_rejected(self):
        rows = self.rows()
        rows.loc[45, "source_start_time"] = rows.loc[44, "source_start_time"]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            observation_segments(rows)

    def test_config_controls_join_without_modifying_source_bundle(self):
        from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows
        from src.soh_data import sha256, write_json
        from test_battery_zeroshot import save_bundle
        with TemporaryDirectory() as temporary:
            root = save_bundle(Path(temporary), sizes=(90,), segment=46)
            rows = pd.read_csv(root / "soh.csv", keep_default_na=False)
            rows["dataset"] = "calce"
            for field in ["source_cycle", "source_start_time", "source_end_time"]:
                rows[field] = self.rows()[field].to_numpy()
            rows.to_csv(root / "soh.csv", index=False, float_format="%.17g")
            batteries = pd.read_csv(root / "batteries.csv", keep_default_na=False)
            batteries["dataset"] = "calce"
            batteries.to_csv(root / "batteries.csv", index=False)
            hashes = {name: sha256(root / name) for name in ["soh.csv", "batteries.csv", "schema.json"]}
            write_json(root / "manifest.json", {"schema_version": 1, "status": "complete",
                "source_files": {}, "artifact_sha256": hashes})
            config = ZeroShotConfig()
            self.assertEqual(prepare_battery_windows(root, config)[1].evaluated_windows.sum(), 0)
            records, coverage, view, _ = prepare_battery_windows(
                root, replace(config, calce_cross_file_windows=True))
            self.assertEqual(coverage.evaluated_windows.sum(), 11)
            self.assertEqual(records[0]["windows"]["context"].shape, (11, 64))
            self.assertEqual(view.source_segment_id.nunique(), 2)
            self.assertEqual(hashes, {name: sha256(root / name) for name in hashes})


if __name__ == "__main__":
    unittest.main()
