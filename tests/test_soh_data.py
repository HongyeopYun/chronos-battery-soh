import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from scipy.io import savemat

from src.soh_data import load_bundle, observed_windows, sha256, validate_tables
from src.xjtu_preprocessing import INITIAL_DESCRIPTION, NORMAL_DESCRIPTION, extract_battery, preprocess

ROOT = Path(__file__).resolve().parents[1]


def policy():
    return json.loads((ROOT / "configs/xjtu_2c_preprocess.json").read_text(encoding="utf-8"))


def save_source(path, q, descriptions=None, cycle_life=None):
    descriptions = descriptions or [INITIAL_DESCRIPTION] + [NORMAL_DESCRIPTION] * (len(q)-1)
    savemat(path, {"summary": {"discharge_capacity_Ah": np.array(q), "charge_capacity_Ah": np.full(len(q), 999.),
                              "cycle_life": len(q) if cycle_life is None else cycle_life,
                              "description": "2C charge experiment"},
                   "data": np.array([{"description": d} for d in descriptions], dtype=object)})


class CommonSOHTests(unittest.TestCase):
    def test_extracts_discharge_and_keeps_initial_reference_and_recovery(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "2C_battery-1.mat"
            save_source(source, [2., 1.9, 2.01, 1.8])
            original = source.read_bytes()
            data, battery, audit = extract_battery(source, policy())
            np.testing.assert_allclose(data.capacity_ah, [2., 1.9, 2.01, 1.8])
            np.testing.assert_allclose(data.soh, [1., .95, 1.005, .9])
            self.assertEqual(data.cycle.tolist(), [1, 2, 3, 4])
            self.assertEqual(battery["reference_cycle"], 1)
            self.assertEqual(data.iloc[0].quality_flags, "initial_capacity_test")
            self.assertIn("capacity_increase", data.iloc[2].quality_flags)
            self.assertIn("soh_above_one", data.iloc[2].quality_flags)
            self.assertIn("protocol_transition", data.iloc[1].quality_flags)
            self.assertNotIn("large_soh_change", data.iloc[1].quality_flags)
            self.assertIn("large_soh_change", data.iloc[2].quality_flags)
            self.assertEqual(audit["n_large_soh_changes"], 2)
            self.assertEqual(audit["status"], "passed_with_notes")
            self.assertFalse(audit["prediction_ready"])
            self.assertEqual(source.read_bytes(), original)

    def test_invalid_observations_are_retained_and_flagged(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "2C_battery-1.mat"
            save_source(source, [2., np.nan, 0., -1., 1.8])
            data, _, audit = extract_battery(source, policy())
            self.assertEqual(len(data), 5)
            self.assertIn("nonfinite_capacity", data.iloc[1].quality_flags)
            self.assertIn("nonpositive_capacity", data.iloc[2].quality_flags)
            self.assertEqual(audit["status"], "failed")
            self.assertEqual(len(observed_windows(data, 1, 1)["target"]), 0)

    def test_invalid_reference_and_mismatched_raw_cycles_fail(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "2C_battery-1.mat"
            save_source(source, [0., 1.8])
            with self.assertRaisesRegex(ValueError, "reference capacity"):
                extract_battery(source, policy())
            save_source(source, [2., 1.8], descriptions=[INITIAL_DESCRIPTION])
            with self.assertRaisesRegex(ValueError, "counts do not match"):
                extract_battery(source, policy())

    def test_unexpected_protocol_is_not_called_full_discharge(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "2C_battery-1.mat"
            save_source(source, [2., .7], descriptions=[INITIAL_DESCRIPTION, "partial discharge"])
            data, _, audit = extract_battery(source, policy())
            self.assertEqual(data.iloc[1].capacity_kind, "unknown")
            self.assertIn("unexpected_protocol", data.iloc[1].quality_flags)
            self.assertEqual(audit["status"], "failed")

    def test_cycle_life_is_metadata_not_a_truncation_rule(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "2C_battery-1.mat"
            save_source(source, [2., 1.9, 1.8, 1.7], cycle_life=2)
            data, battery, audit = extract_battery(source, policy())
            self.assertEqual(len(data), 4)
            self.assertEqual(battery["source_summary_cycle_life"], 2)
            self.assertIn("no_truncation", audit["notes"])

    def test_forecast_cycles_preserve_gaps_and_exclude_initial_test(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "2C_battery-1.mat"
            save_source(source, np.linspace(2., 1.6, 200))
            data, _, _ = extract_battery(source, policy())
            windows = observed_windows(data)
            self.assertEqual(len(windows["target"]), 120)
            self.assertEqual(windows["forecast_start"][0], 66)
            np.testing.assert_array_equal(windows["target_cycle"][0], np.arange(66, 82))
            gapped = observed_windows(data[data.cycle != 100])
            self.assertEqual(len(gapped["target"]), 40)
            self.assertTrue(np.all(np.diff(gapped["target_cycle"], axis=1) == 1))
            self.assertFalse(np.any(gapped["target_cycle"] == 100))

    def test_interpolated_and_partial_observations_cannot_enter_windows(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "2C_battery-1.mat"
            save_source(source, np.linspace(2., 1.6, 90))
            data, _, _ = extract_battery(source, policy())
            self.assertEqual(len(observed_windows(data)["target"]), 10)
            data.loc[data.cycle == 50, "is_observed"] = False
            self.assertEqual(len(observed_windows(data)["target"]), 0)
            data.loc[data.cycle == 50, "is_observed"] = True
            data.loc[data.cycle == 50, "capacity_kind"] = "partial_discharge"
            self.assertEqual(len(observed_windows(data)["target"]), 0)

    def test_large_changes_are_review_flags_and_not_forecast_exclusions(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "2C_battery-1.mat"
            q = np.linspace(2., 1.6, 90)
            q[49] += .1
            save_source(source, q)
            data, _, audit = extract_battery(source, policy())
            self.assertEqual(audit["n_large_soh_changes"], 2)
            self.assertEqual(audit["status"], "passed_with_notes")
            self.assertEqual(len(observed_windows(data)["target"]), 10)

    def test_schema_rejects_duplicate_identity_and_changed_normalization(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "2C_battery-1.mat"
            save_source(source, [2., 1.9, 1.8])
            data, battery, _ = extract_battery(source, policy())
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_tables(pd.concat([data, data.iloc[:1]]), pd.DataFrame([battery]))
            changed = data.copy()
            changed.loc[1, "soh"] = 1.
            with self.assertRaises(AssertionError):
                validate_tables(changed, pd.DataFrame([battery]))

    def test_bundle_round_trip_checksums_and_failure_gate(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            for i in range(1, 9):
                save_source(raw / f"2C_battery-{i}.mat", np.linspace(2. + i/100, 1.6, 90))
            hashes = {p.name: sha256(p) for p in raw.iterdir()}
            data, _, audit, manifest = preprocess(raw, root / "processed", policy())
            loaded, batteries, _ = load_bundle(root / "processed")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(data), 720)
            self.assertEqual(len(batteries), 8)
            self.assertEqual(int(audit.n_prediction_windows.sum()), 80)
            np.testing.assert_allclose(loaded.soh, data.soh, rtol=1e-12)
            self.assertEqual(hashes, {p.name: sha256(p) for p in raw.iterdir()})
            with (root / "processed/soh.csv").open("a", encoding="utf-8") as stream:
                stream.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                load_bundle(root / "processed")
            save_source(raw / "2C_battery-1.mat", [0., 1.8])
            _, _, failed_audit, failed_manifest = preprocess(raw, root / "failed", policy())
            self.assertEqual(failed_manifest["status"], "audit_failed")
            self.assertEqual(int((failed_audit.status == "failed").sum()), 1)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_bundle(root / "failed")


if __name__ == "__main__":
    unittest.main()
