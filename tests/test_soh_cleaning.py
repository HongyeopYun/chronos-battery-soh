import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.battery_preprocessing import make_frame, metadata, save_bundle
from src.soh_cleaning import clean_bundle, flag_capacity_outliers
from src.soh_data import load_bundle, observed_windows, sha256

ROOT = Path(__file__).resolve().parents[1]


def policy():
    return json.loads((ROOT / "configs/soh_outlier_cleaning.json").read_text(encoding="utf-8"))


def frame(values):
    return make_frame("toy", "cell1", np.arange(1, len(values) + 1), np.array(values) * 2., 2., "normal", segment_id="one")


class SOHCleaningTests(unittest.TestCase):
    def test_small_recovery_smooth_growth_and_fast_degradation_are_retained(self):
        for values in [np.linspace(1., 1.4, 200), np.linspace(1., .15, 200),
                       np.r_[np.linspace(1., .9, 40), .98, np.linspace(.9, .6, 40)]]:
            original = frame(values)
            flagged, removed = flag_capacity_outliers(original, policy())
            self.assertEqual(len(removed), 0)
            pd.testing.assert_frame_equal(flagged, original)

    def test_flagged_and_removed_spikes_cannot_enter_context_or_target_even_with_custom_flags(self):
        values = np.ones(200)
        values[99] = 1.4
        original = frame(values)
        flagged, removed = flag_capacity_outliers(original, policy())
        self.assertEqual(removed.cycle.tolist(), [100])
        self.assertEqual(flagged.loc[99, "quality_flags"], "capacity_outlier")
        np.testing.assert_array_equal(flagged.capacity_ah, original.capacity_ah)
        np.testing.assert_array_equal(flagged.soh, original.soh)
        cleaned = flagged[flagged.cycle != 100]
        self.assertEqual(len(observed_windows(original)["target"]), 121)
        for rows in [flagged, cleaned]:
            windows = observed_windows(rows, excluded_flags=[])
            self.assertEqual(len(windows["target"]), 41)
            self.assertFalse(np.any(windows["context"] == 1.4))
            self.assertFalse(np.any(windows["target"] == 1.4))
            self.assertFalse(np.any(windows["target_cycle"] == 100))

    def test_detection_is_causal_and_spikes_do_not_contaminate_later_baselines(self):
        values = np.ones(120)
        values[40:42] = [1.5, 1.6]
        prefix, removed = flag_capacity_outliers(frame(values[:60]), policy())
        longer = frame(np.r_[values, np.full(100, 2.)])
        flagged, _ = flag_capacity_outliers(longer, policy())
        pd.testing.assert_frame_equal(flagged.iloc[:60].reset_index(drop=True), prefix)
        self.assertEqual(removed.cycle.tolist(), [41, 42])
        np.testing.assert_allclose(removed.past_median_soh, 1.)

    def test_history_restarts_at_recording_protocol_and_cycle_gaps(self):
        for change in ["segment_id", "protocol_id", "cycle"]:
            data = frame(np.r_[np.ones(10), 1.5, np.full(10, 1.5)])
            if change == "cycle":
                data.loc[10:, "cycle"] += 1
            else:
                data.loc[10:, change] = "new"
            _, removed = flag_capacity_outliers(data, policy())
            self.assertEqual(len(removed), 0)
        with self.assertRaisesRegex(ValueError, "Minimum history"):
            flag_capacity_outliers(frame(np.ones(20)), dict(policy(), minimum_history=11))

    def test_clean_bundle_preserves_parent_reference_and_original_cycle_numbers(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "fake.mat"
            raw.write_bytes(b"test source")
            values = np.ones(200)
            values[99] = 1.4
            data = frame(values)
            hashes = {raw.name: sha256(raw)}
            battery = metadata("toy", "cell1", "batch", 2., 1, "first_cycle", [raw.name], hashes, "source ordinal")
            parent = root / "original"
            save_bundle("toy", data, pd.DataFrame([battery]), hashes, parent)
            before = {p.name: sha256(p) for p in parent.iterdir()}
            cleaned, removed, audit, manifest = clean_bundle(parent, root / "clean", policy())
            loaded, batteries, _ = load_bundle(root / "clean")
            self.assertEqual(manifest["n_removed_outliers"], 1)
            self.assertEqual(len(removed), 1)
            self.assertEqual(len(cleaned), 199)
            self.assertNotIn(100, loaded.cycle.tolist())
            self.assertEqual(loaded.cycle.iloc[-1], 200)
            self.assertEqual(batteries.reference_capacity_ah.iloc[0], 2.)
            self.assertEqual(audit.windows_before.iloc[0], 121)
            self.assertEqual(audit.windows_after.iloc[0], 41)
            self.assertEqual(before, {p.name: sha256(p) for p in parent.iterdir()})
            self.assertEqual((parent / "batteries.csv").read_bytes(), (root / "clean/batteries.csv").read_bytes())
            with self.assertRaisesRegex(ValueError, "separate"):
                clean_bundle(parent, parent, policy())


if __name__ == "__main__":
    unittest.main()
