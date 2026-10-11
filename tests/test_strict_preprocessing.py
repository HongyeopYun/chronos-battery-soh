import unittest

import numpy as np
import pandas as pd

from src.soh_data import observed_windows
from src.strict_preprocessing import apply_quality, classify_trace, trace_features


def complete_feature():
    return dict(n_samples=200, trace_missing=False, trace_nonfinite=False,
                trace_length_mismatch=False, discharge_samples=100,
                discharge_time_backsteps=0, discharge_counter_backsteps=0,
                voltage_min_v=2.7, voltage_max_v=4.2, discharge_min_voltage_v=2.7,
                discharge_end_voltage_v=2.7, discharge_cc_current_a=-1.1,
                discharge_max_gap_s=30, discharge_median_gap_s=30,
                charge_samples=100, charge_end_voltage_v=4.2, charge_end_current_a=.05,
                charge_cv_step_present=True, discharge_integral_ah=1.,
                discharge_counter_ah=1., raw_capacity_ah=1.)


def soh_rows(values):
    return pd.DataFrame({"dataset": "calce", "battery_id": "test", "cycle": np.arange(1, len(values) + 1),
                         "capacity_ah": values, "soh": values, "capacity_kind": "full_discharge",
                         "is_observed": True, "quality_flags": "", "protocol_id": "test", "cycle_description": ""})


class StrictPreprocessingTests(unittest.TestCase):
    def test_normal_cycle_and_true_recovery_are_preserved(self):
        data = soh_rows([1., .96, 1.01, .97])
        features = pd.DataFrame([dict(complete_feature(), battery_id="test", cycle=k + 1,
                                    raw_capacity_ah=q, discharge_integral_ah=q, discharge_counter_ah=q)
                                 for k, q in enumerate(data.soh)])
        cleaned = apply_quality(data, features)
        self.assertTrue(cleaned.quality_eligible.all())
        self.assertTrue(cleaned.shape_review.iloc[2])
        np.testing.assert_array_equal(cleaned.soh, data.soh)

    def test_missing_cv_not_hidden_by_normal_voltage_cutoffs(self):
        f = dict(complete_feature(), charge_cv_step_present=False, charge_end_current_a=.55)
        reasons, _ = classify_trace("calce", f, 22)
        self.assertIn("missing_cv_charge_step", reasons)
        self.assertIn("charge_termination_not_met", reasons)
        self.assertNotIn("discharge_cutoff_not_reached", reasons)

    def test_incomplete_discharge_and_counter_mismatch_are_distinct(self):
        f = dict(complete_feature(), discharge_min_voltage_v=3., discharge_end_voltage_v=3.,
                 discharge_counter_ah=.6)
        reasons, _ = classify_trace("calce", f, 3)
        self.assertIn("discharge_cutoff_not_reached", reasons)
        self.assertIn("summary_counter_mismatch", reasons)

    def test_time_resets_between_steps_are_allowed_but_not_during_discharge(self):
        f = trace_features([.5, .05, 0, -1, -1, -1, 0], [4, 4.2, 4.1, 4, 3.5, 2.7, 3],
                           [0, 100, 0, 0, 1800, 3600, 0])
        self.assertEqual(f["discharge_time_backsteps"], 0)
        self.assertAlmostEqual(f["discharge_integral_ah"], 1.)
        f = trace_features([.5, .05, 0, -1, -1, -1, 0], [4, 4.2, 4.1, 4, 3.5, 2.7, 3],
                           [0, 100, 0, 0, 1800, 1000, 0])
        self.assertEqual(f["discharge_time_backsteps"], 1)

    def test_nasa_integral_stops_at_capacity_definition_not_lower_cutoff(self):
        f = trace_features([0, -2, -2, -2, -2, 0], [4.2, 4, 3, 2.7, 2.3, 3],
                           [0, 10, 910, 1810, 1910, 1920], cutoff=2.7, charge=([1.5, .02], [4., 4.2]))
        self.assertAlmostEqual(f["discharge_integral_ah"], 1.)
        self.assertAlmostEqual(f["discharge_end_voltage_v"], 2.3)

    def test_calce_first_unsampled_cc_interval_is_accounted_for(self):
        f = trace_features([.05, 0, -1, -1, -1], [4.2, 4.1, 4., 3., 2.7],
                           [0, 10, 40, 1840, 3610], first_step_seconds=[0, 10, 30, 1830, 3600],
                           counter=[12, 12, 12 + 30 / 3600, 12 + 1830 / 3600, 13])
        self.assertAlmostEqual(f["discharge_integral_ah"], 1.)
        self.assertAlmostEqual(f["discharge_counter_ah"], 1.)

    def test_quality_mask_cannot_be_bypassed_by_custom_flags(self):
        data = soh_rows(np.ones(90))
        data["quality_eligible"] = True
        self.assertEqual(len(observed_windows(data, excluded_flags=[])["target"]), 11)
        data.loc[45, "quality_eligible"] = False
        self.assertEqual(len(observed_windows(data, excluded_flags=[])["target"]), 0)
        # Removing the row cannot compress time either.
        self.assertEqual(len(observed_windows(data.drop(index=45), excluded_flags=[])["target"]), 0)
        data["quality_eligible"] = True
        data.loc[45, "quality_flags"] = "strict_quality_excluded"
        self.assertEqual(len(observed_windows(data, excluded_flags=[])["target"]), 0)

    def test_string_false_is_not_silently_true(self):
        data = soh_rows(np.ones(90))
        data["quality_eligible"] = "False"
        with self.assertRaisesRegex(ValueError, "booleans"):
            observed_windows(data)

    def test_future_soh_does_not_change_past_quality_decisions(self):
        data = soh_rows([1., .9, 1.])
        features = pd.DataFrame([dict(complete_feature(), battery_id="test", cycle=k + 1,
                                    raw_capacity_ah=q, discharge_integral_ah=q, discharge_counter_ah=q)
                                 for k, q in enumerate(data.soh)])
        before = apply_quality(data, features)
        data.loc[2, ["soh", "capacity_ah"]] = 100.
        after = apply_quality(data, features)
        pd.testing.assert_frame_equal(before.iloc[:2], after.iloc[:2])

    def test_missing_raw_cycle_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            apply_quality(soh_rows([1., 1.]), pd.DataFrame([dict(complete_feature(), battery_id="test", cycle=1)]))

    def test_integral_disagreement_is_quarantined_not_declared_a_capacity_error(self):
        data = soh_rows([1.])
        features = pd.DataFrame([dict(complete_feature(), battery_id="test", cycle=1, discharge_integral_ah=1.06)])
        cleaned = apply_quality(data, features)
        self.assertEqual(cleaned.quality_status.iloc[0], "quarantined")
        self.assertFalse(cleaned.quality_eligible.iloc[0])
        self.assertEqual(cleaned.capacity_ah.iloc[0], 1.)

    def test_nasa_missing_charge_log_is_review_only_when_discharge_checks_pass(self):
        f = dict(complete_feature(), charge_samples=0, charge_end_voltage_v=np.nan,
                 charge_end_current_a=np.nan, initial_discharge_voltage_v=4.22,
                 capacity_cutoff_reached=True, discharge_cc_current_a=-2.)
        bad, review = classify_trace("nasa", f, 90)
        self.assertEqual(bad, [])
        self.assertIn("preceding_charge_unrecorded_native_capacity_retained", review)
        for update in ({"discharge_integral_ah": 1.2}, {"initial_discharge_voltage_v": 3.8},
                       {"initial_discharge_voltage_v": np.nan}, {"capacity_cutoff_reached": False},
                       {"discharge_time_backsteps": 1}, {"raw_capacity_ah": np.nan}):
            bad, _ = classify_trace("nasa", {**f, **update}, 90)
            self.assertIn("preceding_charge_unverified", bad)
        bad, _ = classify_trace("calce", {**f, "discharge_cc_current_a": -1.1}, 90)
        self.assertIn("preceding_charge_unverified", bad)

    def test_nasa_known_incomplete_charge_still_excluded(self):
        f = dict(complete_feature(), discharge_cc_current_a=-2., charge_end_current_a=.7,
                 initial_discharge_voltage_v=4.22, capacity_cutoff_reached=True)
        bad, _ = classify_trace("nasa", f, 90)
        self.assertIn("charge_termination_not_met", bad)

    def test_nasa_retained_review_does_not_remove_65_valid_one_step_targets(self):
        data = soh_rows(np.linspace(1., .8, 168))
        data['dataset'] = 'nasa'
        features = []
        for k, q in enumerate(data.soh, 1):
            f = dict(complete_feature(), battery_id='test', cycle=k, discharge_cc_current_a=-2.,
                     raw_capacity_ah=q, discharge_integral_ah=q, discharge_counter_ah=q,
                     initial_discharge_voltage_v=4.22, capacity_cutoff_reached=True)
            if k == 90:
                f.update(charge_samples=0, charge_end_voltage_v=np.nan, charge_end_current_a=np.nan)
            features.append(f)
        revised = apply_quality(data, pd.DataFrame(features))
        self.assertTrue(revised.quality_eligible.all())
        self.assertEqual(revised.iloc[89].quality_status, 'accepted_with_note')
        np.testing.assert_array_equal(observed_windows(revised, 64, 1)['forecast_start'], np.arange(65,169))
        np.testing.assert_array_equal(revised.soh, data.soh)


if __name__ == "__main__":
    unittest.main()
