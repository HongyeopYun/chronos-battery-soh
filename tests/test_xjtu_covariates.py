import unittest

import numpy as np

from src.xjtu_covariates import discharge_features
from src.nasa_covariates import FEATURES


class XjtuCovariateTests(unittest.TestCase):
    def test_discharge_only_and_minutes_to_seconds(self):
        # Charging, resting and later discharge measurements must not change the first 10-min mean.
        record = {"relative_time_min": [0, 10, 0, 0, 5, 10, 20, 0],
                  "voltage_V": [999, 999, 999, 4., 3.9, 3.8, 999, 999],
                  "current_A": [4., 4., 0., -2., -2., -2., -2., 0.]}
        scores = discharge_features(record)
        self.assertAlmostEqual(scores[FEATURES[0]], 3.9)
        self.assertAlmostEqual(scores[FEATURES[1]], -2.)
        record["voltage_V"][0] = -999
        self.assertEqual(scores, discharge_features(record))

    def test_ambiguous_discharge_and_short_interval_fail(self):
        record = {"relative_time_min": [0, 10, 0, 0, 10],
                  "voltage_V": np.ones(5), "current_A": [-2., -2., 0., -2., -2.]}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            discharge_features(record)
        record = {"relative_time_min": [0, 5], "voltage_V": [4., 3.8], "current_A": [-2., -2.]}
        with self.assertRaisesRegex(ValueError, "fixed feature interval"):
            discharge_features(record)

    def test_repeated_time_keeps_last_sample_but_backward_time_fails(self):
        record = {"relative_time_min": [0, 0, 5, 10], "voltage_V": [999., 4., 3.9, 3.8],
                  "current_A": [-2., -2., -2., -2.]}
        result = discharge_features(record)
        self.assertAlmostEqual(result[FEATURES[0]], 3.9)
        self.assertEqual(result["duplicate_discharge_timestamps"], 1)
        record["relative_time_min"] = [0, 5, 3, 10]
        with self.assertRaisesRegex(ValueError, "backward"):
            discharge_features(record)


if __name__ == "__main__":
    unittest.main()
