import unittest

import numpy as np
import pandas as pd

from src.nasa_covariates import fixed_time_mean, covariate_inputs


class NasaCovariateTests(unittest.TestCase):
    def test_fixed_interval_ignores_later_discharge_and_sample_density(self):
        # Integral of a linear trace is independent of uneven sample spacing.
        self.assertAlmostEqual(fixed_time_mean([10, 110, 710, 1010], [0, 1, 7, 999]), 3.)
        self.assertAlmostEqual(fixed_time_mean([0, 300, 600, 900], [0, 3, 6, -999]), 3.)
        with self.assertRaises(ValueError):
            fixed_time_mean([0, 500], [1, 2])
        with self.assertRaises(ValueError):
            fixed_time_mean([0, 0, 700], [1, 2, 3])

    def test_future_sensor_and_target_values_never_enter_inputs(self):
        w = {"context": np.array([[1., .9]]), "context_cycle": np.array([[1, 2]]),
             "forecast_start": np.array([3]), "target": np.array([[.8, .7]])}
        f = pd.DataFrame({"cycle": [1, 2, 3, 4], "v": [4., 3.9, 1000., -1000.]})
        before = covariate_inputs(w, f, ["v"])[0]
        f.loc[f.cycle >= 3, "v"] = np.nan
        w["target"][:] = -999
        after = covariate_inputs(w, f, ["v"])[0]
        np.testing.assert_array_equal(before["target"], after["target"])
        np.testing.assert_array_equal(before["past_covariates"]["v"], after["past_covariates"]["v"])
        self.assertNotIn("future_covariates", after)
        w["context_cycle"][0, 1] = 3
        with self.assertRaises(ValueError):
            covariate_inputs(w, f, ["v"])


if __name__ == "__main__":
    unittest.main()
