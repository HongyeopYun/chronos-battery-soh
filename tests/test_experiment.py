import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.soh_experiment import (ExperimentConfig, find_project_root, load_config, load_soh,
                                make_windows, point_metrics, summarize_samples, temporal_split)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig()

    def test_window_indices_are_causal(self):
        series = np.arange(168, dtype=float)
        windows = make_windows(series, 64, 16)
        self.assertEqual(windows["target"].shape, (89, 16))
        for x, y, start in zip(windows["context"], windows["target"], windows["forecast_start"]):
            self.assertEqual(x[-1] + 1, y[0])
            self.assertEqual(start, y[0] + 1)
        self.assertEqual(windows["target"][-1, -1], 167)

    def test_validation_targets_never_appear_in_training(self):
        for length in (168, 132):
            series = np.arange(length, dtype=float)
            train, valid, cutoff = temporal_split(series, self.config)
            seen = np.concatenate([train["context"].ravel(), train["target"].ravel()])
            self.assertLess(seen.max(), cutoff)
            self.assertGreaterEqual(valid["target"].min(), cutoff)
            self.assertEqual(np.intersect1d(seen, valid["target"]).size, 0)
            # Previously observed training values are legitimate validation context.
            self.assertLess(valid["context"][0, 0], cutoff)
            self.assertEqual(valid["forecast_start"][0], cutoff + 1)

    def test_median_interpolates_even_sample_count(self):
        samples = np.array([[[0.0, 4.0], [2.0, 8.0]]])
        summary = summarize_samples(samples)
        np.testing.assert_array_equal(summary["median"], [[1.0, 6.0]])
        self.assertTrue((summary["q10"] <= summary["median"]).all())
        self.assertTrue((summary["median"] <= summary["q90"]).all())

    def test_global_rmse_is_not_mean_window_rmse(self):
        target = np.zeros((2, 2))
        pred = np.array([[0, 0], [3, 4]])
        self.assertAlmostEqual(point_metrics(target, pred)["rmse"], 2.5)
        self.assertAlmostEqual(point_metrics(target, pred)["mae"], 1.75)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            point_metrics(np.zeros((2, 2)), np.zeros((2, 1)))
        with self.assertRaises(ValueError):
            summarize_samples(np.full((1, 2, 3), np.nan))
        with self.assertRaises(ValueError):
            temporal_split(np.arange(80), self.config)
        with self.assertRaises(ValueError):
            replace(self.config, inference_seeds=(42, 42))

    def test_real_data_splits_and_project_root(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(find_project_root(root), root)
        self.assertEqual(find_project_root(root / "notebooks"), root)
        config = load_config(root)
        if not (root / "data/raw/B0005.mat").exists():
            self.skipTest("NASA MAT files are not present")
        series = load_soh(root, config)
        for b, values in series.items():
            expected = 53 if b == "B0018" else 89
            self.assertEqual(len(make_windows(values, 64, 16)["target"]), expected)
            temporal_split(values, config)


if __name__ == "__main__":
    unittest.main()
