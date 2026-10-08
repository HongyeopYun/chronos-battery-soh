import unittest
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.soh_evaluation import r2_score, point_scores, distribution_r2, _r2_from_rmse, mean_window_r2, window_r2_values
from src.soh_evaluation import export_run
from src.soh_experiment import ExperimentConfig, make_windows, summarize_samples, metric_rows, save_forecast, _write_tables


class R2Tests(unittest.TestCase):
    def test_perfect_mean_and_negative_scores(self):
        y = np.array([[1., 2.], [3., 4.]])
        self.assertEqual(r2_score(y, y), 1.)
        self.assertEqual(r2_score(y, np.broadcast_to(y.mean(axis=1,keepdims=True),y.shape)), 0.)
        self.assertLess(r2_score(y, y + 10), 0.)

    def test_r2_score_averages_windows_instead_of_flattening_targets(self):
        y = np.array([[0., 10.], [1., 20.]])
        p = y + 1
        expected = np.mean([r2_score(a,b) for a,b in zip(y,p)])
        self.assertAlmostEqual(r2_score(y, p), expected)
        self.assertNotAlmostEqual(r2_score(y, p), 1 - 4 / np.square(y - y.mean()).sum())

    def test_undefined_constant_or_single_targets_and_invalid_inputs(self):
        self.assertTrue(np.isnan(r2_score([1., 1.], [1., 1.])))
        self.assertTrue(np.isnan(r2_score([1.], [2.])))
        for y, p in [([], []), ([1, 2], [1]), ([1, np.nan], [1, 2])]:
            with self.assertRaises(ValueError):
                r2_score(y, p)

    def test_numeric_precision_and_rmse_identity(self):
        y = np.array([[1, 2, 4], [3, 4, 8]], dtype=np.float32)
        p = y + np.array([[.1, .2, .3], [.4, .5, .6]])
        scores = point_scores(y, p)
        self.assertAlmostEqual(scores["r2"], np.mean([r2_score(a,b) for a,b in zip(y,p)]))
        self.assertTrue(np.isnan(_r2_from_rmse(y, scores["rmse"])))
        self.assertNotAlmostEqual(scores["r2"], 1-np.square(y-p).sum()/np.square(y-y.mean()).sum())

    def test_mean_window_r2_equal_weights_levels_and_undefined_windows(self):
        y = np.array([[0.,1.,2.],[10.,12.,14.],[2.,2.,2.]])
        p = y + .5
        self.assertAlmostEqual(mean_window_r2(y,p), (.625+.90625)/2)
        self.assertTrue(np.isnan(window_r2_values(y,p)[2]))
        self.assertAlmostEqual(mean_window_r2(y+np.array([[100.],[500.],[0.]]),p+np.array([[100.],[500.],[0.]])), (.625+.90625)/2)
        self.assertTrue(np.isnan(mean_window_r2(y[:,:1],p[:,:1])))

    def test_mean_and_median_are_scored_separately(self):
        y = np.array([[1., 2.]])
        samples = np.array([[[1., 2.], [1., 2.], [7., 8.]]])
        scores = distribution_r2(y, samples)
        self.assertEqual(scores["median_r2"], 1.)
        self.assertLess(scores["mean_r2"], 0.)

    def test_export_preserves_raw_files_and_averages_batteries_not_points(self):
        cfg = ExperimentConfig(battery_ids=("A", "B", "C"), context_length=2, prediction_length=2,
                               inference_seeds=(42,))
        series = {"A": np.arange(6.), "B": np.arange(8.), "C": np.arange(10.)}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "results/runs/fixed"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({"status": "complete", "kind": "zeroshot", "config": asdict(cfg)}))
            rows, horizons, windows, expected, targets, predictions = [], [], [], [], [], []
            for index, (battery, values) in enumerate(series.items()):
                w = make_windows(values, 2, 2)
                samples = np.repeat((w["target"] + index)[:, None], 3, axis=1)
                save_forecast(run / "predictions" / f"{battery}_infer42.npz", w, samples)
                summary = summarize_samples(samples)
                r, h, v = metric_rows(w, summary, "ZeroShot", battery, inference_seed=42)
                rows.extend(r); horizons.extend(h); windows.extend(v)
                expected.append(mean_window_r2(w["target"], summary["median"]))
                targets.extend(w["target"].ravel()); predictions.extend(summary["median"].ravel())
            _write_tables(run, rows, horizons, windows)
            originals = {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
            with patch("src.soh_evaluation.verify_saved_run"), patch("src.soh_evaluation.load_soh", return_value=series):
                out = export_run(root, run)
            result = pd.read_csv(out / "summary.csv").query("estimator == 'median'").iloc[0]
            self.assertAlmostEqual(result.r2_mean, np.mean(expected))
            self.assertNotAlmostEqual(result.r2_mean, r2_score(targets, predictions))
            for filename in ("metrics.csv", "horizon_metrics.csv", "window_metrics.csv"):
                self.assertIn("r2", pd.read_csv(out / filename))
            self.assertEqual(originals, {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
