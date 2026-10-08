from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.battery_zeroshot import ZeroShotConfig, battery_zeroshot, prepare_battery_windows, verify_battery_zeroshot
from src.battery_zeroshot_minimal import compact_scores, compact_point_scores, refresh_compact_r2, R2_DEFINITION
from test_battery_zeroshot import save_bundle, fake_predict


def fake_figures(directory, records, config, **kwargs):
    for record in records:
        assert len(record["forecast"]["target"]) == 3
    (directory / "forecast_windows_01.png").write_bytes(b"figure" * 40)
    return ["forecast_windows_01.png"]


class CompactZeroShotTests(unittest.TestCase):
    def test_r2_uses_within_window_variation_and_is_invariant_to_window_levels(self):
        target = np.array([[0., 1., 2.], [10., 12., 14.]])
        prediction = target + .5
        score = compact_point_scores(target, prediction)
        self.assertAlmostEqual(score["r2"], .765625)  # arithmetic mean of .625 and .90625
        shifted = np.array([[0.], [1000.]])
        self.assertAlmostEqual(compact_point_scores(target + shifted, prediction + shifted)["r2"], .765625)
        self.assertTrue(np.isnan(compact_point_scores(np.array([[1., 1.], [2., 2.]]), np.zeros((2, 2)))["r2"]))
        self.assertLess(compact_point_scores(target, target + 10)["r2"], 0)

    def test_every_window_scored_macro_weights_and_minimal_retention(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = save_bundle(base / "input", sizes=(10, 16, 3))
            cfg = ZeroShotConfig(context_length=3, prediction_length=2, num_samples=5, inference_seeds=(42, 43))
            with patch("src.battery_zeroshot_minimal.fresh_pipeline", return_value=object()) as factory, \
                 patch("src.battery_zeroshot_minimal.predict_samples", side_effect=fake_predict) as predict, \
                 patch("src.battery_zeroshot_minimal.clear_device_cache"):
                run = battery_zeroshot(bundle, cfg, base / "output")
                self.assertEqual(predict.call_count, 4)
                self.assertEqual(verify_battery_zeroshot(run)["evaluated_windows"], 18)
                self.assertEqual(battery_zeroshot(bundle, cfg, base / "output"), run)
                self.assertEqual(factory.call_count, 1)
            self.assertEqual({p.name for p in run.iterdir()},
                {"summary.csv", "battery_metrics.csv", "run.json", "report.md", "forecast_windows_01.png"})
            battery = pd.read_csv(run / "battery_metrics.csv", keep_default_na=False, na_values=[""])
            self.assertTrue((battery.battery_id == "NA").any())
            self.assertEqual((battery.status != "evaluated").sum(), 1)
            summary = pd.read_csv(run / "summary.csv")
            records, _, _, _ = prepare_battery_windows(bundle, cfg)
            expected = [compact_scores(r["windows"]["target"], fake_predict(None, r["windows"]["context"], cfg, 42))[0] for r in records]
            actual = summary[summary.model == "Chronos"].iloc[0]
            for key in ["rmse", "r2", "wql", "picp80", "width80"]:
                self.assertAlmostEqual(actual[key], np.mean([r[key] for r in expected]))
            self.assertNotIn("r2_within", summary.columns)
            self.assertIn("r2", summary.columns)
            for defect in ["missing_chronos", "nonfinite_wql", "window_count"]:
                with self.subTest(defect=defect):
                    damaged = battery.copy()
                    index = damaged.index[(damaged.status == "evaluated") & (damaged.model == "Chronos")][0]
                    if defect == "missing_chronos":
                        damaged = damaged.drop(index)
                    elif defect == "nonfinite_wql":
                        damaged.loc[index, "wql"] = float("nan")
                    else:
                        damaged.loc[index, "n_windows"] += 1
                    damaged.to_csv(run / "battery_metrics.csv", index=False)
                    with self.assertRaises((ValueError, AssertionError)):
                        verify_battery_zeroshot(run)
            battery.to_csv(run / "battery_metrics.csv", index=False)
            verify_battery_zeroshot(run)

    def test_legacy_single_seed_r2_refresh_requires_no_inference_and_preserves_figures(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = save_bundle(base / "input", sizes=(10, 16, 3))
            cfg = ZeroShotConfig(context_length=3, prediction_length=2, num_samples=5, inference_seeds=(42,))
            with patch("src.battery_zeroshot_minimal.fresh_pipeline", return_value=object()), \
                 patch("src.battery_zeroshot_minimal.predict_samples", side_effect=fake_predict), \
                 patch("src.battery_zeroshot_minimal.clear_device_cache"), \
                 patch("src.battery_zeroshot_plots.forecast_figures", side_effect=fake_figures):
                run = battery_zeroshot(bundle, cfg, base / "output")
            expected = pd.read_csv(run / "battery_metrics.csv", keep_default_na=False, na_values=[""])
            figure = (run / "forecast_windows_01.png").read_bytes()
            for name in ["battery_metrics.csv", "summary.csv"]:
                pd.read_csv(run / name, keep_default_na=False, na_values=[""]).drop(columns="r2").to_csv(run / name, index=False)
            manifest = json.loads((run / "run.json").read_text())
            manifest.pop("r2_definition")
            (run / "run.json").write_text(json.dumps(manifest))
            before = (run / "battery_metrics.csv").read_bytes()
            with self.assertRaisesRegex(ValueError, "raw forecasts or re-inference"):
                refresh_compact_r2(run)
            self.assertEqual(before, (run / "battery_metrics.csv").read_bytes())
            self.assertEqual(figure, (run / "forecast_windows_01.png").read_bytes())

    def test_resume_failed_run_retains_completed_batteries_and_finishes_cleanup(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = save_bundle(base / "input", sizes=(10, 16))
            cfg = ZeroShotConfig(context_length=3, prediction_length=2, num_samples=5, inference_seeds=(42,))
            calls = []
            def interrupted(pipeline, context, config, seed):
                calls.append(len(context))
                if len(calls) == 2:
                    raise RuntimeError("interrupted")
                return fake_predict(pipeline, context, config, seed)
            with patch("src.battery_zeroshot_minimal.fresh_pipeline", return_value=object()), \
                 patch("src.battery_zeroshot_minimal.predict_samples", side_effect=interrupted), \
                 patch("src.battery_zeroshot_minimal.clear_device_cache"):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    battery_zeroshot(bundle, cfg, base / "output")
            run = base / "output/input"
            self.assertEqual(json.loads((run / "run.json").read_text())["status"], "failed")
            with patch("src.battery_zeroshot_minimal.fresh_pipeline", return_value=object()), \
                 patch("src.battery_zeroshot_minimal.predict_samples", side_effect=fake_predict) as predict, \
                 patch("src.battery_zeroshot_minimal.clear_device_cache"), \
                 patch("src.battery_zeroshot_plots.forecast_figures", side_effect=fake_figures):
                battery_zeroshot(bundle, cfg, base / "output")
                self.assertEqual(predict.call_count, 1)
            self.assertFalse(list(run.glob("*.npz")))
            self.assertEqual(verify_battery_zeroshot(run)["batteries_evaluated"], 2)


if __name__ == "__main__":
    unittest.main()
