import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.battery_zeroshot import (
    ZeroShotConfig, baseline_predictions, battery_zeroshot, prepare_battery_windows,
    refresh_battery_zeroshot, score_prediction, verify_battery_zeroshot, weighted_quantile_loss,
)
from src.soh_data import sha256, write_json
from src.soh_experiment import load_config, load_soh, make_windows

ROOT = Path(__file__).resolve().parents[1]


def save_bundle(directory, sizes=(100,), flags=None, gap=None, segment=None, constant=False):
    directory.mkdir(parents=True, exist_ok=True)
    records, metadata = [], []
    for index, size in enumerate(sizes):
        name = "NA" if index == 0 else f"cell{index}"
        q = np.ones(size) * 2 if constant else np.linspace(2., 1.6, size)
        for i, value in enumerate(q, 1):
            if gap == i:
                continue
            records.append({"dataset": "synthetic", "battery_id": name, "cycle": i, "capacity_ah": value,
                "soh": value / q[0], "is_observed": True, "capacity_kind": "full_discharge",
                "protocol_id": "normal", "cycle_description": "", "quality_flags": (flags or {}).get(i, ""),
                "segment_id": "later" if segment and i >= segment else "first"})
        metadata.append({"dataset": "synthetic", "battery_id": name, "batch": "batch",
            "reference_method": "first_cycle", "reference_cycle": 1, "reference_capacity_ah": 2.,
            "source_file": "synthetic", "source_sha256": "test", "cycle_number_source": "original",
            "physical_battery_id": f"physical{index}"})
    pd.DataFrame(records).to_csv(directory / "soh.csv", index=False, float_format="%.17g")
    pd.DataFrame(metadata).to_csv(directory / "batteries.csv", index=False)
    write_json(directory / "schema.json", {"version": 1})
    write_json(directory / "manifest.json", {"schema_version": 1, "status": "complete", "source_files": {},
        "artifact_sha256": {name: sha256(directory / name) for name in ["soh.csv", "batteries.csv", "schema.json"]}})
    return directory


def fake_predict(pipeline, context, config, seed):
    noise = np.linspace(-.01, .01, config.num_samples)[None, :, None]
    return np.broadcast_to(context[:, -1:, None] + noise,
        (len(context), config.num_samples, config.prediction_length)).copy()


def fake_report(directory, *args, **kwargs):
    (directory / "report.md").write_text("test report", encoding="utf-8")


class BatteryZeroShotTests(unittest.TestCase):
    def test_wql_hand_calculation_scaling_quantile_direction_and_zero_denominator(self):
        target = np.array([[2.]])
        samples = np.array([[[1.]]])
        self.assertAlmostEqual(weighted_quantile_loss(target, samples), .5)
        self.assertAlmostEqual(weighted_quantile_loss(target * 10, samples * 10), .5)
        self.assertAlmostEqual(weighted_quantile_loss(target, samples, (.9,)), .9)
        self.assertAlmostEqual(weighted_quantile_loss(target, np.array([[[3.]]]), (.9,)), .1)
        self.assertEqual(weighted_quantile_loss(target, target[:, None]), 0)
        self.assertTrue(np.isnan(weighted_quantile_loss(np.zeros((1, 1)), samples)))
        for levels in [(), (0.,), (1.,), (.5, .5), (.9, .1), (float("nan"),)]:
            with self.assertRaises(ValueError):
                weighted_quantile_loss(target, samples, levels)
        with self.assertRaises(ValueError):
            weighted_quantile_loss(target, np.ones((2, 1, 1)))

    def test_picp_includes_interval_boundaries_and_all_metric_table_levels(self):
        target = np.array([[1., 9.], [0., 10.]])
        samples = np.broadcast_to(np.array([0., 10.])[None, :, None], (2, 2, 2)).copy()
        record = {"dataset": "test", "battery_id": "test", "physical_battery_id": "test",
                  "windows": {"target": target, "forecast_start": np.array([10, 20])}}
        row, horizons, windows = score_prediction(record, np.full((2, 2), 5.), "Chronos", "median", samples=samples)
        self.assertEqual(row["picp80"], .5)
        self.assertEqual(row["width80"], 8.)
        self.assertEqual([w["picp80"] for w in windows], [1., 0.])
        self.assertEqual([h["picp80"] for h in horizons], [.5, .5])
        for h, values in enumerate(horizons):
            self.assertAlmostEqual(values["wql"], weighted_quantile_loss(target[:, h:h+1], samples[:, :, h:h+1]))
        for i, values in enumerate(windows):
            self.assertAlmostEqual(values["wql"], weighted_quantile_loss(target[i:i+1], samples[i:i+1]))
        baseline, h, w = score_prediction(record, np.full((2, 2), 5.), "LastValue", "point")
        self.assertTrue(all(np.isnan(values["wql"]) and np.isnan(values["picp80"]) for values in [baseline, *h, *w]))
        float_samples = np.broadcast_to(np.array([1.01, 1.17], dtype=np.float32)[None, :, None], (2, 2, 2)).copy()
        float_row, _, _ = score_prediction(record, np.full((2, 2), 1.09), "Chronos", "median", samples=float_samples)
        quantiles = np.quantile(float_samples.astype(np.float64), [.1, .9], axis=1, method="linear")
        self.assertEqual(float_row["width80"], (quantiles[1] - quantiles[0]).mean())

    def test_wql_normalizes_across_targets_before_averaging_windows(self):
        target = np.array([[1.], [100.]])
        samples = np.array([[[0.]], [[100.]]])
        self.assertAlmostEqual(weighted_quantile_loss(target, samples), 1 / 101)
        per_window_average = np.mean([weighted_quantile_loss(target[i:i+1], samples[i:i+1]) for i in range(2)])
        self.assertAlmostEqual(per_window_average, .5)

    def test_metric_refresh_preserves_forecasts_and_original_manifest_without_inference(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = save_bundle(base / "input", sizes=(10,))
            cfg = ZeroShotConfig(context_length=3, prediction_length=2, num_samples=5, inference_seeds=(42,))
            with patch("src.battery_zeroshot.fresh_pipeline", return_value=object()), \
                 patch("src.battery_zeroshot.predict_samples", side_effect=fake_predict), \
                 patch("src.battery_zeroshot.clear_device_cache"), \
                 patch("src.battery_zeroshot_plots.write_report", side_effect=fake_report):
                run = battery_zeroshot(bundle, cfg, base / "output")
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["signature"]["implementation_sha256"]["battery_zeroshot.py"] = "earlier metric code"
            write_json(manifest_path, manifest)
            original = manifest_path.read_bytes()
            raw = {p.name: sha256(p) for p in (run / "predictions").glob("*.npz")}
            with patch("src.battery_zeroshot.fresh_pipeline") as factory, \
                 patch("src.battery_zeroshot.predict_samples") as predict, \
                 patch("src.battery_zeroshot_plots.write_report", side_effect=fake_report):
                refresh_battery_zeroshot(run)
                factory.assert_not_called(); predict.assert_not_called()
                self.assertEqual(battery_zeroshot(bundle, cfg, base / "output"), run)
                factory.assert_not_called()
            self.assertEqual((run / "inference_manifest.json").read_bytes(), original)
            self.assertEqual(raw, {p.name: sha256(p) for p in (run / "predictions").glob("*.npz")})
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["evaluation_protocol"]["primary_metrics"], ["rmse", "r2", "wql", "picp80"])
            manifest["signature"]["implementation_sha256"]["soh_experiment.py"] = "changed inference"
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "inference/data implementation changed"):
                refresh_battery_zeroshot(run)

    def test_preparation_preserves_gaps_masks_segments_and_sampling_identity(self):
        with TemporaryDirectory() as temporary:
            bundle = save_bundle(Path(temporary) / "input", flags={1: "initial_capacity_test", 50: "capacity_outlier"}, gap=70, segment=80)
            cfg = ZeroShotConfig(context_length=4, prediction_length=2)
            full, coverage, _, _ = prepare_battery_windows(bundle, cfg)
            windows = full[0]["windows"]
            joined = np.concatenate([windows["context_cycle"], windows["target_cycle"]], axis=1)
            self.assertTrue((np.diff(joined, axis=1) == 1).all())
            self.assertFalse(np.isin(joined, [1, 50, 70]).any())
            self.assertFalse(((joined[:, 0] < 80) & (joined[:, -1] >= 80)).any())
            sampled, small, _, _ = prepare_battery_windows(bundle, replace(cfg, stride=2, max_windows_per_battery=3))
            positions = np.arange(len(windows["target"]))[::2]
            expected = positions[np.linspace(0, len(positions)-1, 3, dtype=int)]
            np.testing.assert_array_equal(sampled[0]["windows"]["forecast_start"], windows["forecast_start"][expected])
            self.assertEqual(small.evaluated_windows.sum(), 3)
            self.assertEqual(small.eligible_windows.sum(), coverage.eligible_windows.sum())
            self.assertEqual(full[0]["physical_battery_id"], "physical0")

    def test_real_nasa_common_bundle_matches_frozen_window_definition(self):
        records, _, _, _ = prepare_battery_windows(ROOT / "data/processed/clean/nasa_aging", ZeroShotConfig())
        old = load_config(ROOT)
        series = load_soh(ROOT, old)
        self.assertEqual(len(records), 4)
        for record in records:
            windows = make_windows(series[record["battery_id"]], old.context_length, old.prediction_length)
            for key, value in windows.items():
                np.testing.assert_allclose(record["windows"][key], value, rtol=0, atol=1e-14)

    def test_run_reuse_one_battery_r2_aggregation_and_raw_data_preservation(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = save_bundle(base / "input", sizes=(10, 16))
            source = {p.name: sha256(p) for p in bundle.iterdir()}
            cfg = ZeroShotConfig(context_length=3, prediction_length=2, num_samples=5, inference_seeds=(42, 43))
            with patch("src.battery_zeroshot.fresh_pipeline", return_value=object()) as factory, \
                 patch("src.battery_zeroshot.predict_samples", side_effect=fake_predict), \
                 patch("src.battery_zeroshot.clear_device_cache"), \
                 patch("src.battery_zeroshot_plots.write_report", side_effect=fake_report):
                run = battery_zeroshot(bundle, cfg, base / "output")
                self.assertEqual(battery_zeroshot(bundle, cfg, base / "output"), run)
                self.assertEqual(factory.call_count, 1)
                verified = verify_battery_zeroshot(run)
                self.assertEqual(verified["windows"], 18)
                metrics = pd.read_csv(run / "metrics.csv", keep_default_na=False)
                self.assertTrue((metrics.battery_id == "NA").any())
                battery = pd.read_csv(run / "battery_summary.csv")
                macro = pd.read_csv(run / "summary.csv")
                reference = battery[(battery.model == "Chronos") & (battery.estimator == "median")].rmse.mean()
                actual = macro[(macro.model == "Chronos") & (macro.estimator == "median")].rmse_mean.iloc[0]
                self.assertAlmostEqual(actual, reference)
                self.assertTrue({"rmse", "r2", "wql", "picp80", "width80", "crps"}.issubset(metrics.columns))
                single = battery_zeroshot(bundle, replace(cfg, battery_ids=("NA",)), base / "output")
                self.assertEqual(verify_battery_zeroshot(single)["batteries_evaluated"], 1)
                self.assertNotEqual(single, run)
            self.assertEqual(source, {p.name: sha256(p) for p in bundle.iterdir()})

    def test_tampered_metric_rejected_even_when_checksum_is_updated(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = save_bundle(base / "input", sizes=(10,))
            cfg = ZeroShotConfig(context_length=3, prediction_length=2, num_samples=5, inference_seeds=(42,))
            with patch("src.battery_zeroshot.fresh_pipeline", return_value=object()), \
                 patch("src.battery_zeroshot.predict_samples", side_effect=fake_predict), \
                 patch("src.battery_zeroshot.clear_device_cache"), \
                 patch("src.battery_zeroshot_plots.write_report", side_effect=fake_report):
                run = battery_zeroshot(bundle, cfg, base / "output")
            table = pd.read_csv(run / "metrics.csv", keep_default_na=False, na_values=[""])
            table.loc[0, "rmse"] = 999.
            table.to_csv(run / "metrics.csv", index=False)
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                verify_battery_zeroshot(run)
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            manifest["artifact_sha256"]["metrics.csv"] = sha256(run / "metrics.csv")
            write_json(run / "manifest.json", manifest)
            with self.assertRaises(AssertionError):
                verify_battery_zeroshot(run)

    def test_empty_windows_skip_records_or_fail_before_loading_model(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = save_bundle(base / "input", sizes=(5, 100))
            cfg = ZeroShotConfig()
            records, coverage, _, _ = prepare_battery_windows(bundle, cfg)
            self.assertEqual(len(records), 1)
            self.assertEqual(coverage.status.iloc[0], "skipped_no_eligible_windows")
            with patch("src.battery_zeroshot.fresh_pipeline") as factory:
                with self.assertRaisesRegex(ValueError, "No eligible"):
                    battery_zeroshot(bundle, replace(cfg, battery_ids=("NA",)), base / "output")
                factory.assert_not_called()
                self.assertFalse((base / "output").exists())
            with self.assertRaisesRegex(ValueError, "Unknown batteries"):
                prepare_battery_windows(bundle, replace(cfg, battery_ids=("missing",)))

    def test_failed_inference_is_recorded_and_constant_target_r2_remains_nan(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = save_bundle(base / "input", sizes=(10,), constant=True)
            cfg = ZeroShotConfig(context_length=1, prediction_length=2, num_samples=5, inference_seeds=(42,))
            records, _, _, _ = prepare_battery_windows(bundle, cfg)
            baselines = baseline_predictions(records[0]["windows"])
            np.testing.assert_array_equal(baselines["Linear"], baselines["LastValue"])
            row, _, _ = score_prediction(records[0], baselines["LastValue"], "LastValue", "point")
            self.assertEqual(row["rmse"], 0)
            self.assertTrue(np.isnan(row["r2"]))
            with patch("src.battery_zeroshot.fresh_pipeline", return_value=object()), \
                 patch("src.battery_zeroshot.predict_samples", side_effect=RuntimeError("test failure")), \
                 patch("src.battery_zeroshot.clear_device_cache"):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    battery_zeroshot(bundle, cfg, base / "output")
            manifest = json.loads(next((base / "output").glob("*/manifest.json")).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verify_battery_zeroshot(base / "output" / manifest["run_id"])

    def test_invalid_sampling_configuration_and_overlapping_output_rejected(self):
        for changes in [{"stride": 0}, {"num_samples": True}, {"inference_seeds": (42, 42)},
                        {"max_windows_per_battery": 0}, {"revision": "main"}, {"battery_ids": ()}, {"device": "bad"}]:
            with self.assertRaises(ValueError):
                ZeroShotConfig(**changes)
        with TemporaryDirectory() as temporary:
            bundle = save_bundle(Path(temporary) / "input")
            with self.assertRaisesRegex(ValueError, "separate"):
                battery_zeroshot(bundle, output_root=bundle / "results")


if __name__ == "__main__":
    unittest.main()
