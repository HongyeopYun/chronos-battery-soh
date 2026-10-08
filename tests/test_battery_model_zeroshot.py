from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, weighted_quantile_loss
from src.battery_model_zeroshot import (
    PHASES, phase_labels, quantile_scores, predict_quantiles, run_model_battery, verify_result, write_comparison,
)
from test_battery_zeroshot import save_bundle


def fake_quantiles(pipeline, context, config):
    point = np.repeat(context[:, -1:], config.prediction_length, axis=1)
    return point, point[..., None] + np.linspace(-.01, .01, 9)


def fake_figures(directory, records, config, **kwargs):
    name = kwargs['filename_prefix'] + 'forecast_windows_01.png'
    for record in records:
        assert len(record['forecast']['target']) == 3
    (directory / name).write_bytes(b'figure' * 40)
    return [name]


class ModelBatteryTests(unittest.TestCase):
    def test_thirds_use_observed_cycle_span_and_ignore_window_density(self):
        np.testing.assert_array_equal(phase_labels([5, 20, 31, 59, 61, 80], np.arange(1, 91)),
                                      ['early', 'early', 'middle', 'middle', 'late', 'late'])
        # A continued measurement starts at cycle 1001; it is not early physical life.
        np.testing.assert_array_equal(phase_labels([1020, 1060, 1080], np.arange(1001, 1091)), PHASES)
        with self.assertRaises(ValueError):
            phase_labels([0], np.arange(1, 91))

    def test_r2_is_arithmetic_mean_of_window_scores(self):
        target = np.array([[0., 1., 2.], [10., 12., 14.]])
        result = quantile_scores(target, target + .5)
        expected = (.625 + .90625) / 2
        self.assertAlmostEqual(result['r2'], expected)
        self.assertNotAlmostEqual(result['r2'], .85)
        self.assertEqual(result['r2_n_windows'], 2)
        self.assertLess(quantile_scores(target, target + 20)['r2'], 0)
        self.assertTrue(np.isnan(quantile_scores(np.ones((2, 2)), np.zeros((2, 2)))['r2']))

    def test_same_wql_for_sample_quantiles_and_direct_quantiles(self):
        target = np.array([[1., 9.], [0., 10.]])
        samples = np.broadcast_to(np.array([0., 10.])[None, :, None], (2, 2, 2)).copy()
        values = np.quantile(samples, np.arange(1, 10)/10, axis=1).transpose(1, 2, 0)
        result = quantile_scores(target, np.full((2, 2), 5.), values)
        self.assertAlmostEqual(result['wql'], weighted_quantile_loss(target, samples))
        self.assertEqual(result['picp80'], .5)
        self.assertEqual(result['width80'], 8.)
        crossed = values.copy()
        crossed[..., 0] = 11
        with self.assertRaisesRegex(ValueError, 'Crossed'):
            quantile_scores(target, np.full((2, 2), 5.), crossed)

    def test_c2_axis_conversion_and_independent_univariate_tasks(self):
        class FakePipeline:
            def predict_quantiles(self, **kwargs):
                self.kwargs = kwargs
                point = torch.arange(4, dtype=torch.float32).reshape(1, 4)
                values = point[..., None] + torch.arange(9, dtype=torch.float32)
                return [values, values + 10], [point, point + 10]
        pipeline = FakePipeline()
        config = ZeroShotConfig(model_id='amazon/chronos-2', context_length=8, prediction_length=4, inference_seeds=(42,))
        point, values = predict_quantiles(pipeline, np.ones((2, 8)), config)
        self.assertEqual(point.shape, (2, 4))
        self.assertEqual(values.shape, (2, 4, 9))
        self.assertEqual(values[1, 3, 8], 21)
        self.assertFalse(pipeline.kwargs['cross_learning'])
        self.assertTrue(all(value.ndim == 1 for value in pipeline.kwargs['inputs']))
        self.assertNotIn('num_samples', pipeline.kwargs)

    def test_phase_macro_checkpoint_resume_cache_and_minimum_retention(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = save_bundle(base / 'input', sizes=(100, 160))
            config = ZeroShotConfig(context_length=3, prediction_length=2, inference_seeds=(42,))
            call_count = 0
            def interrupted(pipeline, context, config):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError('interrupted')
                return fake_quantiles(pipeline, context, config)
            with patch('src.battery_model_zeroshot.load_model', return_value=object()), \
                 patch('src.battery_model_zeroshot.predict_quantiles', side_effect=interrupted), \
                 patch('src.battery_model_zeroshot.clear_device_cache'):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    run_model_battery(bundle, config, base / 'output')
            with patch('src.battery_model_zeroshot.load_model', return_value=object()) as factory, \
                 patch('src.battery_model_zeroshot.predict_quantiles', side_effect=fake_quantiles) as predict, \
                 patch('src.battery_model_zeroshot.clear_device_cache'), \
                 patch('src.battery_zeroshot_plots.forecast_figures', side_effect=fake_figures):
                run = run_model_battery(bundle, config, base / 'output')
                self.assertEqual(predict.call_count, 1)
                run_model_battery(bundle, config, base / 'output')
                self.assertEqual(factory.call_count, 1)
            self.assertEqual({p.name for p in run.iterdir()}, {f't5-battery_{name}' for name in
                ['summary.csv', 'battery_metrics.csv', 'run.json', 'report.md', 'forecast_windows_01.png']})
            manifest = verify_result(run, 't5-battery')
            self.assertEqual(manifest['evaluated_windows'], 252)
            battery = pd.read_csv(run / 't5-battery_battery_metrics.csv', keep_default_na=False, na_values=[''])
            self.assertTrue((battery.battery_id == 'NA').any())
            summary = pd.read_csv(run / 't5-battery_summary.csv')
            records, _, data, _ = prepare_battery_windows(bundle, config)
            expected = []
            for record in records:
                rows = data[data.battery_id == record['battery_id']]
                labels = phase_labels(record['windows']['forecast_start'], rows.cycle)
                mask = labels == 'early'
                point, values = fake_quantiles(None, record['windows']['context'], config)
                expected.append(quantile_scores(record['windows']['target'][mask], point[mask], values[mask]))
            actual = summary[(summary.model == 't5-battery') & (summary.phase == 'early')].iloc[0]
            for key in ['rmse', 'r2', 'wql', 'picp80', 'width80']:
                self.assertAlmostEqual(actual[key], np.mean([result[key] for result in expected]))
            with self.assertRaisesRegex(ValueError, 'different inputs/settings'):
                run_model_battery(bundle, replace(config, stride=2), base / 'output')

    def test_comparison_requires_identical_inputs_and_baselines(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = save_bundle(base / 'input', sizes=(100,))
            with patch('src.battery_model_zeroshot.load_model', return_value=object()), \
                 patch('src.battery_model_zeroshot.predict_quantiles', side_effect=fake_quantiles), \
                 patch('src.battery_model_zeroshot.clear_device_cache'), \
                 patch('src.battery_zeroshot_plots.forecast_figures', side_effect=fake_figures), \
                 patch('src.battery_model_zeroshot.PILOT', {'input': ('NA',)}):
                for tag, model in [('t5-battery', 'amazon/chronos-t5-base'), ('c2-battery', 'amazon/chronos-2')]:
                    config = ZeroShotConfig(model_id=model, context_length=3, prediction_length=2, inference_seeds=(42,))
                    run_model_battery(bundle, config, base / tag / 'pilot')
                write_comparison(base)
                table = pd.read_csv(base / 't5-c2-battery-comparison/t5-c2-battery_summary.csv')
                self.assertEqual(len(table), 24)
                manifest_path = base / 'c2-battery/pilot/input/c2-battery_run.json'
                manifest = json.loads(manifest_path.read_text())
                manifest['config']['stride'] = 2
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'Unmatched'):
                    write_comparison(base)

    def test_full_scope_keeps_excluded_records_and_unavailable_early_phase(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = save_bundle(base / 'input', sizes=(100, 3))
            config = ZeroShotConfig(model_id='amazon/chronos-2', context_length=64, prediction_length=16, inference_seeds=(42,))
            with patch('src.battery_model_zeroshot.load_model', return_value=object()), \
                 patch('src.battery_model_zeroshot.predict_quantiles', side_effect=fake_quantiles), \
                 patch('src.battery_model_zeroshot.clear_device_cache'):
                # Render the real graph to exercise an empty Early panel and legend.
                run = run_model_battery(bundle, config, base / 'output', scope='full')
            manifest = verify_result(run, 'c2-battery')
            self.assertEqual(manifest['batteries_evaluated'], 1)
            self.assertEqual(manifest['batteries_skipped'], 1)
            self.assertEqual(manifest['evaluated_windows'], 21)
            coverage = pd.read_csv(run / 'c2-battery_coverage.csv', keep_default_na=False)
            self.assertEqual(coverage.iloc[0].early_windows, 0)
            self.assertEqual(coverage.iloc[0].middle_windows, 3)
            self.assertEqual(coverage.iloc[0].late_windows, 18)
            scores = pd.read_csv(run / 'c2-battery_battery_metrics.csv', keep_default_na=False)
            self.assertEqual(set(scores.phase), {'all', 'middle', 'late'})
            self.assertFalse(list(run.glob('*.npz')))
            with self.assertRaisesRegex(ValueError, 'must not restrict'):
                run_model_battery(bundle, replace(config, battery_ids=('NA',)), base / 'restricted', scope='full')


if __name__ == '__main__':
    unittest.main()
