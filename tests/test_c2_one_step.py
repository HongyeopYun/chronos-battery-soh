import unittest
from types import SimpleNamespace

import numpy as np
import torch
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model

from src.c2_one_step import (crop_windows, fixed_inputs, make_dataset, one_step_loss,
                             validation_scores, point_metrics, probability_metrics, baselines)
from src.c2_partial_tuning import configure_subset, model_batch


def tiny_model():
    return Chronos2Model(Chronos2CoreConfig(d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=2,
        chronos_config=dict(context_length=64, input_patch_size=16, input_patch_stride=16,
                            output_patch_size=16, quantiles=[i/10 for i in range(1, 10)], use_reg_token=True)))


class OneStepTests(unittest.TestCase):
    def test_dataset_has_exact_fixed_split_and_one_label(self):
        context = np.arange(128, dtype=np.float32).reshape(2, 64)
        w = dict(context=context, context_cycle=np.array([np.arange(1, 65), np.arange(65, 129)]),
                 target=np.array([[64], [128]], dtype=np.float32), target_cycle=np.array([[65], [129]]))
        tasks = fixed_inputs(crop_windows(w, 64), [0, 1])
        batch = next(iter(make_dataset(tasks, 64)))
        np.testing.assert_array_equal(batch['context'].numpy(), context)
        np.testing.assert_array_equal(batch['future_target'].numpy(), w['target'])
        self.assertTrue(torch.isnan(batch['future_covariates']).all())
        train = iter(make_dataset(tasks, 64, training=True))
        for _ in range(3):
            batch = next(train)
            self.assertEqual(tuple(batch['future_target'].shape), (4, 1))
            np.testing.assert_array_equal(batch['context'][:, -1].numpy()+1, batch['future_target'][:, 0].numpy())

    def test_only_head_changes_and_unused_output_positions_have_zero_gradient(self):
        torch.manual_seed(42)
        model = tiny_model()
        original = {n: p.detach().clone() for n, p in model.named_parameters()}
        names = configure_subset(model, 'head')
        captured = []
        def capture(module, inputs, output):
            output.retain_grad()
            captured.append(output)
        handle = model.output_patch_embedding.register_forward_hook(capture)
        tasks = [dict(target=np.linspace(1, .9, 65, dtype=np.float32))]
        batch = model_batch(next(iter(make_dataset(tasks, 64))), 'cpu')
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        loss, output = one_step_loss(model, batch)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, output.loss*16)
        loss.backward()
        grad = captured[0].grad.reshape(-1, 9, 16)
        self.assertTrue(torch.count_nonzero(grad[:, :, 0]) > 0)
        self.assertEqual(torch.count_nonzero(grad[:, :, 1:]).item(), 0)
        optimizer.step()
        changed = [n for n, p in model.named_parameters() if not torch.equal(p, original[n])]
        self.assertTrue(changed)
        self.assertTrue(set(changed).issubset(names))
        handle.remove()

    def test_validation_ignores_unsupervised_quantile_crossings(self):
        class Fake:
            quantiles = torch.arange(1, 10)/10
            def eval(self): pass
            def __call__(self, **batch):
                q = self.quantiles[None, :, None].repeat(1, 1, 16)
                q[:, :, 1:] = torch.flip(q[:, :, 1:], dims=[1])
                return SimpleNamespace(loss=torch.tensor(2.), quantile_preds=q)
        scores = validation_scores(Fake(), [dict(future_target=torch.tensor([[.5]]))])
        self.assertEqual(scores['validation_loss'], 32.)
        self.assertEqual(scores['validation_crossing_fraction'], 0.)
        self.assertEqual(scores['validation_invalid_interval_fraction'], 0.)

    def test_point_metrics_are_across_timestamps_not_single_target_r2(self):
        y, p = np.array([1., 2., 3.]), np.array([1., 1., 3.])
        s = point_metrics(y, p)
        self.assertAlmostEqual(s['mae'], 1/3)
        self.assertAlmostEqual(s['rmse'], np.sqrt(1/3))
        self.assertAlmostEqual(s['temporal_r2'], .5)
        self.assertTrue(np.isnan(point_metrics([1.], [1.])['temporal_r2']))
        self.assertTrue(np.isnan(point_metrics([1., 1.], [1., 1.])['temporal_r2']))

    def test_persistence_and_linear_use_past_only_and_crossings_remain_visible(self):
        s = baselines(np.array([[4., 3., 2., 1.]]))
        np.testing.assert_array_equal(s['LastValue'], [1.])
        np.testing.assert_allclose(s['Linear'], [0.])
        q = np.linspace(.7, 1.1, 9)[None, :]
        q[0, 0], q[0, -1] = 1.2, .6
        metrics = probability_metrics([.9], q)
        self.assertTrue(np.isnan(metrics['picp80']))
        self.assertEqual(metrics['invalid_interval_fraction'], 1.)
        self.assertGreater(metrics['wql'], 0.)


if __name__ == '__main__':
    unittest.main()
