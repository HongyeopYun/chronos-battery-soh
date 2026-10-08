"""Exercise the actual pipeline-copy/Trainer boundary; flag-only checks miss it."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model
from chronos.chronos2.pipeline import Chronos2Pipeline
from transformers import TrainerCallback

from src.c2_fewshot_trainer import HeadOnlyChronos2Trainer, HEAD_PREFIX


class HeadTuningTests(unittest.TestCase):
    def test_crossed_intervals_preserve_point_scores_but_invalidate_picp(self):
        from src.battery_fewshot_tuning import adaptation_scores
        y = np.linspace(.9, .85, 16)[None]
        q = y[..., None] + np.linspace(-.01, .01, 9)
        good = adaptation_scores(y, q)
        q[0, 0, 0] = q[0, 0, -1] + .01
        crossed = adaptation_scores(y, q)
        self.assertEqual(crossed["r2"], good["r2"])
        self.assertEqual(crossed["rmse"], good["rmse"])
        self.assertTrue(np.isfinite(crossed["wql"]))
        self.assertTrue(np.isnan(crossed["picp80"]))
        self.assertTrue(np.isnan(crossed["width80"]))
        self.assertEqual(crossed["invalid_interval_fraction"], 1 / 16)
        self.assertGreater(q[0, 0, 0], q[0, 0, -1])

    def test_pipeline_fit_updates_only_head_and_keeps_base_intact(self):
        config = Chronos2CoreConfig(
            d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=1,
            chronos_config=dict(context_length=64, input_patch_size=16, input_patch_stride=16,
                                output_patch_size=16, quantiles=[.1, .5, .9], use_reg_token=True))
        base = Chronos2Pipeline(Chronos2Model(config))
        initial = {n: p.detach().clone() for n, p in base.model.named_parameters()}
        inspected = []

        class InspectOptimizer(TrainerCallback):
            def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
                expected = {id(p) for n, p in model.named_parameters() if n.startswith(HEAD_PREFIX)}
                actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
                if expected != actual:
                    raise AssertionError("Optimizer contains frozen weights or omits head weights")
                inspected.append(True)

        task = {"target": np.linspace(1, .8, 80, dtype=np.float32),
                "past_covariates": {"v": np.r_[np.linspace(4, 3, 64), np.full(16, np.nan)]}}
        with tempfile.TemporaryDirectory() as directory:
            with patch("chronos.chronos2.trainer.Chronos2Trainer", HeadOnlyChronos2Trainer):
                tuned = base.fit(inputs=[task], finetune_mode="full", context_length=64,
                                 prediction_length=16, min_past=64, num_steps=2, batch_size=2,
                                 learning_rate=1e-3, output_dir=directory, optim="adamw_torch",
                                 callbacks=[InspectOptimizer()], use_cpu=True,
                                 disable_tqdm=True, dataloader_pin_memory=False)
        self.assertEqual(inspected, [True])
        changed = []
        for name, parameter in tuned.model.named_parameters():
            self.assertTrue(torch.equal(initial[name], dict(base.model.named_parameters())[name]))
            if not torch.equal(initial[name], parameter):
                changed.append(name)
                self.assertTrue(name.startswith(HEAD_PREFIX), name)
            if not name.startswith(HEAD_PREFIX):
                self.assertIsNone(parameter.grad)
                self.assertFalse(parameter.requires_grad)
        self.assertTrue(changed)
        self.assertFalse(tuned.model.encoder.training)


if __name__ == "__main__":
    unittest.main()
