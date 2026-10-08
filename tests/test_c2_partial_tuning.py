import unittest
import numpy as np
import torch
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

from src.c2_partial_tuning import configure_subset, corrected_loss, model_batch, validation_scores, choose_checkpoint


def tiny_model():
    return Chronos2Model(Chronos2CoreConfig(
        d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=2,
        chronos_config=dict(context_length=64, input_patch_size=16, input_patch_stride=16,
                            output_patch_size=16, quantiles=[i / 10 for i in range(1, 10)], use_reg_token=True)))


def tasks(count):
    return [dict(target=np.linspace(1 + i * .01, .8, 80, dtype=np.float32),
                 past_covariates=dict(v=np.r_[np.linspace(4, 3, 64), np.full(16, np.nan)].astype(np.float32)))
            for i in range(count)]


def batches(inputs, size):
    return [model_batch(batch, "cpu") for batch in Chronos2Dataset(
        inputs=inputs, context_length=64, prediction_length=16, batch_size=size,
        output_patch_size=16, min_past=64, mode=DatasetMode.VALIDATION)]


class PartialTuningTests(unittest.TestCase):
    def test_update_reaches_only_head_and_last_group_attention(self):
        torch.manual_seed(42)
        model = tiny_model()
        original = {n: p.detach().clone() for n, p in model.named_parameters()}
        names = configure_subset(model, "head_group")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
        batch = batches(tasks(2), 4)[0]
        self.assertEqual(torch.isfinite(batch["future_target"]).sum().item(), 32)
        self.assertTrue(torch.isnan(batch["future_covariates"]).all())
        loss = corrected_loss(model, batch)
        loss.backward()
        optimizer.step()
        changed = [n for n, p in model.named_parameters() if not torch.equal(p, original[n])]
        self.assertTrue(any(n.startswith("encoder.block.1.layer.1.") for n in changed))
        self.assertTrue(any(n.startswith("output_patch_embedding.") for n in changed))
        self.assertTrue(set(changed).issubset(names))
        self.assertFalse(model.encoder.block[0].training)
        self.assertFalse(model.encoder.block[1].layer[0].training)
        self.assertTrue(model.encoder.block[1].layer[1].training)
        self.assertFalse(model.encoder.block[1].layer[2].training)
        self.assertTrue(model.output_patch_embedding.training)
        for n, p in model.named_parameters():
            if n not in names:
                self.assertIsNone(p.grad)
                self.assertTrue(torch.equal(p, original[n]))

    def test_validation_loss_weights_incomplete_batch_by_soh_windows(self):
        model = tiny_model()
        inputs = tasks(5)
        full = validation_scores(model, batches(inputs, 10))
        split = validation_scores(model, batches(inputs, 4))
        np.testing.assert_allclose(full["validation_loss"], split["validation_loss"], rtol=1e-5)
        self.assertEqual(full["validation_crossing_fraction"], split["validation_crossing_fraction"])

    def test_crossing_guard_and_step_zero_fallback(self):
        baseline = dict(step=0, learning_rate=0., validation_loss=1.,
                        validation_crossing_fraction=0., validation_invalid_interval_fraction=0.)
        worse = {**baseline, "step": 20, "learning_rate": 1e-5, "validation_loss": 1.1}
        crossed = {**worse, "validation_loss": .8, "validation_crossing_fraction": .1}
        good = {**worse, "step": 40, "validation_loss": .9}
        self.assertIs(choose_checkpoint([baseline, worse, crossed], baseline), baseline)
        self.assertIs(choose_checkpoint([baseline, worse, crossed, good], baseline), good)


if __name__ == "__main__":
    unittest.main()
