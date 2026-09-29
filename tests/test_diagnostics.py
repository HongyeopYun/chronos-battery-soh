import unittest
import numpy as np
import torch
import torch.nn.functional as F

from scripts.diagnose_lora import distribution_metrics, token_diagnostics


class DiagnosticTests(unittest.TestCase):
    def test_empirical_crps_matches_pairwise_definition(self):
        samples = np.array([[[0., 1.], [2., 4.], [7., 9.]]])
        target = np.array([[1., 3.]])
        expected = (abs(samples - target[:, None]).mean(axis=1)
                    - .5 * abs(samples[:, :, None, :] - samples[:, None, :, :]).mean(axis=(1, 2))).mean()
        self.assertAlmostEqual(distribution_metrics(target, samples)["crps"], expected)

    def test_eos_mask_preserves_forecast_weight_and_decoder_prefix(self):
        logits = torch.tensor([[[1., 2., 3.], [3., 2., 1.], [2., 3., 1.]]], requires_grad=True)
        labels = torch.tensor([[2, 0, 1]])
        masked = labels.clone(); masked[:, -1] = -100
        # Teacher forcing shifts labels right; changing only the last label does
        # not alter any decoder input for this fixed-length forward pass.
        self.assertTrue(torch.equal(labels[:, :-1], masked[:, :-1]))
        full = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
        masked_loss = F.cross_entropy(logits.flatten(0, 1), masked.flatten()) * 2 / 3
        self.assertAlmostEqual(masked_loss.item(), (full[:2].sum() / 3).item())

    def test_max_bin_is_distinct_from_special_token(self):
        tokens = np.array([[[4095, 1, 2170]]])
        values = np.array([[[14., -14., .8]]])
        stats = token_diagnostics(tokens, values)
        self.assertEqual(stats["outside_vocab_rate"], 0.)
        self.assertAlmostEqual(stats["eos_rate"], 1/3)
        self.assertAlmostEqual(stats["max_bin_rate"], 1/3)


if __name__ == "__main__":
    unittest.main()
