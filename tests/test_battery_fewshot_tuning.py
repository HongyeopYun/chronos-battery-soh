import unittest
import numpy as np
import pandas as pd

from src.battery_fewshot_tuning import check_battery_split, select_label_windows, fixed_training_inputs


class FixedTuningProtocolTests(unittest.TestCase):
    def fixture(self):
        contexts = np.asarray([np.arange(s - 64, s) for s in range(65, 145)])
        targets = np.asarray([np.arange(s, s + 16) for s in range(65, 145)])
        w = {"context_cycle": contexts, "target_cycle": targets,
             "forecast_start": targets[:, 0], "context": 1 - contexts / 1000,
             "target": 1 - targets / 1000}
        f = pd.DataFrame({"cycle": np.arange(1, 161), "v": np.linspace(4, 3, 160)})
        return w, f

    def test_nonoverlapping_labels_selection_does_not_depend_on_soh(self):
        w, _ = self.fixture()
        selected = select_label_windows(w, 5)
        self.assertTrue(np.all(w["target_cycle"][selected][1:, 0] > w["target_cycle"][selected][:-1, -1]))
        w["target"][:] = -999
        np.testing.assert_array_equal(selected, select_label_windows(w, 5))
        with self.assertRaises(ValueError): select_label_windows(w, 6)

    def test_future_sensor_changes_cannot_enter_training(self):
        w, f = self.fixture()
        before = fixed_training_inputs(w, f, [0], ["v"])
        f.loc[f.cycle >= 65, "v"] = 999
        after = fixed_training_inputs(w, f, [0], ["v"])
        np.testing.assert_array_equal(before[0]["past_covariates"]["v"], after[0]["past_covariates"]["v"])
        self.assertTrue(np.isnan(after[0]["past_covariates"]["v"][64:]).all())
        np.testing.assert_allclose(after[0]["target"][64:], w["target"][0])

    def test_heldout_cell_cannot_enter_other_splits(self):
        check_battery_split(["4", "5"], ["6"], ["1", "2", "3"])
        with self.assertRaises(ValueError): check_battery_split(["4", "5"], ["6"], ["1", "5"])

    def test_actual_chronos_dataset_masks_future_and_fixes_loss_horizon(self):
        from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
        w, f = self.fixture()
        task = fixed_training_inputs(w, f, [0], ["v"])
        dataset = Chronos2Dataset(inputs=task, context_length=64, prediction_length=16,
                                 batch_size=2, output_patch_size=16, min_past=64, mode=DatasetMode.TRAIN)
        for _ in range(3):
            batch = next(iter(dataset))
            self.assertEqual(tuple(batch["context"].shape), (2, 64))
            np.testing.assert_allclose(batch["context"][0].numpy(), w["context"][0], atol=1e-7)
            np.testing.assert_allclose(batch["future_target"][0].numpy(), w["target"][0], atol=1e-7)
            self.assertTrue(np.isnan(batch["future_target"][1].numpy()).all())
            self.assertTrue(np.isnan(batch["future_covariates"].numpy()).all())

    def test_masked_sensor_rows_do_not_dilute_soh_loss_scale(self):
        import torch
        from src.c2_fewshot_trainer import target_loss_factor
        y = torch.ones(4, 16)
        masked = torch.cat([y, torch.full((8, 16), torch.nan)])
        self.assertEqual(float(target_loss_factor(y)), 1.)
        self.assertEqual(float(target_loss_factor(masked)), 3.)
        # An identical sum of SOH losses has the same mean after correction.
        self.assertAlmostEqual(float(torch.ones(4).sum() / 12 * target_loss_factor(masked)), 1.)


if __name__ == "__main__": unittest.main()
