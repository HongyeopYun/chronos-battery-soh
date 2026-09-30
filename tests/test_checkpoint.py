import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.soh_experiment import ExperimentConfig
from src.soh_checkpoint import (
    CheckpointPolicy, distribution_scores, mask_output_eos,
    preserve_training_state, select_trial, train_trial, run_checkpoint_search,
)


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.))
        self.calls = 0

    def forward(self, input_ids, attention_mask, labels):
        if not torch.all(labels[:, -1] == -100):
            raise AssertionError("Output EOS entered the objective")
        if not torch.all(input_ids[:, -1] == 1):
            raise AssertionError("Encoder EOS changed")
        self.calls += 1
        return SimpleNamespace(loss=(self.weight - 1).square())

    def save_pretrained(self, path, **kwargs):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "weight.json").write_text(json.dumps(float(self.weight.detach())))


class CheckpointTests(unittest.TestCase):
    def test_eos_mask_preserves_numeric_gradients_and_inputs(self):
        labels = torch.tensor([[4, 5, 1]])
        masked = mask_output_eos(labels, 2, 1)
        logits = torch.randn(1, 3, 8, requires_grad=True)
        loss = F.cross_entropy(logits.flatten(0, 1), masked.flatten()) * 2 / 3
        expected = F.cross_entropy(logits[:, :2].flatten(0, 1), labels[:, :2].flatten(), reduction="sum") / 3
        actual_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertEqual(actual_grad[:, -1].abs().sum().item(), 0)
        torch.testing.assert_close(labels, torch.tensor([[4, 5, 1]]))
        torch.testing.assert_close(labels[:, :-1], masked[:, :-1])  # shifted decoder inputs
        with self.assertRaises(ValueError):
            mask_output_eos(torch.tensor([[4, 5, 7]]), 2, 1)

    def test_validation_restores_random_streams_and_training_mode(self):
        model = SmallModel().train()
        torch.manual_seed(13); np.random.seed(13); random.seed(13)
        cpu, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
        with preserve_training_state(model):
            torch.manual_seed(123); np.random.seed(123); random.seed(123)
            torch.rand(10); np.random.rand(10); random.random()
            model.eval()
        torch.testing.assert_close(torch.get_rng_state(), cpu)
        for actual, expected in zip(np.random.get_state(), numpy_state):
            np.testing.assert_equal(actual, expected)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(model.training)

    def test_step_budget_and_best_checkpoint_not_last_weights(self):
        model = SmallModel()
        pipeline = SimpleNamespace(model=SimpleNamespace(model=model, config=SimpleNamespace(eos_token_id=1)))
        cfg = ExperimentConfig(context_length=2, prediction_length=2, batch_size=1)
        policy = CheckpointPolicy(max_steps=5, eval_every_steps=2, validation_seeds=(10,))
        series = {b: np.linspace(1, .7, 20) for b in cfg.battery_ids}
        loader = DataLoader(TensorDataset(torch.tensor([[4, 1]] * 3), torch.ones(3, 2),
                                         torch.tensor([[4, 5, 1]] * 3)), batch_size=1)
        scores = iter([.3, .1, .2, .25])
        visited_weights = []

        def validate(pipeline, validation, config, policy):
            self.assertNotIn("B0005", validation)
            visited_weights.append(float(model.weight.detach()))
            return {"macro": {"crps": next(scores)}}, {}

        with TemporaryDirectory() as temporary, \
             patch("src.soh_checkpoint.fresh_pipeline", return_value=pipeline), \
             patch("src.soh_checkpoint._training_loader", return_value=loader), \
             patch("src.soh_checkpoint._optimizer", side_effect=lambda p, c: torch.optim.SGD(model.parameters(), lr=.1)), \
             patch("src.soh_checkpoint.validation_metrics", side_effect=validate):
            case, adapter = Path(temporary) / "case", Path(temporary) / "adapter"
            result = train_trial(series, "B0005", cfg, policy, 42, 1e-4, case, adapter)
            self.assertEqual(model.calls, 5)
            self.assertEqual(result["executed_steps"], 5)
            self.assertEqual([h["step"] for h in result["history"]], [0, 2, 4, 5])
            self.assertEqual(result["selected_step"], 2)
            self.assertFalse(result["refit_full_data"])
            saved_weight = json.loads((adapter / "weight.json").read_text())
            self.assertAlmostEqual(saved_weight, visited_weights[1])
            self.assertNotAlmostEqual(saved_weight, visited_weights[-1])
            self.assertEqual(result["executed_examples_seen"], 5)

    def test_learning_rate_selection_uses_validation_and_stable_ties(self):
        trials = [{"learning_rate": lr, "selected_validation": {"macro": {"crps": score}}, "test_rmse": test}
                  for lr, score, test in [(1e-5, .02, .001), (3e-5, .01, .2), (1e-4, .009999, .0001)]]
        self.assertEqual(select_trial(trials, 1e-5)["learning_rate"], 3e-5)

    def test_distribution_crps_matches_pairwise_score(self):
        values = np.array([[[.1, .4], [.3, .9], [.8, .6]], [[.5, .2], [.4, .6], [.7, .8]]])
        target = np.array([[.4, .5], [.3, .6]])
        expected = (np.abs(values - target[:, None]).mean(axis=1)
                    - .5 * np.abs(values[:, :, None] - values[:, None, :]).mean(axis=(1, 2))).mean()
        self.assertAlmostEqual(distribution_scores(target, values)["crps"], expected)

    def test_all_candidates_finish_before_any_selected_model_is_evaluated(self):
        cfg, policy = ExperimentConfig(), CheckpointPolicy()
        completed = []

        class ReachedEvaluation(Exception):
            pass

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "results" / "run"
            directory.mkdir(parents=True)

            def trial(series, test_battery, config, policy, seed, learning_rate, case, adapter):
                completed.append((test_battery, learning_rate))
                return {"test_battery": test_battery, "train_batteries": [b for b in cfg.battery_ids if b != test_battery],
                        "train_seed": seed, "learning_rate": learning_rate, "selected_step": 10,
                        "selected_validation": {"macro": {"crps": learning_rate}}, "adapter_path": str(adapter)}

            def reload(config, adapter):
                self.assertEqual(len(completed), 12)
                frozen = json.loads((directory / "selection_frozen.json").read_text())
                self.assertEqual(len(frozen["folds"]), 4)
                self.assertEqual(set(f["test_battery"] for f in frozen["folds"]), set(cfg.battery_ids))
                raise ReachedEvaluation

            with patch("src.soh_checkpoint.load_soh", return_value={}), \
                 patch("src.soh_checkpoint.find_zeroshot", return_value=root / "baseline"), \
                 patch("src.soh_checkpoint.verify_zeroshot"), \
                 patch("src.soh_checkpoint._start_run", return_value=(directory, {"status": "running"})), \
                 patch("src.soh_checkpoint._implementation", return_value={}), \
                 patch("src.soh_checkpoint.train_trial", side_effect=trial), \
                 patch("src.soh_checkpoint.reload_adapter", side_effect=reload), \
                 patch("src.soh_checkpoint._finish_run"), \
                 self.assertRaises(ReachedEvaluation):
                run_checkpoint_search(root, cfg, policy)


if __name__ == "__main__":
    unittest.main()
