"""CPU-only regression checks for robust validation and battery-balanced loss."""

import json
import math
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.soh_experiment import (
    ExperimentConfig,
    _start_run,
    _train_epoch,
    _training_loader,
    _validation_score,
    distribution_metrics,
    find_lobo,
    point_metrics,
    summarize_samples,
    verify_saved_run,
)
from src.soh_robust import _trajectory, policy_configs


class _ToyTokenizer:
    def context_input_transform(self, context):
        return context.long(), torch.ones_like(context, dtype=torch.bool), None

    def label_input_transform(self, target, scale):
        return target.long(), target >= 0


@dataclass(frozen=True)
class _ToyModelConfig:
    context_length: int = 1
    prediction_length: int = 2

    def create_tokenizer(self):
        return _ToyTokenizer()


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(2, 2))

    def disable_adapter(self):
        return nullcontext()

    def forward(self, input_ids, attention_mask, labels):
        logits = self.logits[input_ids[:, 0]].unsqueeze(1)
        return SimpleNamespace(logits=logits.expand(-1, labels.shape[1], -1))


class RobustTrainingTests(unittest.TestCase):
    def test_validation_averages_metrics_over_five_seeds_then_batteries(self):
        seeds = (101, 102, 103, 104, 105)
        config = replace(ExperimentConfig(), validation_seeds=seeds,
                         validation_samples=3, validation_batch_size=7,
                         inference_batch_size=2)
        validation = {
            "short": {"context": np.zeros((1, 2)), "target": np.zeros((1, 2))},
            "long": {"context": np.ones((4, 2)), "target": np.zeros((4, 2))},
        }
        shifts = dict(zip(seeds, (-.8, -.4, .1, .3, .9)))
        calls = []

        def predict(pipeline, context, config, seed, num_samples):
            calls.append((tuple(context[:, 0]), seed, num_samples, config.inference_batch_size))
            values = shifts[seed] + .5 * context[:, 0, None, None] + np.array([-.1, 0., .4])[None, :, None]
            return np.broadcast_to(values, (len(context), 3, 2)).copy()

        expected = {}
        for name, windows in validation.items():
            rows = []
            for seed in seeds:
                samples = predict(None, windows["context"], config, seed, 3)
                summary = summarize_samples(samples)
                rows.append({
                    "median_rmse": point_metrics(windows["target"], summary["median"])["rmse"],
                    "mean_rmse": point_metrics(windows["target"], summary["mean"])["rmse"],
                    **distribution_metrics(windows["target"], samples),
                })
            expected[name] = {key: np.mean([row[key] for row in rows]) for key in rows[0]}
        calls.clear()

        with patch("src.soh_experiment.predict_samples", side_effect=predict):
            macro, by_battery = _validation_score(object(), validation, config)

        self.assertCountEqual(calls, [((0, 1, 1, 1, 1), seed, 3, 7) for seed in seeds])
        self.assertEqual(set(by_battery), set(expected))
        for name, metrics in expected.items():
            self.assertEqual(set(by_battery[name]), set(metrics))
            for metric, value in metrics.items():
                self.assertAlmostEqual(by_battery[name][metric], value, places=12)
        for metric in expected["short"]:
            self.assertAlmostEqual(macro[metric],
                                   (expected["short"][metric] + expected["long"][metric]) / 2,
                                   places=12)
        # Unequal window counts must not silently reweight the two batteries.
        pooled = (expected["short"]["mean_rmse"] + 4 * expected["long"]["mean_rmse"]) / 5
        self.assertNotAlmostEqual(macro["mean_rmse"], pooled)

    def test_empty_validation_seeds_preserve_legacy_single_seed(self):
        config = replace(ExperimentConfig(), validation_seeds=(), validation_seed=731)
        windows = {"context": np.zeros((1, 2)), "target": np.zeros((1, 2))}
        with patch("src.soh_experiment.predict_samples", return_value=np.zeros((1, 3, 2))) as predict:
            macro, by_battery = _validation_score(None, {"battery": windows}, config)
        self.assertEqual(predict.call_count, 1)
        self.assertEqual(predict.call_args.args[3], 731)
        self.assertEqual(macro["mean_rmse"], 0)
        self.assertEqual(by_battery["battery"]["crps"], 0)

    def test_duplicate_validation_seeds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "validation_seeds"):
            replace(ExperimentConfig(), validation_seeds=(31415, 31416, 31415))

    def test_battery_balance_equalizes_total_ce_and_kl_contribution(self):
        # One battery has one window and another has three. Padding must not
        # dilute the per-example loss, and KL must use its cached tensor slot.
        datasets = [
            {"context": np.zeros((1, 1)), "target": np.array([[0, -1]])},
            {"context": np.ones((3, 1)), "target": np.array([[0, -1]] * 3)},
        ]
        for balanced in (False, True):
            for kl_weight in (0., .5):
                with self.subTest(balanced=balanced, kl_weight=kl_weight):
                    model = _ToyModel()
                    pipeline = SimpleNamespace(model=SimpleNamespace(model=model, config=_ToyModelConfig()))
                    config = replace(ExperimentConfig(), context_length=1, prediction_length=2,
                                     batch_size=4, battery_balance=balanced, kl_weight=kl_weight)
                    loader = _training_loader(pipeline, datasets, config, seed=42)
                    tensors = loader.dataset.tensors
                    self.assertEqual(len(tensors), 3 + bool(kl_weight) + balanced)
                    self.assertEqual(getattr(loader, "battery_balance", False), balanced)
                    if kl_weight:
                        torch.testing.assert_close(tensors[3], torch.full((4, 2, 2), -math.log(2)))
                    if balanced:
                        torch.testing.assert_close(tensors[-1], torch.tensor([2., 2/3, 2/3, 2/3]))
                        self.assertAlmostEqual(tensors[-1][0].item(), tensors[-1][1:].sum().item())

                    # The teacher cached above is uniform. These adapted logits
                    # create nonzero CE and KL gradients with known values.
                    with torch.no_grad():
                        model.logits.copy_(torch.tensor([[.4, -.4], [.4, -.4]]))
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
                    loss = _train_epoch(pipeline, loader, optimizer, seed=42, kl_weight=kl_weight)
                    probability = 1 / (1 + math.exp(-.8))
                    expected_loss = -math.log(probability) + kl_weight * (
                        -.5 * math.log(4 * probability * (1 - probability)))
                    self.assertAlmostEqual(loss, expected_loss, places=6)
                    single_gradient = probability - 1 + kl_weight * (probability - .5)
                    proportions = (.5, .5) if balanced else (.25, .75)
                    expected_gradient = torch.tensor([
                        [fraction * single_gradient, -fraction * single_gradient]
                        for fraction in proportions
                    ])
                    torch.testing.assert_close(model.logits.grad, expected_gradient, atol=1e-7, rtol=1e-5)

    def test_balanced_epoch_loss_does_not_depend_on_partial_last_batch(self):
        datasets = [
            {"context": np.zeros((1, 1)), "target": np.array([[0, -1]])},
            {"context": np.ones((3, 1)), "target": np.array([[0, -1]] * 3)},
        ]
        for batch_size in (3, 4):
            with self.subTest(batch_size=batch_size):
                model = _ToyModel()
                pipeline = SimpleNamespace(model=SimpleNamespace(model=model, config=_ToyModelConfig()))
                config = replace(ExperimentConfig(), context_length=1, prediction_length=2,
                                 batch_size=batch_size, battery_balance=True, kl_weight=.5)
                loader = _training_loader(pipeline, datasets, config, seed=19)
                with torch.no_grad():
                    model.logits.copy_(torch.tensor([[.4, -.4], [-.2, .2]]))
                optimizer = torch.optim.SGD(model.parameters(), lr=0.)
                loss = _train_epoch(pipeline, loader, optimizer, seed=19, kl_weight=.5)
                expected = []
                for log_odds in (.8, -.4):
                    probability = 1 / (1 + math.exp(-log_odds))
                    expected.append(-math.log(probability) - .25 * math.log(4 * probability * (1 - probability)))
                self.assertAlmostEqual(loss, sum(expected) / 2, places=6)

    def test_equal_battery_counts_preserve_exact_loss_gradients_and_updates(self):
        # Vary valid token counts and include a partial final batch. Equal battery
        # counts must use exactly the original reduction, including its padding.
        datasets = [
            {"context": np.zeros((2, 1)), "target": np.array([[0, -1], [1, 0]])},
            {"context": np.ones((2, 1)), "target": np.array([[1, -1], [0, 1]])},
        ]
        for kl_weight in (0., .5):
            with self.subTest(kl_weight=kl_weight):
                outcomes = []
                for balanced in (False, True):
                    model = _ToyModel()
                    pipeline = SimpleNamespace(model=SimpleNamespace(model=model, config=_ToyModelConfig()))
                    config = replace(ExperimentConfig(), context_length=1, prediction_length=2,
                                     batch_size=3, battery_balance=balanced, kl_weight=kl_weight)
                    loader = _training_loader(pipeline, datasets, config, seed=19)
                    if balanced:
                        self.assertTrue(torch.equal(loader.dataset.tensors[-1], torch.ones(4)))
                    with torch.no_grad():
                        model.logits.copy_(torch.tensor([[.4, -.4], [-.2, .2]]))
                    optimizer = torch.optim.SGD(model.parameters(), lr=.07)
                    loss = _train_epoch(pipeline, loader, optimizer, seed=19, kl_weight=kl_weight)
                    outcomes.append((loss, model.logits.grad.clone(), model.logits.detach().clone()))
                self.assertEqual(outcomes[0][0], outcomes[1][0])
                self.assertTrue(torch.equal(outcomes[0][1], outcomes[1][1]))
                self.assertTrue(torch.equal(outcomes[0][2], outcomes[1][2]))

    def test_generic_verifier_explicitly_rejects_special_experiment_kinds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kind in ("lora_ablation", "lora_robust"):
                with self.subTest(kind=kind):
                    (root / "manifest.json").write_text(json.dumps({"kind": kind}), encoding="utf-8")
                    with patch("src.soh_experiment.evaluation_signature") as signature, \
                         patch("src.soh_experiment.audit_saved_run") as audit:
                        with self.assertRaisesRegex(ValueError, "Special experiment kinds"):
                            verify_saved_run(root, root, ExperimentConfig())
                        signature.assert_not_called()
                        audit.assert_not_called()

    def test_special_runs_are_created_under_ablations_immediately(self):
        config = replace(ExperimentConfig(), device="cpu")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("src.soh_experiment.evaluation_signature", return_value={"model": "test"}), \
                 patch("src.soh_experiment._environment", return_value={"device": "cpu"}):
                for kind in ("lora_ablation", "lora_robust", "lora", "zeroshot"):
                    with self.subTest(kind=kind):
                        directory, manifest = _start_run(root, config, kind)
                        expected_parent = root / "results/runs"
                        if kind in ("lora_ablation", "lora_robust"):
                            expected_parent = expected_parent / "ablations"
                        self.assertEqual(directory.parent, expected_parent)
                        saved = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                        self.assertEqual(saved["kind"], kind)
                        self.assertEqual(saved["status"], "running")
                        self.assertEqual(saved["run_id"], manifest["run_id"])

    def test_trajectory_respects_epoch_caps_and_preserves_actual_best_checkpoint(self):
        config = replace(ExperimentConfig(), battery_ids=("trainA", "trainB", "heldout"),
                         device="cpu", max_epochs=20, validation_seeds=(101, 102))
        policies = {name: policy for name, policy in policy_configs(config).items()
                    if name != "balanced20"}
        state = {"epoch": 0}

        def save_checkpoint(path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_model.safetensors").write_text(str(state["epoch"]), encoding="utf-8")

        def train(*args):
            state["epoch"] += 1
            return float(state["epoch"])

        def validation_scores(pipeline, validation, current_config):
            self.assertEqual(set(validation), {"trainA", "trainB"})
            epoch = state["epoch"]
            value = 1. if epoch == 0 else .9
            if epoch == 8:
                value = .8
            elif 9 <= epoch < 14:
                value = .85
            elif epoch == 14:
                value = .6
            elif epoch > 14:
                value = .7
            score = {"median_rmse": value, "mean_rmse": value, "crps": value}
            return {seed: {battery: score.copy() for battery in validation}
                    for seed in current_config.validation_seeds}

        pipeline = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(save_pretrained=save_checkpoint)))
        # Deliberately omit the held-out series entirely: the selection trajectory
        # must be executable using training batteries alone.
        series = {"trainA": np.arange(168.), "trainB": np.arange(168.)}
        with tempfile.TemporaryDirectory() as temp, \
             patch("src.soh_experiment.fresh_pipeline", return_value=pipeline) as fresh, \
             patch("src.soh_experiment._training_loader", return_value=[]), \
             patch("src.soh_experiment._optimizer"), \
             patch("src.soh_experiment._train_epoch", side_effect=train) as train_epoch, \
             patch("src.soh_experiment.validation_seed_scores", side_effect=validation_scores), \
             patch("src.soh_experiment.clear_device_cache"), \
             patch("builtins.print"):
            root = Path(temp)
            directory = root / "results/runs/ablations/test_trajectory"
            record = _trajectory(root, directory, series, "heldout", 42, config, policies, "test")
            self.assertEqual(train_epoch.call_count, 20)
            self.assertEqual([row["epoch"] for row in record["history"]], list(range(21)))
            for name, expected_epoch in (("single_seed10", 8), ("multi_seed10", 8), ("multi_seed20", 14)):
                self.assertEqual(record["policies"][name]["selected_epoch"], expected_epoch)
                adapter = root / record["policies"][name]["adapter_path"] / "adapter_model.safetensors"
                self.assertEqual(adapter.read_text(encoding="utf-8"), str(expected_epoch))
            self.assertEqual(record["status"], "complete")
            fresh.reset_mock()
            resumed = _trajectory(root, directory, series, "heldout", 42, config, policies, "test")
            self.assertEqual(resumed, record)
            fresh.assert_not_called()

    def test_lobo_cache_rejects_ablation_kind_despite_matching_signature(self):
        config = ExperimentConfig()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "results/runs/20260924T000000Z_lora_ablation_example"
            run.mkdir(parents=True)
            manifest = {"kind": "lora_ablation", "status": "complete",
                        "evaluation_signature": {}, "config": asdict(config),
                        "artifact_sha256": {}, "zeroshot_run": "results/runs/baseline"}
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("src.soh_experiment.evaluation_signature", return_value={}), \
                 patch("src.soh_experiment.audit_saved_run", return_value={}), \
                 patch("src.soh_experiment.verify_zeroshot", return_value={}):
                self.assertIsNone(find_lobo(root, config))


if __name__ == "__main__":
    unittest.main()
