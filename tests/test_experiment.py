import unittest
import json
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.soh_experiment import (ExperimentConfig, audit_saved_run, distribution_metrics, find_project_root, load_config, load_soh,
                                make_windows, model_dtype, point_metrics, resolve_device,
                                masked_reference_kl, resolve_reference, select_epochs, summarize_samples,
                                temporal_split, validation_candidate_improves, verify_saved_run)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig()

    def test_cuda_request_fails_early_with_cpu_torch(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                resolve_device("cuda")

    def test_auto_prefers_cuda_and_canonicalizes_the_device(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=1), \
             patch("torch.cuda.device_count", return_value=2), \
             patch("torch.xpu.is_available", return_value=True):
            self.assertEqual(resolve_device("auto"), "cuda:1")
            self.assertEqual(resolve_device("cuda:0"), "cuda:0")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                resolve_device("cuda:2")

    def test_cpu_uses_float32(self):
        self.assertEqual(resolve_device("cpu"), "cpu")
        self.assertEqual(model_dtype("cpu:0"), torch.float32)

    def test_pre_ampere_cuda_uses_float32_even_with_bf16_emulation(self):
        with patch("torch.cuda.device", return_value=nullcontext()), \
             patch("torch.cuda.get_device_capability", return_value=(7, 5)), \
             patch("torch.cuda.is_bf16_supported", return_value=True):
            self.assertEqual(model_dtype("cuda:0"), torch.float32)

    def test_ampere_cuda_uses_bfloat16_when_supported(self):
        with patch("torch.cuda.device", return_value=nullcontext()), \
             patch("torch.cuda.get_device_capability", return_value=(8, 6)), \
             patch("torch.cuda.is_bf16_supported", return_value=True):
            self.assertEqual(model_dtype("cuda:0"), torch.bfloat16)
        with patch("torch.cuda.device", return_value=nullcontext()), \
             patch("torch.cuda.get_device_capability", return_value=(8, 6)), \
             patch("torch.cuda.is_bf16_supported", return_value=False):
            self.assertEqual(model_dtype("cuda:0"), torch.float32)

    def test_window_indices_are_causal(self):
        series = np.arange(168, dtype=float)
        windows = make_windows(series, 64, 16)
        self.assertEqual(windows["target"].shape, (89, 16))
        for x, y, start in zip(windows["context"], windows["target"], windows["forecast_start"]):
            self.assertEqual(x[-1] + 1, y[0])
            self.assertEqual(start, y[0] + 1)
        self.assertEqual(windows["target"][-1, -1], 167)

    def test_validation_targets_never_appear_in_training(self):
        for length in (168, 132):
            series = np.arange(length, dtype=float)
            train, valid, cutoff = temporal_split(series, self.config)
            seen = np.concatenate([train["context"].ravel(), train["target"].ravel()])
            self.assertLess(seen.max(), cutoff)
            self.assertGreaterEqual(valid["target"].min(), cutoff)
            self.assertEqual(np.intersect1d(seen, valid["target"]).size, 0)
            # Previously observed training values are legitimate validation context.
            self.assertLess(valid["context"][0, 0], cutoff)
            self.assertEqual(valid["forecast_start"][0], cutoff + 1)

    def test_median_interpolates_even_sample_count(self):
        samples = np.array([[[0.0, 4.0], [2.0, 8.0]]])
        summary = summarize_samples(samples)
        np.testing.assert_array_equal(summary["median"], [[1.0, 6.0]])
        self.assertTrue((summary["q10"] <= summary["median"]).all())
        self.assertTrue((summary["median"] <= summary["q90"]).all())

    def test_global_rmse_is_not_mean_window_rmse(self):
        target = np.zeros((2, 2))
        pred = np.array([[0, 0], [3, 4]])
        self.assertAlmostEqual(point_metrics(target, pred)["rmse"], 2.5)
        self.assertAlmostEqual(point_metrics(target, pred)["mae"], 1.75)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            point_metrics(np.zeros((2, 2)), np.zeros((2, 1)))
        with self.assertRaises(ValueError):
            summarize_samples(np.full((1, 2, 3), np.nan))
        with self.assertRaises(ValueError):
            temporal_split(np.arange(80), self.config)
        with self.assertRaises(ValueError):
            replace(self.config, inference_seeds=(42, 42))

    def test_real_data_splits_and_project_root(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(find_project_root(root), root)
        self.assertEqual(find_project_root(root / "notebooks"), root)
        config = load_config(root)
        if not (root / "data/raw/B0005.mat").exists():
            self.skipTest("NASA MAT files are not present")
        series = load_soh(root, config)
        for b, values in series.items():
            expected = 53 if b == "B0018" else 89
            self.assertEqual(len(make_windows(values, 64, 16)["target"]), expected)
            temporal_split(values, config)

    def test_sorted_crps_matches_pairwise_definition(self):
        rng = np.random.default_rng(7)
        target = rng.normal(size=(3, 4))
        for n in (1, 2, 9):
            samples = rng.normal(size=(3, n, 4))
            expected = (np.abs(samples - target[:, None]).mean()
                        - 0.5 * np.abs(samples[:, :, None] - samples[:, None, :]).mean())
            self.assertAlmostEqual(distribution_metrics(target, samples)["crps"], expected, places=12)

    def test_kl_masks_padding_and_preserves_gradients(self):
        logits = torch.tensor([[[1., 2., 3.], [9., -9., 0.]]], requires_grad=True)
        reference = torch.log_softmax(torch.zeros_like(logits), dim=-1)
        labels = torch.tensor([[1, -100]])
        loss = masked_reference_kl(logits, reference, labels)
        expected = (reference[0, 0].exp() * (reference[0, 0] - logits[0, 0].log_softmax(-1))).sum()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(logits.grad[0, 0].abs().sum(), 0)
        self.assertEqual(logits.grad[0, 1].abs().sum(), 0)
        self.assertAlmostEqual(masked_reference_kl(logits.detach(), logits.detach().log_softmax(-1), labels).item(), 0)

    def test_selection_rejects_distribution_and_point_regression(self):
        baseline = {"median_rmse": .02, "mean_rmse": .03, "crps": .015}
        good = {"median_rmse": .019, "mean_rmse": .025, "crps": .012}
        self.assertTrue(validation_candidate_improves(good, baseline, baseline, self.config))
        for bad in ({**good, "mean_rmse": .3}, {**good, "median_rmse": .025}, {**good, "crps": .016}):
            self.assertFalse(validation_candidate_improves(bad, baseline, baseline, self.config))

    def test_selected_checkpoint_is_not_overwritten_by_last_epoch(self):
        scores = [{"median_rmse": .02, "mean_rmse": .03, "crps": .015},
                  {"median_rmse": .019, "mean_rmse": .025, "crps": .012},
                  {"median_rmse": .018, "mean_rmse": .3, "crps": .02}]
        state = {"epoch": 0}
        saved = []
        model = SimpleNamespace(save_pretrained=lambda path: saved.append(state["epoch"]))
        pipeline = SimpleNamespace(model=SimpleNamespace(model=model))
        def train(*args):
            state["epoch"] += 1
            return 1.0
        with tempfile.TemporaryDirectory() as temp, \
             patch("src.soh_experiment.fresh_pipeline", return_value=pipeline), \
             patch("src.soh_experiment._training_loader", return_value=[]), \
             patch("src.soh_experiment._optimizer"), \
             patch("src.soh_experiment._train_epoch", side_effect=train), \
             patch("src.soh_experiment._validation_score", side_effect=[(s, {}) for s in scores]), \
             patch("src.soh_experiment.clear_device_cache"):
            # No held-out series exists in this mapping; selection cannot read it.
            chosen = select_epochs({"train": np.arange(168.)}, ["train"],
                                    replace(self.config, max_epochs=2), 42,
                                    Path(temp) / "selection.json", Path(temp) / "adapter")
            self.assertEqual(chosen, 1)
            self.assertEqual(saved, [0, 1])

    def test_legacy_absolute_paths_relocate_to_current_project(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            expected = root / "results/runs/example"
            self.assertEqual(resolve_reference(root, "results/runs/example"), expected)
            self.assertEqual(resolve_reference(root, r"Z:\old-user\repo\results\runs\example"), expected)
            self.assertEqual(resolve_reference(root, r"Z:\old-user\repo\models\example"), root / "models/example")

    def test_lora_cache_requires_linked_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = {"kind": "lora", "evaluation_signature": {},
                        "config": json.loads(json.dumps(asdict(self.config))),
                        "zeroshot_run": "results/runs/missing"}
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("src.soh_experiment.evaluation_signature", return_value={}), \
                 patch("src.soh_experiment.audit_saved_run") as audit:
                with self.assertRaises(FileNotFoundError):
                    verify_saved_run(directory, directory, self.config)
                audit.assert_not_called()

    def test_historical_artifact_audit_does_not_query_gpu(self):
        root = Path(__file__).resolve().parents[1]
        run = root / "results/runs/20260923T063628Z_lora_2df6cb"
        if not (root / "data/raw/B0005.mat").exists() or not run.exists():
            self.skipTest("Historical run/data not present")
        with patch("src.soh_experiment.runtime_info", side_effect=AssertionError("GPU queried")):
            report = audit_saved_run(run, root)
        self.assertEqual(report["metric_rows_verified"], 24)
        self.assertFalse(report["adapters_verified"])


if __name__ == "__main__":
    unittest.main()
