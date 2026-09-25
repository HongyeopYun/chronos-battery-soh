"""Regression checks for protocol validation and checkpoint reproduction entry points."""
import copy
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_reproducibility as reproduction
from src import soh_ablation as ablation
from src import soh_robust as robust
from src.soh_experiment import ExperimentConfig, write_json


ROOT = Path(__file__).resolve().parents[1]


class RunProtocolTests(unittest.TestCase):
    def test_robust_invalid_controls_fail_before_baseline_or_training(self):
        invalid = [
            {"max_epochs": 10}, {"max_epochs": 30}, {"battery_balance": True},
            {"validation_seeds": []}, {"validation_seeds": [101]},
            {"selection_metric": "median_rmse"}, {"refit_full_data": True},
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "base.json", asdict(ExperimentConfig(device="cpu")))
            for overrides in invalid:
                with self.subTest(overrides=overrides):
                    write_json(root / "protocol.json", {
                        "base_config": "base.json",
                        "overrides": {"max_epochs": 20, "validation_seeds": [101, 102], **overrides},
                    })
                    with patch.object(robust.e, "find_zeroshot") as find, \
                         patch.object(robust.e, "fresh_pipeline") as fresh:
                        with self.assertRaisesRegex(ValueError, "Robust"):
                            robust.run_robust(root, Path("protocol.json"))
                        find.assert_not_called()
                        fresh.assert_not_called()

    def test_ablation_rejects_changes_to_shared_protocol_before_baseline(self):
        forbidden = {
            "num_samples": 1, "inference_seeds": [7], "inference_batch_size": 1,
            "prediction_length": 8, "context_length": 32, "temperature": .5,
            "selection_metric": "median_rmse", "refit_full_data": True,
            "validation_fraction": .3, "validation_seed": 123, "validation_samples": 8,
            "min_delta": 0., "revision": "other", "train_seeds": [7],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "base.json", asdict(ExperimentConfig(device="cpu")))
            for field, value in forbidden.items():
                with self.subTest(field=field):
                    write_json(root / "protocol.json", {
                        "base_config": "base.json", "search_train_seed": 42,
                        "candidates": [{"name": "bad", "overrides": {field: value}}],
                    })
                    with patch.object(ablation.e, "find_zeroshot") as find, \
                         patch.object(ablation.e, "fresh_pipeline") as fresh:
                        with self.assertRaisesRegex(ValueError, field):
                            ablation.run_ablation(root, Path("protocol.json"))
                        find.assert_not_called()
                        fresh.assert_not_called()

    def test_shipped_ablation_candidates_preserve_shared_fields(self):
        protocol = json.loads((ROOT / "configs/lora_ablation.json").read_text(encoding="utf-8"))
        base = ablation.e.load_config(ROOT, protocol["base_config"])
        candidates = ablation.candidate_configs(base, protocol["candidates"])
        self.assertEqual(list(candidates), [item["name"] for item in protocol["candidates"]])
        for item in protocol["candidates"]:
            self.assertEqual(candidates[item["name"]], replace(base, **item["overrides"]))

    def test_completed_compatibility_only_allows_reviewed_runner_change(self):
        saved = {"protocol": {"name": "fixed"}, "config": {"max_epochs": 20},
                 "source_sha256": {"core": "unchanged", "runner": next(iter(robust._COMPATIBLE_COMPLETED_RUNNERS))}}
        current = copy.deepcopy(saved)
        current["source_sha256"]["runner"] = "current"
        self.assertTrue(robust._completed_protocol_matches(saved, current))
        for section, key, value in (("source_sha256", "core", "changed"),
                                    ("source_sha256", "runner", "unknown"),
                                    ("config", "max_epochs", 10),
                                    ("protocol", "name", "different")):
            altered = copy.deepcopy(saved)
            altered[section][key] = value
            with self.subTest(section=section, key=key):
                self.assertFalse(robust._completed_protocol_matches(altered, current))

    def test_old_runner_cannot_resume_unfinished_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = ExperimentConfig(device="cpu", max_epochs=20, validation_seeds=(101, 102))
            write_json(root / "base.json", asdict(config))
            write_json(root / "protocol.json", {"base_config": "base.json", "overrides": {},
                       "search_train_seed": 42, "candidate_order": ["multi_seed10", "multi_seed20", "balanced20"]})
            run = root / "run"
            frozen = robust._frozen(root, Path("protocol.json"))
            frozen["source_sha256"]["runner"] = next(iter(robust._COMPATIBLE_COMPLETED_RUNNERS))
            write_json(run / "protocol.json", frozen)
            write_json(run / "manifest.json", {"kind": "lora_robust", "status": "running"})
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                robust.run_robust(root, Path("protocol.json"), run)

    def test_reproduction_uses_saved_config_for_all_lora_run_kinds(self):
        config = ExperimentConfig(device="cpu", max_epochs=20, validation_seeds=(101, 102))
        zero_config = replace(config, max_epochs=10)
        signature = {"same": "evaluation"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run, zero = root / "run", root / "zero"
            write_json(zero / "manifest.json", {"kind": "zeroshot", "config": asdict(zero_config),
                       "evaluation_signature": signature})
            for kind in ("lora", "lora_ablation", "lora_robust"):
                with self.subTest(kind=kind):
                    write_json(run / "manifest.json", {"kind": kind, "config": asdict(config),
                               "evaluation_signature": signature, "zeroshot_run": "zero",
                               "folds": [{"test_battery": "B0005", "train_seed": 42}]})
                    with patch.object(reproduction, "evaluation_signature", return_value=signature), \
                         patch.object(reproduction, "audit_saved_run", return_value={"status": "passed"}) as audit, \
                         patch.object(reproduction, "verify_saved_run", return_value={"status": "passed"}) as verify:
                        _, actual, actual_zero, actual_zero_config, checks = reproduction.prepare_run(run, root)
                        self.assertEqual(actual, config)
                        self.assertEqual(actual_zero, zero)
                        self.assertEqual(actual_zero_config, zero_config)
                        audit.assert_called_once_with(run, root, verify_adapters=True)
                        verify.assert_called_once_with(zero, root, zero_config)
                        self.assertEqual(checks["lora_artifacts"]["status"], "passed")
                        write_json(root / "wrong.json", asdict(replace(config, num_samples=1)))
                        with self.assertRaisesRegex(ValueError, "Explicit --config"):
                            reproduction.prepare_run(run, root, "wrong.json")

    def test_robust_cli_preserves_user_offline_preference(self):
        for value in (None, "0", "1"):
            with self.subTest(value=value), patch.dict(os.environ):
                if value is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = value
                runpy.run_path(str(ROOT / "scripts/run_robust_lora.py"), run_name="_test_import")
                self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), value)


if __name__ == "__main__":
    unittest.main()
