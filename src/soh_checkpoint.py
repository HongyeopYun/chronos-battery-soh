"""EOS-masked LoRA: validation-selected learning rate and saved checkpoint.

This module leaves the historical epoch/refit experiment and its signatures
intact. Test forecasting starts only after every fold's choices are frozen.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from src.soh_experiment import (
    load_config, load_soh, make_windows, temporal_split, fresh_pipeline,
    predict_samples, summarize_samples, point_metrics, reload_adapter,
    seed_everything, clear_device_cache, file_sha256, write_json,
    find_zeroshot, run_zeroshot, verify_zeroshot, verify_saved_run,
    save_forecast, metric_rows, _training_loader, _optimizer,
    _start_run, _finish_run, _write_tables, _paired_comparison,
)


@dataclass(frozen=True)
class CheckpointPolicy:
    learning_rates: tuple[float, ...] = (1e-5, 3e-5, 1e-4)
    max_steps: int = 100
    eval_every_steps: int = 10
    validation_samples: int = 32
    validation_seeds: tuple[int, ...] = (31415, 31416, 31417)
    min_delta: float = 1e-5

    def __post_init__(self):
        for name in ("learning_rates", "validation_seeds"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            values = getattr(self, name)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
        if any(not np.isfinite(x) or x <= 0 for x in self.learning_rates):
            raise ValueError("Learning rates must be finite and positive")
        if min(self.max_steps, self.eval_every_steps, self.validation_samples) < 1 or self.min_delta < 0:
            raise ValueError("Invalid step budget, validation samples or min_delta")


def load_checkpoint_config(root, path="configs/lora_checkpoint.json"):
    raw = json.loads((Path(root) / path).read_text(encoding="utf-8"))
    config = load_config(Path(root), raw.pop("base_config"))
    return config, CheckpointPolicy(**raw)


def distribution_scores(target, samples):
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[0::2] != target.shape or samples.shape[1] < 1:
        raise ValueError("Expected samples [windows, samples, horizon] and matching targets")
    if not np.isfinite(samples).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite forecasts or targets")
    n = samples.shape[1]
    order = np.arange(1, n + 1)[None, :, None]
    crps = np.abs(samples - target[:, None]).mean(axis=1) - (
        np.sort(samples, axis=1) * (2 * order - n - 1)).sum(axis=1) / n**2
    summary = summarize_samples(samples)
    return {"median_rmse": point_metrics(target, summary["median"])["rmse"],
            "median_mae": point_metrics(target, summary["median"])["mae"],
            "mean_rmse": point_metrics(target, summary["mean"])["rmse"],
            "crps": float(crps.mean()),
            "coverage80": float(((target >= summary["q10"]) & (target <= summary["q90"])).mean()),
            "width80": float((summary["q90"] - summary["q10"]).mean()),
            "negative_rate": float((samples < 0).mean()), "above2_rate": float((samples > 2).mean())}


def mask_output_eos(labels, horizon, eos_token_id):
    if labels.ndim != 2 or labels.shape[1] != horizon + 1:
        raise ValueError("Expected horizon numeric labels followed by EOS")
    if not torch.all(labels[:, -1] == eos_token_id):
        raise ValueError("Final label must be EOS")
    if not torch.all(labels[:, :horizon] >= 0):
        raise ValueError("This fixed-window protocol requires fully observed numeric targets")
    result = labels.clone()
    result[:, -1] = -100
    return result


@contextmanager
def preserve_training_state(model):
    cpu = torch.get_rng_state()
    xpu = torch.xpu.get_rng_state_all() if torch.xpu.is_available() else None
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    python_state, numpy_state = random.getstate(), np.random.get_state()
    training = model.training
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if xpu is not None:
            torch.xpu.set_rng_state_all(xpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        model.train(training)


def validation_metrics(pipeline, validation, config, policy):
    rows, forecasts = [], {}
    for battery, windows in validation.items():
        for seed in policy.validation_seeds:
            samples = predict_samples(pipeline, windows["context"], config, seed, policy.validation_samples)
            rows.append({"battery": battery, "inference_seed": seed,
                         **distribution_scores(windows["target"], samples)})
            forecasts[f"{battery}_infer{seed}"] = (windows, samples)
    frame = pd.DataFrame(rows)
    columns = [c for c in frame.columns if c not in ("battery", "inference_seed")]
    by_battery = frame.groupby("battery")[columns].mean()
    return {"macro": by_battery.mean().to_dict(), "by_battery": by_battery.to_dict("index"),
            "by_seed_and_battery": rows}, forecasts


def _split_training(series, train_batteries, config):
    train, validation, splits = [], {}, {}
    for battery in train_batteries:
        training, valid, cutoff = temporal_split(series[battery], config)
        train.append(training)
        validation[battery] = valid
        splits[battery] = {"cutoff_zero_based": cutoff,
                           "training_windows": len(training["target"]),
                           "validation_windows": len(valid["target"]),
                           "last_training_target_cycle": int(training["forecast_start"][-1] + config.prediction_length - 1),
                           "first_validation_target_cycle": int(valid["forecast_start"][0])}
    return train, validation, splits


def is_improvement(score, best, min_delta):
    return best is None or score < best - min_delta


def train_trial(series, test_battery, config, policy, seed, learning_rate, case, adapter):
    """Save each improved adapter; never refit or return the final live model."""
    case, adapter = Path(case), Path(adapter)
    case.mkdir(parents=True, exist_ok=True)
    cfg = replace(config, learning_rate=learning_rate)
    train_batteries = [b for b in cfg.battery_ids if b != test_battery]
    train, validation, splits = _split_training(series, train_batteries, cfg)
    pipeline = fresh_pipeline(cfg, seed, lora=True)
    optimizer = None
    try:
        model = pipeline.model.model
        loader = _training_loader(pipeline, train, cfg, seed)
        optimizer = _optimizer(pipeline, cfg)
        parameters = [p for p in model.parameters() if p.requires_grad]
        device = next(model.parameters()).device
        record = {"status": "running", "test_battery": test_battery, "train_batteries": train_batteries,
                  "train_seed": seed, "learning_rate": learning_rate, "splits": splits,
                  "optimizer_steps_per_epoch": len(loader), "batch_size": cfg.batch_size,
                  "gradient_accumulation_steps": 1, "policy": asdict(policy),
                  "exclude_output_eos_loss": True, "refit_full_data": False,
                  "loss_rule": "Mask final EOS label; multiply forecast-only mean CE by horizon/(horizon+1)",
                  "adapter_path": str(adapter.resolve()), "history": []}
        best, step, epoch, examples_seen = None, 0, 0, 0
        loss_sum, loss_count = 0., 0

        def evaluate():
            nonlocal best, loss_sum, loss_count
            with preserve_training_state(model):
                score, forecasts = validation_metrics(pipeline, validation, cfg, policy)
                crps = score["macro"]["crps"]
                if not np.isfinite(crps):
                    raise FloatingPointError("Nonfinite validation CRPS")
                improved = is_improvement(crps, best, policy.min_delta)
                entry = {"step": step, "epoch": epoch, "examples_seen": examples_seen,
                         "train_loss_since_validation": loss_sum / loss_count if loss_count else None,
                         "validation": score, "selected_at_this_check": improved}
                record["history"].append(entry)
                if improved:
                    best = crps
                    model.save_pretrained(adapter, save_embedding_layers=False)
                    for name, (windows, samples) in forecasts.items():
                        save_forecast(case / "selected_validation" / f"{name}.npz", windows, samples)
                    record.update(selected_step=step, selected_epoch=epoch,
                                  selected_examples_seen=examples_seen, selected_validation=score)
                write_json(case / "history.json", record)
                print(f"  {test_battery} lr={learning_rate:g} step={step}/{policy.max_steps} "
                      f"val_CRPS={crps:.7f} best_step={record['selected_step']}", flush=True)
            loss_sum, loss_count = 0., 0

        evaluate()  # Step zero is eligible; a worse fine-tune need not be selected.
        while step < policy.max_steps:
            epoch += 1
            seed_everything(seed + epoch)
            model.train()
            for ids, mask, labels in loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                labels = mask_output_eos(labels, cfg.prediction_length, pipeline.model.config.eos_token_id)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = outputs.loss * cfg.prediction_length / (cfg.prediction_length + 1)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                step += 1
                examples_seen += len(ids)
                loss_sum += loss.item() * len(ids)
                loss_count += len(ids)
                if step % policy.eval_every_steps == 0 or step == policy.max_steps:
                    evaluate()
                if step == policy.max_steps:
                    break
        record.update(status="complete", executed_steps=step, executed_examples_seen=examples_seen)
        write_json(case / "history.json", record)
        record["adapter_sha256"] = {p.name: file_sha256(p) for p in adapter.iterdir() if p.is_file()}
        record["artifact_sha256"] = {str(p.relative_to(case)): file_sha256(p)
                                     for p in case.rglob("*") if p.is_file() and p.name != "complete.json"}
        write_json(case / "complete.json", record)
        return record
    finally:
        del optimizer, pipeline
        clear_device_cache()


def verify_trial(case):
    record = json.loads((Path(case) / "complete.json").read_text(encoding="utf-8"))
    if record["status"] != "complete" or record["refit_full_data"]:
        raise ValueError("Trial not complete or uses an incompatible refit protocol")
    for rel, digest in record["artifact_sha256"].items():
        if file_sha256(Path(case) / rel) != digest:
            raise ValueError(f"Trial artifact changed: {rel}")
    for name, digest in record["adapter_sha256"].items():
        if file_sha256(Path(record["adapter_path"]) / name) != digest:
            raise ValueError(f"Trial adapter changed: {name}")
    return record


def select_trial(records, min_delta):
    best = None
    for record in records:
        score = record["selected_validation"]["macro"]["crps"]
        if best is None or is_improvement(score, best["selected_validation"]["macro"]["crps"], min_delta):
            best = record
    if best is None:
        raise ValueError("No candidate trials")
    return best


def _implementation(root):
    return {p: file_sha256(Path(root) / p) for p in ("src/soh_experiment.py", "src/soh_checkpoint.py")}


def run_checkpoint_search(root, config, policy, resume=None):
    root = Path(root).resolve()
    series = load_soh(root, config)
    if resume is None:
        baseline = find_zeroshot(root, config)
        if baseline is None:
            baseline = run_zeroshot(root, config)
        verify_zeroshot(baseline, root, config)
        directory, manifest = _start_run(root, config, "checkpointlora")
        manifest.update(training_policy=asdict(policy), implementation_sha256=_implementation(root),
            zeroshot_run=str(baseline.resolve()),
            protocol="Output EOS loss excluded; train temporal prefixes only; select LR and step using macro validation CRPS; no refit; all choices frozen before held-out forecasting",
            epoch_and_patience_note="Base max_epochs/patience are historical fields; this run uses only the explicit optimizer-step budget; no pruning or early termination")
    else:
        directory = Path(resume).resolve()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        _verify_identity(manifest, root, config, policy)
        if manifest["status"] == "complete":
            verify_checkpoint_run(directory, root, config, policy)
            return directory
        baseline = Path(manifest["zeroshot_run"])
        verify_zeroshot(baseline, root, config)
        manifest["status"] = "running"
        manifest.pop("error", None)
    write_json(directory / "manifest.json", manifest)
    try:
        choices, trials = [], []
        # No held-out forecast is generated anywhere in this selection phase.
        for test_battery in config.battery_ids:
            for seed in config.train_seeds:
                candidates = []
                for index, lr in enumerate(policy.learning_rates):
                    name = f"{test_battery}_train{seed}_candidate{index}"
                    case = directory / "trials" / name
                    adapter = root / "models" / directory.name / name
                    if (case / "complete.json").exists():
                        record = verify_trial(case)
                        print(f"Reusing complete trial: {name}", flush=True)
                    else:
                        print(f"TRAIN {name} lr={lr:g}", flush=True)
                        record = train_trial(series, test_battery, config, policy, seed, lr, case, adapter)
                    candidates.append(record)
                    trials.append({"test_battery": test_battery, "train_seed": seed, "learning_rate": lr,
                                   "selected_step": record["selected_step"],
                                   **record["selected_validation"]["macro"], "trial_path": str(case.resolve())})
                selected = select_trial(candidates, policy.min_delta)
                choices.append(selected)
                print(f"SELECTED {test_battery} lr={selected['learning_rate']:g} step={selected['selected_step']}", flush=True)
        pd.DataFrame(trials).to_csv(directory / "validation_candidates.csv", index=False)
        frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(), "selection_metric": "validation_macro_crps",
                  "tie_rule": "Keep earlier step / earlier learning-rate candidate unless CRPS drops by more than min_delta",
                  "folds": choices}
        write_json(directory / "selection_frozen.json", frozen)
        manifest["folds"] = choices
        write_json(directory / "manifest.json", manifest)
        rows, horizons, window_rows, distributions, replay_checks = [], [], [], [], []
        for fold in choices:
            test_battery, seed = fold["test_battery"], fold["train_seed"]
            cfg = replace(config, learning_rate=fold["learning_rate"])
            pipeline = reload_adapter(cfg, Path(fold["adapter_path"]))
            try:
                _, validation, _ = _split_training(series, fold["train_batteries"], cfg)
                replay_score, replay = validation_metrics(pipeline, validation, cfg, policy)
                case = next(Path(r["trial_path"]) for r in trials if r["test_battery"] == test_battery
                            and r["train_seed"] == seed and r["learning_rate"] == fold["learning_rate"])
                for name, (_, samples) in replay.items():
                    with np.load(case / "selected_validation" / f"{name}.npz", allow_pickle=False) as saved:
                        np.testing.assert_array_equal(samples, saved["samples"])
                np.testing.assert_allclose(replay_score["macro"]["crps"], fold["selected_validation"]["macro"]["crps"], rtol=0, atol=1e-12)
                replay_checks.append({"test_battery": test_battery, "train_seed": seed,
                                      "selected_step": fold["selected_step"], "validation_arrays_identical": len(replay)})
                write_json(directory / "checkpoint_replay.json", replay_checks)
                windows = make_windows(series[test_battery], cfg.context_length, cfg.prediction_length)
                for inference_seed in cfg.inference_seeds:
                    samples = predict_samples(pipeline, windows["context"], cfg, inference_seed)
                    save_forecast(directory / "predictions" / f"{test_battery}_train{seed}_infer{inference_seed}.npz", windows, samples)
                    r, h, w = metric_rows(windows, summarize_samples(samples), "LoRA", test_battery, seed, inference_seed)
                    rows.extend(r); horizons.extend(h); window_rows.extend(w)
                    distributions.append({"model": "SelectedLoRA", "battery": test_battery, "train_seed": seed,
                        "inference_seed": inference_seed, **distribution_scores(windows["target"], samples)})
                    with np.load(baseline / "predictions" / f"{test_battery}_infer{inference_seed}.npz", allow_pickle=False) as zero:
                        np.testing.assert_array_equal(zero["target"], windows["target"])
                        np.testing.assert_array_equal(zero["context"], windows["context"])
                        distributions.append({"model": "ZeroShot", "battery": test_battery, "train_seed": seed,
                            "inference_seed": inference_seed, **distribution_scores(windows["target"], zero["samples"])})
                    print(f"TEST {test_battery} train={seed} infer={inference_seed} median_RMSE={r[0]['rmse']:.7f}", flush=True)
                _write_tables(directory, rows, horizons, window_rows)
                pd.DataFrame(distributions).to_csv(directory / "distribution_metrics.csv", index=False)
            finally:
                del pipeline
                clear_device_cache()
        _paired_comparison(directory, baseline)
        _finish_run(directory, manifest)
    except Exception as error:
        _finish_run(directory, manifest, error)
        raise
    return directory


def _verify_identity(manifest, root, config, policy):
    if manifest["kind"] != "checkpointlora" or manifest["config"] != json.loads(json.dumps(asdict(config))):
        raise ValueError("Different experiment kind or base config")
    if manifest["training_policy"] != json.loads(json.dumps(asdict(policy))):
        raise ValueError("Different checkpoint / search policy")
    if manifest["implementation_sha256"] != _implementation(root):
        raise ValueError("Checkpoint search implementation changed")


def verify_checkpoint_run(directory, root, config, policy):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    _verify_identity(manifest, root, config, policy)
    checks = verify_saved_run(directory, Path(root), config)
    verify_saved_run(Path(manifest["zeroshot_run"]), Path(root), config)
    for case in (directory / "trials").iterdir():
        verify_trial(case)
    values = pd.read_csv(directory / "distribution_metrics.csv")
    for row in values.to_dict("records"):
        if row["model"] == "ZeroShot":
            path = Path(manifest["zeroshot_run"]) / "predictions" / f"{row['battery']}_infer{row['inference_seed']}.npz"
        else:
            path = directory / "predictions" / f"{row['battery']}_train{row['train_seed']}_infer{row['inference_seed']}.npz"
        with np.load(path, allow_pickle=False) as saved:
            recalculated = distribution_scores(saved["target"], saved["samples"])
        for key, value in recalculated.items():
            np.testing.assert_allclose(value, row[key], rtol=0, atol=1e-12)
    checks["distribution_rows_verified"] = len(values)
    checks["trial_count"] = len(list((directory / "trials").iterdir()))
    return checks


def find_checkpoint_run(root, config, policy):
    for path in sorted((Path(root) / "results/runs").glob("*_checkpointlora_*/manifest.json"), reverse=True):
        try:
            verify_checkpoint_run(path.parent, root, config, policy)
            return path.parent
        except (ValueError, KeyError, FileNotFoundError, AssertionError):
            continue
    return None
