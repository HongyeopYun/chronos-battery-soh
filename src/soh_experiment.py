"""Shared zero-shot / LoRA LOBO implementation, used by both notebooks and CLI.

Targets are Q_k / Q_1. Evaluation is rolling-origin forecasting with observed
contexts. Temporal validation is confined to the outer training batteries.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str = "amazon/chronos-t5-base"
    revision: str = "ad294eaacead15db499b740ea4122266dd2a81a2"
    battery_ids: tuple[str, ...] = ("B0005", "B0006", "B0007", "B0018")
    context_length: int = 64
    prediction_length: int = 16
    device: str = "auto"
    inference_batch_size: int = 16
    num_samples: int = 100
    inference_seeds: tuple[int, ...] = (42, 43, 44)
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    train_seeds: tuple[int, ...] = (42,)
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_epochs: int = 10
    validation_fraction: float = 0.2
    validation_samples: int = 32
    validation_seed: int = 31415
    patience: int = 3
    min_delta: float = 1e-5

    def __post_init__(self):
        for name in ("battery_ids", "inference_seeds", "train_seeds"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            if not getattr(self, name) or len(set(getattr(self, name))) != len(getattr(self, name)):
                raise ValueError(f"{name} must be nonempty and contain unique values")
        if len(self.battery_ids) < 3:
            raise ValueError("LOBO requires at least three batteries")
        for name in ("context_length", "prediction_length", "inference_batch_size",
                     "num_samples", "batch_size", "lora_rank", "validation_samples", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.validation_fraction < 1 or self.max_epochs < 0:
            raise ValueError("Invalid validation_fraction or max_epochs")
        if not self.revision or self.learning_rate <= 0 or self.min_delta < 0:
            raise ValueError("A pinned revision, positive learning_rate and nonnegative min_delta are required")


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data" / "raw").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot locate project from {current}")


def load_config(root: Path, path: str | Path = "configs/experiment.json") -> ExperimentConfig:
    return ExperimentConfig(**json.loads((root / path).read_text(encoding="utf-8")))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_soh(root: Path, config: ExperimentConfig) -> dict[str, np.ndarray]:
    series = {}
    for battery in config.battery_ids:
        cycles = loadmat(root / "data" / "raw" / f"{battery}.mat", simplify_cells=True)[battery]["cycle"]
        capacity = np.asarray([c["data"]["Capacity"] for c in cycles if c["type"] == "discharge"], dtype=np.float64)
        if capacity.ndim != 1 or len(capacity) < config.context_length + config.prediction_length:
            raise ValueError(f"{battery}: insufficient or invalid capacity data")
        if not np.isfinite(capacity).all() or not (capacity > 0).all():
            raise ValueError(f"{battery}: nonfinite or nonpositive capacity")
        series[battery] = capacity / capacity[0]
    return series


def make_windows(series: np.ndarray, context_length: int, prediction_length: int) -> dict[str, np.ndarray]:
    series = np.asarray(series, dtype=np.float64)
    total = context_length + prediction_length
    if series.ndim != 1 or len(series) < total or not np.isfinite(series).all():
        raise ValueError("Series must be finite, one-dimensional and long enough for a window")
    windows = np.lib.stride_tricks.sliding_window_view(series, total)
    return {"context": windows[:, :context_length].copy(),
            "target": windows[:, context_length:].copy(),
            "forecast_start": np.arange(context_length + 1, len(series) - prediction_length + 2)}


def temporal_split(series: np.ndarray, config: ExperimentConfig):
    """Split by target time BEFORE selecting windows; contexts may use observed past.

    cutoff is zero-based, the first validation target. Training windows end before
    cutoff. Validation targets start at/after cutoff; straddling targets are omitted.
    """
    cutoff = int(np.floor(len(series) * (1 - config.validation_fraction)))
    windows = make_windows(series, config.context_length, config.prediction_length)
    start = windows["forecast_start"] - 1
    train_mask = start + config.prediction_length <= cutoff
    validation_mask = start >= cutoff
    if not train_mask.any() or not validation_mask.any():
        raise ValueError(f"Empty temporal split at cutoff {cutoff}; change lengths or validation fraction")
    take = lambda mask: {k: v[mask].copy() for k, v in windows.items()}
    return take(train_mask), take(validation_mask), cutoff


def point_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or target.ndim != 2 or not target.size:
        raise ValueError("Targets and predictions must have matching nonempty [windows, horizon] shapes")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite target or prediction")
    error = prediction.astype(np.float64) - target.astype(np.float64)
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(abs(error)))}


def summarize_samples(samples: np.ndarray) -> dict[str, np.ndarray]:
    samples = np.asarray(samples)
    if samples.ndim != 3 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Samples must be finite [windows, samples, horizon]")
    return {"median": np.quantile(samples, 0.5, axis=1),
            "mean": samples.mean(axis=1, dtype=np.float64),
            "q10": np.quantile(samples, 0.1, axis=1),
            "q90": np.quantile(samples, 0.9, axis=1)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Includes XPU in the installed PyTorch.


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def clear_device_cache() -> None:
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fresh_pipeline(config: ExperimentConfig, seed: int, lora: bool = False):
    seed_everything(seed)  # MUST precede model and adapter creation.
    device = resolve_device(config.device)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    pipeline = ChronosPipeline.from_pretrained(config.model_id, revision=config.revision,
                                                device_map=device, dtype=dtype)
    if config.context_length > pipeline.model.config.context_length:
        raise ValueError("Context exceeds pretrained model capacity")
    if config.prediction_length > pipeline.model.config.prediction_length:
        raise ValueError("Prediction exceeds pretrained horizon")
    if lora:
        pipeline.model.model = get_peft_model(pipeline.model.model, LoraConfig(
            r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            target_modules=["q", "v"], bias="none", task_type=TaskType.SEQ_2_SEQ_LM))
        trainable = [(n, p) for n, p in pipeline.model.model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in n for n, _ in trainable):
            raise RuntimeError("Unexpected trainable parameters")
    pipeline.model.model.eval()
    return pipeline


def predict_samples(pipeline, context: np.ndarray, config: ExperimentConfig,
                    seed: int, num_samples: int | None = None) -> np.ndarray:
    seed_everything(seed)
    pipeline.model.model.eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(context), config.inference_batch_size):
            prediction = pipeline.predict(
                torch.tensor(context[start:start + config.inference_batch_size], dtype=torch.float32),
                prediction_length=config.prediction_length,
                num_samples=num_samples or config.num_samples, temperature=config.temperature,
                top_k=config.top_k, top_p=config.top_p)
            batches.append(prediction.cpu().numpy())
    samples = np.concatenate(batches)
    expected = (len(context), num_samples or config.num_samples, config.prediction_length)
    if samples.shape != expected or not np.isfinite(samples).all():
        raise RuntimeError(f"Invalid forecasts: {samples.shape}, expected {expected}")
    return samples


def _environment(root: Path, device: str) -> dict:
    versions = {}
    for package in ("torch", "chronos-forecasting", "transformers", "peft", "accelerate", "numpy", "scipy", "pandas"):
        versions[package] = importlib.metadata.version(package)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": versions, "device": device, "git_commit": commit,
            "source_sha256": {str(p.relative_to(root)): file_sha256(p)
                              for p in sorted((root / "src").glob("*.py"))},
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "reproducibility_note": "Seeds fixed; bitwise equality across devices or library versions is not guaranteed."}


def evaluation_signature(root: Path, config: ExperimentConfig) -> dict:
    fields = ("model_id", "revision", "battery_ids", "context_length", "prediction_length",
              "inference_batch_size", "num_samples", "inference_seeds", "temperature", "top_k", "top_p")
    raw = asdict(config)
    signature = {k: raw[k] for k in fields}
    signature.update(schema_version=SCHEMA_VERSION, normalization="Q_k / Q_1", aggregation="numpy linear quantile 0.5; sample mean",
                     device=resolve_device(config.device),
                     data_sha256={b: file_sha256(root / "data" / "raw" / f"{b}.mat") for b in config.battery_ids},
                     versions={p: importlib.metadata.version(p) for p in ("torch", "chronos-forecasting", "transformers", "peft", "numpy")},
                     source_sha256=file_sha256(Path(__file__)))
    return json.loads(json.dumps(signature))


def _start_run(root: Path, config: ExperimentConfig, kind: str):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{kind}_{uuid4().hex[:6]}"
    directory = root / "results" / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "kind": kind, "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
                "evaluation_signature": evaluation_signature(root, config),
                "environment": _environment(root, resolve_device(config.device)),
                "protocol": "Observed 64-cycle contexts, rolling-origin 16-cycle forecasts; overlapping windows are not independent."}
    write_json(directory / "manifest.json", manifest)
    return directory, manifest


def _finish_run(directory: Path, manifest: dict, error: Exception | None = None):
    manifest["status"] = "failed" if error else "complete"
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    if error:
        manifest["error"] = f"{type(error).__name__}: {error}"
    else:
        manifest["artifact_sha256"] = {str(p.relative_to(directory)): file_sha256(p)
                                        for p in directory.rglob("*") if p.is_file() and p.name != "manifest.json"}
    write_json(directory / "manifest.json", manifest)


def save_forecast(path: Path, windows: dict, samples: np.ndarray):
    summary = summarize_samples(samples)
    point_metrics(windows["target"], summary["median"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **windows, samples=samples, **summary)


def metric_rows(windows: dict, prediction: dict, model: str, battery: str,
                train_seed: int = -1, inference_seed: int = -1):
    rows, horizons, window_rows = [], [], []
    target = windows["target"]
    for estimator in ("median", "mean"):
        if estimator not in prediction:
            continue
        pred = prediction[estimator]
        tags = {"model": model, "battery": battery, "train_seed": train_seed,
                "inference_seed": inference_seed, "estimator": estimator}
        row = {**tags, **point_metrics(target, pred), "n_windows": len(target)}
        if "q10" in prediction:
            row.update(coverage80=float(np.mean((target >= prediction["q10"]) & (target <= prediction["q90"]))),
                       width80=float(np.mean(prediction["q90"] - prediction["q10"])))
        rows.append(row)
        for h in range(target.shape[1]):
            horizons.append({**tags, "horizon": h + 1, **point_metrics(target[:, h:h+1], pred[:, h:h+1])})
        error = pred - target
        for i, origin in enumerate(windows["forecast_start"]):
            window_rows.append({**tags, "forecast_start": int(origin),
                                "rmse": float(np.sqrt(np.mean(error[i] ** 2))), "mae": float(np.mean(abs(error[i])))})
    return rows, horizons, window_rows


def _write_tables(directory: Path, rows, horizons, windows):
    metrics = pd.DataFrame(rows)
    metrics.to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(horizons).to_csv(directory / "horizon_metrics.csv", index=False)
    pd.DataFrame(windows).to_csv(directory / "window_metrics.csv", index=False)
    # Equal battery weights, then average repeated runs. Std is over run-level
    # macro scores, not a confidence interval based on overlapping windows.
    keys = ["model", "estimator", "train_seed", "inference_seed"]
    macro = metrics.groupby(keys, as_index=False)[["rmse", "mae"]].mean()
    macro.to_csv(directory / "macro_by_run.csv", index=False)
    summary = macro.groupby(["model", "estimator"]).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"), n_runs=("rmse", "size")).reset_index()
    summary.to_csv(directory / "summary.csv", index=False)
    return summary


def _baselines(windows):
    x, y = windows["context"], windows["target"]
    last = np.repeat(x[:, -1:], y.shape[1], axis=1)
    linear = np.array([np.polyval(np.polyfit(np.arange(x.shape[1]), v, 1),
                       np.arange(x.shape[1], x.shape[1] + y.shape[1])) for v in x])
    return {"LastValue": last, "Linear": linear}


def run_zeroshot(root: Path, config: ExperimentConfig) -> Path:
    root = Path(root).resolve()
    series = load_soh(root, config)
    directory, manifest = _start_run(root, config, "zeroshot")
    pipeline = None
    try:
        rows, horizons, window_rows = [], [], []
        pipeline = fresh_pipeline(config, config.inference_seeds[0])
        for battery in config.battery_ids:
            windows = make_windows(series[battery], config.context_length, config.prediction_length)
            for model, pred in _baselines(windows).items():
                r, h, w = metric_rows(windows, {"median": pred}, model, battery)
                rows.extend(r); horizons.extend(h); window_rows.extend(w)
            for seed in config.inference_seeds:
                samples = predict_samples(pipeline, windows["context"], config, seed)
                save_forecast(directory / "predictions" / f"{battery}_infer{seed}.npz", windows, samples)
                r, h, w = metric_rows(windows, summarize_samples(samples), "ZeroShot", battery, inference_seed=seed)
                rows.extend(r); horizons.extend(h); window_rows.extend(w)
                print(f"Zero-shot {battery} seed={seed}: median RMSE={r[0]['rmse']:.6f}", flush=True)
        _write_tables(directory, rows, horizons, window_rows)
        _finish_run(directory, manifest)
    except Exception as error:
        _finish_run(directory, manifest, error)
        raise
    finally:
        del pipeline
        clear_device_cache()
    return directory


def verify_zeroshot(directory: Path, root: Path, config: ExperimentConfig):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["kind"] != "zeroshot" or manifest["status"] != "complete":
        raise ValueError("Baseline run is not a completed zero-shot experiment")
    if manifest["evaluation_signature"] != evaluation_signature(root, config):
        raise ValueError("Baseline data, model, environment, code or inference settings do not match")
    for relative, digest in manifest["artifact_sha256"].items():
        path = directory / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Baseline artifact missing or changed: {path}")
    return manifest


def find_zeroshot(root: Path, config: ExperimentConfig) -> Path | None:
    for path in sorted((root / "results" / "runs").glob("*_zeroshot_*/manifest.json"), reverse=True):
        try:
            verify_zeroshot(path.parent, root, config)
            return path.parent
        except (ValueError, KeyError, FileNotFoundError):
            continue
    return None


def _training_loader(pipeline, datasets: list[dict], config: ExperimentConfig, seed: int):
    context = torch.tensor(np.concatenate([d["context"] for d in datasets]), dtype=torch.float32)
    target = torch.tensor(np.concatenate([d["target"] for d in datasets]), dtype=torch.float32)
    tokenizer = replace(pipeline.model.config, context_length=config.context_length,
                        prediction_length=config.prediction_length).create_tokenizer()
    ids, mask, scale = tokenizer.context_input_transform(context)
    labels, label_mask = tokenizer.label_input_transform(target, scale)
    labels[~label_mask] = -100
    dataset = TensorDataset(ids, mask, labels)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def _optimizer(pipeline, config):
    return torch.optim.AdamW([p for p in pipeline.model.model.parameters() if p.requires_grad],
                              lr=config.learning_rate, weight_decay=config.weight_decay)


def _train_epoch(pipeline, loader, optimizer, seed: int):
    # Independent of how many validation forecasts were generated last epoch.
    seed_everything(seed)
    model = pipeline.model.model
    model.train()
    device = next(model.parameters()).device
    parameters = [p for p in model.parameters() if p.requires_grad]
    total, count = 0.0, 0
    for ids, mask, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        total += loss.item() * len(ids)
        count += len(ids)
    return total / count


def _validation_score(pipeline, validation: dict, config):
    by_battery = {}
    for battery, windows in validation.items():
        samples = predict_samples(pipeline, windows["context"], config,
                                  config.validation_seed, config.validation_samples)
        by_battery[battery] = point_metrics(windows["target"], summarize_samples(samples)["median"])["rmse"]
    return float(np.mean(list(by_battery.values()))), by_battery


def select_epochs(series: dict, train_batteries: list[str], config, seed: int, output: Path):
    train, validation, split_info = [], {}, {}
    for battery in train_batteries:
        training, valid, cutoff = temporal_split(series[battery], config)
        train.append(training)
        validation[battery] = valid
        split_info[battery] = {"cutoff_zero_based": cutoff, "last_training_target_cycle": cutoff,
                               "first_validation_target_cycle": cutoff + 1,
                               "training_windows": len(training["target"]), "validation_windows": len(valid["target"])}
    pipeline = fresh_pipeline(config, seed, lora=True)
    optimizer = None
    history = []
    try:
        loader = _training_loader(pipeline, train, config, seed)
        optimizer = _optimizer(pipeline, config)
        best, by_battery = _validation_score(pipeline, validation, config)
        best_epoch, stale = 0, 0
        history.append({"epoch": 0, "train_ce": None, "validation_macro_rmse": best, "by_battery": by_battery})
        print(f"  validation epoch=0 macro RMSE={best:.6f}", flush=True)
        for epoch in range(1, config.max_epochs + 1):
            loss = _train_epoch(pipeline, loader, optimizer, seed + epoch)
            score, by_battery = _validation_score(pipeline, validation, config)
            history.append({"epoch": epoch, "train_ce": loss, "validation_macro_rmse": score, "by_battery": by_battery})
            if score < best - config.min_delta:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
            print(f"  validation epoch={epoch} CE={loss:.4f} macro RMSE={score:.6f} best={best_epoch}", flush=True)
            write_json(output, {"status": "running", "train_batteries": train_batteries, "splits": split_info,
                                "seed": seed, "history": history, "selected_epoch": best_epoch})
            if stale >= config.patience:
                break
        record = {"status": "complete", "train_batteries": train_batteries, "splits": split_info,
                  "seed": seed, "history": history, "selected_epoch": best_epoch,
                  "selection_metric": "Macro validation RMSE of interpolated median; epoch 0 is eligible",
                  "validation_protocol": "Chronological tail of outer training batteries; no held-out battery used"}
        write_json(output, record)
        return best_epoch
    finally:
        del optimizer, pipeline
        clear_device_cache()


def reload_adapter(config: ExperimentConfig, adapter_path: Path):
    pipeline = fresh_pipeline(config, 0)
    pipeline.model.model = PeftModel.from_pretrained(pipeline.model.model, adapter_path, is_trainable=False)
    pipeline.model.model.eval()
    return pipeline


def _paired_comparison(directory: Path, baseline: Path):
    columns = ["battery", "inference_seed", "estimator"]
    tuned = pd.read_csv(directory / "metrics.csv")
    zero = pd.read_csv(baseline / "metrics.csv")
    zero = zero[zero.model == "ZeroShot"]
    paired = tuned.merge(zero[columns + ["rmse", "mae"]], on=columns, suffixes=("_lora", "_zero"), validate="many_to_one")
    paired["rmse_improvement"] = paired.rmse_zero - paired.rmse_lora
    paired["mae_improvement"] = paired.mae_zero - paired.mae_lora
    paired.to_csv(directory / "paired_comparison.csv", index=False)
    macro = paired.groupby(["train_seed", "inference_seed", "estimator"], as_index=False)[
        ["rmse_lora", "rmse_zero", "mae_lora", "mae_zero", "rmse_improvement", "mae_improvement"]].mean()
    macro["rmse_improvement_pct"] = 100 * macro.rmse_improvement / macro.rmse_zero
    macro.to_csv(directory / "paired_macro.csv", index=False)
    # A small tolerance makes improved / worsened / tied mutually exclusive.
    tw = pd.read_csv(directory / "window_metrics.csv")
    zw = pd.read_csv(baseline / "window_metrics.csv")
    zw = zw[zw.model == "ZeroShot"]
    win_keys = columns + ["forecast_start"]
    joined = tw.merge(zw[win_keys + ["rmse"]], on=win_keys, suffixes=("_lora", "_zero"), validate="many_to_one")
    delta = joined.rmse_zero - joined.rmse_lora
    joined["delta_rmse"] = delta
    joined["outcome"] = np.where(np.abs(delta) <= 1e-8, "tied", np.where(delta > 0, "improved", "worsened"))
    joined.to_csv(directory / "paired_windows.csv", index=False)


def run_lobo(root: Path, config: ExperimentConfig, zeroshot_run: Path | None = None) -> Path:
    root = Path(root).resolve()
    series = load_soh(root, config)
    baseline = Path(zeroshot_run) if zeroshot_run else find_zeroshot(root, config)
    if baseline is None:
        baseline = run_zeroshot(root, config)
    verify_zeroshot(baseline, root, config)
    directory, manifest = _start_run(root, config, "lora")
    manifest["zeroshot_run"] = str(baseline.resolve())
    write_json(directory / "manifest.json", manifest)
    pipeline, optimizer = None, None
    try:
        rows, horizons, window_rows, folds = [], [], [], []
        for test_battery in config.battery_ids:
            train_batteries = [b for b in config.battery_ids if b != test_battery]
            for seed in config.train_seeds:
                print(f"LOBO test={test_battery}, train={train_batteries}, seed={seed}", flush=True)
                selected = select_epochs(series, train_batteries, config, seed,
                                         directory / "selection" / f"{test_battery}_train{seed}.json")
                print(f"  refit on all training data: {selected} epoch(s)", flush=True)
                pipeline = fresh_pipeline(config, seed, lora=True)
                training = [make_windows(series[b], config.context_length, config.prediction_length) for b in train_batteries]
                loader = _training_loader(pipeline, training, config, seed)
                optimizer = _optimizer(pipeline, config)
                losses = []
                for epoch in range(1, selected + 1):
                    loss = _train_epoch(pipeline, loader, optimizer, seed + epoch)
                    losses.append(loss)
                    print(f"  refit epoch={epoch}/{selected} CE={loss:.4f}", flush=True)
                pipeline.model.model.eval()
                adapter = root / "models" / directory.name / f"{test_battery}_train{seed}"
                pipeline.model.model.save_pretrained(adapter)
                fold = {"test_battery": test_battery, "train_batteries": train_batteries, "train_seed": seed,
                        "selected_epoch": selected, "refit_losses": losses,
                        "adapter_path": str(adapter.resolve()),
                        "adapter_sha256": {p.name: file_sha256(p) for p in adapter.iterdir() if p.is_file()}}
                write_json(adapter / "experiment.json", {"config": asdict(config), "fold": fold})
                write_json(directory / "folds" / f"{test_battery}_train{seed}.json", fold)
                windows = make_windows(series[test_battery], config.context_length, config.prediction_length)
                for infer_seed in config.inference_seeds:
                    samples = predict_samples(pipeline, windows["context"], config, infer_seed)
                    save_forecast(directory / "predictions" / f"{test_battery}_train{seed}_infer{infer_seed}.npz", windows, samples)
                    r, h, w = metric_rows(windows, summarize_samples(samples), "LoRA", test_battery, seed, infer_seed)
                    rows.extend(r); horizons.extend(h); window_rows.extend(w)
                    print(f"  test {test_battery} train={seed} infer={infer_seed}: median RMSE={r[0]['rmse']:.6f}", flush=True)
                folds.append(fold)
                _write_tables(directory, rows, horizons, window_rows)
                del optimizer, pipeline
                optimizer, pipeline = None, None
                clear_device_cache()
        manifest["folds"] = folds
        _paired_comparison(directory, baseline)
        _finish_run(directory, manifest)
    except Exception as error:
        _finish_run(directory, manifest, error)
        raise
    finally:
        del optimizer, pipeline
        clear_device_cache()
    return directory


def verify_saved_run(directory: Path, root: Path, config: ExperimentConfig) -> dict:
    """Check recorded checksums and recalculate metrics from saved raw samples."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["evaluation_signature"] != evaluation_signature(root, config):
        raise ValueError("Run incomplete or evaluation signature incompatible")
    if manifest["kind"] == "lora" and manifest["config"] != json.loads(json.dumps(asdict(config))):
        raise ValueError("LoRA training configuration differs")
    for relative, digest in manifest["artifact_sha256"].items():
        if file_sha256(directory / relative) != digest:
            raise ValueError(f"Changed artifact: {relative}")
    for fold in manifest.get("folds", []):
        for name, digest in fold["adapter_sha256"].items():
            if file_sha256(Path(fold["adapter_path"]) / name) != digest:
                raise ValueError(f"Changed adapter: {fold['adapter_path']}/{name}")
    metrics = pd.read_csv(directory / "metrics.csv")
    series = load_soh(root, config)
    checked = 0
    for row in metrics.itertuples(index=False):
        windows = make_windows(series[row.battery], config.context_length, config.prediction_length)
        if row.model in ("LastValue", "Linear"):
            pred = _baselines(windows)[row.model]
        else:
            name = (f"{row.battery}_infer{row.inference_seed}.npz" if row.model == "ZeroShot" else
                    f"{row.battery}_train{row.train_seed}_infer{row.inference_seed}.npz")
            with np.load(directory / "predictions" / name, allow_pickle=False) as saved:
                for key in ("context", "target", "forecast_start"):
                    np.testing.assert_array_equal(saved[key], windows[key])
                summary = summarize_samples(saved["samples"])
                for key, value in summary.items():
                    np.testing.assert_array_equal(saved[key], value)
                pred = summary[row.estimator]
                if hasattr(row, "coverage80") and np.isfinite(row.coverage80):
                    coverage = np.mean((windows["target"] >= summary["q10"]) & (windows["target"] <= summary["q90"]))
                    np.testing.assert_allclose(coverage, row.coverage80, rtol=0, atol=1e-12)
        calculated = point_metrics(windows["target"], pred)
        np.testing.assert_allclose([calculated["rmse"], calculated["mae"]], [row.rmse, row.mae], rtol=0, atol=1e-12)
        checked += 1
    return {"run_id": manifest["run_id"], "metric_rows_verified": checked,
            "artifacts_verified": len(manifest["artifact_sha256"]), "status": "passed"}


def find_lobo(root: Path, config: ExperimentConfig) -> Path | None:
    for path in sorted((root / "results" / "runs").glob("*_lora_*/manifest.json"), reverse=True):
        try:
            verify_saved_run(path.parent, root, config)
            return path.parent
        except (ValueError, KeyError, FileNotFoundError, AssertionError):
            continue
    return None
