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
from pathlib import Path, PureWindowsPath
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from chronos import ChronosPipeline
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset

SCHEMA_VERSION = 2


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
    train_seeds: tuple[int, ...] = (42, 43, 44)
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_epochs: int = 10
    validation_fraction: float = 0.2
    validation_samples: int = 64
    validation_seed: int = 31415
    validation_seeds: tuple[int, ...] = ()
    validation_batch_size: int = 32
    battery_balance: bool = False
    patience: int = 3
    min_delta: float = 1e-5
    kl_weight: float = 0.5
    selection_metric: str = "guarded_crps"
    refit_full_data: bool = False

    def __post_init__(self):
        for name in ("battery_ids", "inference_seeds", "train_seeds"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            if not getattr(self, name) or len(set(getattr(self, name))) != len(getattr(self, name)):
                raise ValueError(f"{name} must be nonempty and contain unique values")
        object.__setattr__(self, "validation_seeds", tuple(self.validation_seeds))
        if len(set(self.validation_seeds)) != len(self.validation_seeds):
            raise ValueError("validation_seeds must contain unique values")
        if not isinstance(self.battery_balance, bool):
            raise ValueError("battery_balance must be boolean")
        if len(self.battery_ids) < 3:
            raise ValueError("LOBO requires at least three batteries")
        for name in ("context_length", "prediction_length", "inference_batch_size",
                     "num_samples", "batch_size", "lora_rank", "validation_samples", "validation_batch_size", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.validation_fraction < 1 or self.max_epochs < 0:
            raise ValueError("Invalid validation_fraction or max_epochs")
        if not self.revision or self.learning_rate <= 0 or self.min_delta < 0:
            raise ValueError("A pinned revision, positive learning_rate and nonnegative min_delta are required")
        if not np.isfinite(self.kl_weight) or self.kl_weight < 0:
            raise ValueError("kl_weight must be finite and nonnegative")
        if self.selection_metric not in ("median_rmse", "guarded_crps"):
            raise ValueError("selection_metric must be median_rmse or guarded_crps")


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


def resolve_reference(root: Path, reference: str | Path) -> Path:
    """Resolve new project-relative references and relocate legacy Windows runs."""
    root = Path(root).resolve()
    path = Path(reference)
    if path.is_absolute() and path.exists():
        return path
    windows = PureWindowsPath(str(reference))
    if not path.is_absolute() and not windows.is_absolute():
        return root / path
    parts = windows.parts if windows.is_absolute() else path.parts
    for marker in ("results", "models"):
        if marker in parts:
            return root.joinpath(*parts[parts.index(marker):])
    return path


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


def distribution_metrics(target: np.ndarray, samples: np.ndarray) -> dict[str, float]:
    """Empirical CRPS: E|X-y| - E|X-X'|/2, computed by sorting (no S x S array)."""
    summary = summarize_samples(samples)
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    point_metrics(target, summary["median"])
    count = samples.shape[1]
    weights = (2 * np.arange(1, count + 1) - count - 1)[None, :, None]
    pair_term = np.sum(np.sort(samples, axis=1) * weights, axis=1) / count ** 2
    crps = np.mean(np.abs(samples - target[:, None, :]), axis=1) - pair_term
    return {"crps": float(np.mean(crps)),
            "coverage80": float(np.mean((target >= summary["q10"]) & (target <= summary["q90"]))),
            "width80": float(np.mean(summary["q90"] - summary["q10"])),
            "negative_sample_rate": float(np.mean(samples < 0)),
            "above_two_sample_rate": float(np.mean(samples > 2))}


def masked_reference_kl(logits, reference_log_probs, labels):
    """KL(base || adapted) averaged over valid tokens, including EOS, excluding padding."""
    valid = labels != -100
    if not valid.any():
        raise ValueError("KL requires at least one valid label")
    token_kl = F.kl_div(F.log_softmax(logits.float(), dim=-1), reference_log_probs.float(),
                        reduction="none", log_target=True).sum(dim=-1)
    return token_kl[valid].mean()


def validation_candidate_improves(candidate: dict, best: dict, baseline: dict, config) -> bool:
    if config.selection_metric == "median_rmse":
        return candidate["median_rmse"] < best["median_rmse"] - config.min_delta
    # Guardrails are relative to the untrained adapter on the SAME validation data.
    eligible = all(candidate[key] <= baseline[key] + config.min_delta
                   for key in ("median_rmse", "mean_rmse"))
    return eligible and candidate["crps"] < best["crps"] - config.min_delta


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Seeds all available torch devices.


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            requested = "xpu"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Select the project .venv kernel, install the "
                "CUDA PyTorch wheel from README.md, and restart the kernel. "
                f"torch={torch.__version__}, compiled CUDA={torch.version.cuda}"
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist")
        return f"cuda:{index}"
    return str(device)


def model_dtype(device: str) -> torch.dtype:
    target = torch.device(device)
    if target.type == "cuda":
        with torch.cuda.device(target):
            # Native BF16 requires Ampere or newer. FP32 avoids T5 FP16 overflow.
            supports_bf16 = (torch.cuda.get_device_capability(target)[0] >= 8
                             and torch.cuda.is_bf16_supported())
        return torch.bfloat16 if supports_bf16 else torch.float32
    return torch.bfloat16 if target.type == "xpu" else torch.float32


def runtime_info(requested: str) -> dict:
    device = resolve_device(requested)
    info = {"device": device, "dtype": str(model_dtype(device)), "torch": torch.__version__}
    if torch.device(device).type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(gpu=props.name, capability=[props.major, props.minor],
                    total_memory_bytes=props.total_memory, cuda_runtime=torch.version.cuda)
    return info


def clear_device_cache() -> None:
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fresh_pipeline(config: ExperimentConfig, seed: int, lora: bool = False):
    seed_everything(seed)  # MUST precede model and adapter creation.
    device = resolve_device(config.device)
    dtype = model_dtype(device)
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
            "versions": versions, "device": device, "runtime": runtime_info(device), "git_commit": commit,
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
                     runtime=runtime_info(config.device),
                     data_sha256={b: file_sha256(root / "data" / "raw" / f"{b}.mat") for b in config.battery_ids},
                     versions={p: importlib.metadata.version(p) for p in ("torch", "chronos-forecasting", "transformers", "peft", "numpy")},
                     source_sha256=file_sha256(Path(__file__)))
    return json.loads(json.dumps(signature))


def _start_run(root: Path, config: ExperimentConfig, kind: str):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{kind}_{uuid4().hex[:6]}"
    run_root = root / "results" / "runs"
    if kind in ("lora_ablation", "lora_robust"):
        run_root = run_root / "ablations"
    directory = run_root / run_id
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
                train_seed: int = -1, inference_seed: int = -1, samples=None):
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
        if samples is not None:
            row.update(distribution_metrics(target, samples))
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
                r, h, w = metric_rows(windows, summarize_samples(samples), "ZeroShot", battery, inference_seed=seed, samples=samples)
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
    tensors = [ids, mask, labels]
    if config.kl_weight:
        # Teacher is the frozen base with adapters disabled. Cache only training
        # contexts/labels; neither validation nor held-out battery enters this loss.
        model = pipeline.model.model
        model.eval()
        device = next(model.parameters()).device
        reference = []
        with torch.no_grad(), model.disable_adapter():
            for start in range(0, len(ids), config.batch_size):
                stop = start + config.batch_size
                logits = model(input_ids=ids[start:stop].to(device),
                               attention_mask=mask[start:stop].to(device),
                               labels=labels[start:stop].to(device)).logits
                reference.append(F.log_softmax(logits.float(), dim=-1).cpu())
        tensors.append(torch.cat(reference))
    if config.battery_balance:
        sizes = [len(d["context"]) for d in datasets]
        if not sizes or min(sizes) < 1:
            raise ValueError("Battery-balanced training requires nonempty batteries")
        # Mean weight is one over the full dataset; shuffle/update count is unchanged.
        tensors.append(torch.cat([torch.full((n,), sum(sizes) / (len(sizes) * n), dtype=torch.float32)
                                  for n in sizes]))
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    loader.battery_balance = config.battery_balance
    return loader


def _optimizer(pipeline, config):
    return torch.optim.AdamW([p for p in pipeline.model.model.parameters() if p.requires_grad],
                              lr=config.learning_rate, weight_decay=config.weight_decay)


def _train_epoch(pipeline, loader, optimizer, seed: int, kl_weight: float = 0.0):
    # Independent of how many validation forecasts were generated last epoch.
    seed_everything(seed)
    model = pipeline.model.model
    model.train()
    device = next(model.parameters()).device
    parameters = [p for p in model.parameters() if p.requires_grad]
    total, count = 0.0, 0
    for batch in loader:
        ids, mask, labels = (tensor.to(device) for tensor in batch[:3])
        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        # Accumulate token losses in FP32 even when the frozen base uses BF16.
        balanced = getattr(loader, "battery_balance", False)
        weights = batch[-1].to(device) if balanced else None
        # Exactly preserve the original reduction when all battery weights equal one.
        weighted = weights is not None and not bool(torch.all(weights == 1))
        if kl_weight and len(batch) != 4 + int(balanced):
            raise ValueError("Reference probabilities are required for KL regularization")
        if weighted:
            valid = labels != -100
            counts = valid.sum(dim=1).clamp_min(1)
            ce = F.cross_entropy(outputs.logits.float().transpose(1, 2), labels,
                                 ignore_index=-100, reduction="none")
            per_example = ce.sum(dim=1) / counts
            if kl_weight:
                kl = F.kl_div(F.log_softmax(outputs.logits.float(), dim=-1), batch[3].to(device).float(),
                              reduction="none", log_target=True).sum(dim=-1)
                per_example = per_example + kl_weight * (kl * valid).sum(dim=1) / counts
            loss = (per_example * weights).mean()
        else:
            loss = F.cross_entropy(outputs.logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100)
            if kl_weight:
                loss = loss + kl_weight * masked_reference_kl(outputs.logits, batch[3].to(device), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        total += loss.item() * len(ids)
        count += len(ids)
    return total / count


def validation_seed_scores(pipeline, validation: dict, config):
    """Score each seed separately, without pooling Monte Carlo samples."""
    def score(windows, samples):
        summary = summarize_samples(samples)
        return {
            "median_rmse": point_metrics(windows["target"], summary["median"])["rmse"],
            "mean_rmse": point_metrics(windows["target"], summary["mean"])["rmse"],
            **distribution_metrics(windows["target"], samples),
        }
    seeds = config.validation_seeds or (config.validation_seed,)
    by_seed = {}
    for seed in seeds:
        if not config.validation_seeds:
            # Preserve the exact legacy RNG/batch layout for historical experiments.
            by_seed[seed] = {b: score(w, predict_samples(pipeline, w["context"], config, seed, config.validation_samples))
                             for b, w in validation.items()}
        else:
            combined = np.concatenate([w["context"] for w in validation.values()])
            samples = predict_samples(pipeline, combined, replace(config, inference_batch_size=config.validation_batch_size),
                                      seed, config.validation_samples)
            by_seed[seed], offset = {}, 0
            for battery, windows in validation.items():
                stop = offset + len(windows["context"])
                by_seed[seed][battery] = score(windows, samples[offset:stop])
                offset = stop
    return by_seed


def aggregate_validation_scores(by_seed):
    seeds = list(by_seed)
    batteries = list(by_seed[seeds[0]])
    metrics = list(by_seed[seeds[0]][batteries[0]])
    by_battery = {b: {k: float(np.mean([by_seed[s][b][k] for s in seeds])) for k in metrics} for b in batteries}
    macro = {key: float(np.mean([row[key] for row in by_battery.values()]))
             for key in next(iter(by_battery.values()))}
    return macro, by_battery


def _validation_score(pipeline, validation: dict, config):
    return aggregate_validation_scores(validation_seed_scores(pipeline, validation, config))


def select_epochs(series: dict, train_batteries: list[str], config, seed: int, output: Path,
                  checkpoint: Path | None = None):
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
        baseline = best.copy()
        best_epoch, stale = 0, 0
        if checkpoint is not None:
            pipeline.model.model.save_pretrained(checkpoint)
        history.append({"epoch": 0, "train_loss": None, "validation_macro_rmse": best["median_rmse"],
                        "validation": best, "by_battery": by_battery})
        print(f"  validation epoch=0 RMSE={best['median_rmse']:.6f} CRPS={best['crps']:.6f}", flush=True)
        for epoch in range(1, config.max_epochs + 1):
            loss = _train_epoch(pipeline, loader, optimizer, seed + epoch, config.kl_weight)
            score, by_battery = _validation_score(pipeline, validation, config)
            history.append({"epoch": epoch, "train_loss": loss, "validation_macro_rmse": score["median_rmse"],
                            "validation": score, "by_battery": by_battery})
            if validation_candidate_improves(score, best, baseline, config):
                best, best_epoch, stale = score, epoch, 0
                if checkpoint is not None:
                    pipeline.model.model.save_pretrained(checkpoint)
            else:
                stale += 1
            print(f"  validation epoch={epoch} loss={loss:.4f} RMSE={score['median_rmse']:.6f} "
                  f"CRPS={score['crps']:.6f} mean_RMSE={score['mean_rmse']:.6f} best={best_epoch}", flush=True)
            write_json(output, {"status": "running", "train_batteries": train_batteries, "splits": split_info,
                                "seed": seed, "history": history, "selected_epoch": best_epoch})
            if stale >= config.patience:
                break
        record = {"status": "complete", "train_batteries": train_batteries, "splits": split_info,
                  "seed": seed, "history": history, "selected_epoch": best_epoch,
                  "selection_metric": config.selection_metric, "baseline_validation": baseline,
                  "selected_validation": best, "train_objective": "FP32 CE + kl_weight * KL(base || adapted)",
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
    baseline = resolve_reference(root, zeroshot_run) if zeroshot_run else find_zeroshot(root, config)
    if baseline is None:
        baseline = run_zeroshot(root, config)
    verify_zeroshot(baseline, root, config)
    directory, manifest = _start_run(root, config, "lora")
    manifest["zeroshot_run"] = baseline.resolve().relative_to(root).as_posix()
    write_json(directory / "manifest.json", manifest)
    pipeline, optimizer = None, None
    try:
        rows, horizons, window_rows, folds = [], [], [], []
        for test_battery in config.battery_ids:
            train_batteries = [b for b in config.battery_ids if b != test_battery]
            for seed in config.train_seeds:
                print(f"LOBO test={test_battery}, train={train_batteries}, seed={seed}", flush=True)
                adapter = root / "models" / directory.name / f"{test_battery}_train{seed}"
                selected = select_epochs(series, train_batteries, config, seed,
                                         directory / "selection" / f"{test_battery}_train{seed}.json",
                                         checkpoint=adapter)
                losses = []
                if config.refit_full_data:
                    print(f"  refit on all training data: {selected} epoch(s)", flush=True)
                    pipeline = fresh_pipeline(config, seed, lora=True)
                    training = [make_windows(series[b], config.context_length, config.prediction_length) for b in train_batteries]
                    loader = _training_loader(pipeline, training, config, seed)
                    optimizer = _optimizer(pipeline, config)
                    for epoch in range(1, selected + 1):
                        loss = _train_epoch(pipeline, loader, optimizer, seed + epoch, config.kl_weight)
                        losses.append(loss)
                        print(f"  refit epoch={epoch}/{selected} loss={loss:.4f}", flush=True)
                    del loader
                else:
                    print(f"  loading validated checkpoint: epoch {selected}", flush=True)
                    pipeline = reload_adapter(config, adapter)
                pipeline.model.model.eval()
                pipeline.model.model.save_pretrained(adapter)
                fold = {"test_battery": test_battery, "train_batteries": train_batteries, "train_seed": seed,
                        "selected_epoch": selected, "refit_losses": losses,
                        "fit_protocol": "full_data_refit" if config.refit_full_data else "validated_temporal_checkpoint",
                        "adapter_path": adapter.relative_to(root).as_posix(),
                        "adapter_sha256": {p.name: file_sha256(p) for p in adapter.iterdir() if p.is_file()}}
                write_json(adapter / "experiment.json", {"config": asdict(config), "fold": fold})
                write_json(directory / "folds" / f"{test_battery}_train{seed}.json", fold)
                windows = make_windows(series[test_battery], config.context_length, config.prediction_length)
                for infer_seed in config.inference_seeds:
                    samples = predict_samples(pipeline, windows["context"], config, infer_seed)
                    save_forecast(directory / "predictions" / f"{test_battery}_train{seed}_infer{infer_seed}.npz", windows, samples)
                    r, h, w = metric_rows(windows, summarize_samples(samples), "LoRA", test_battery, seed, infer_seed, samples=samples)
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


def audit_saved_run(directory: Path, root: Path, verify_adapters: bool = False) -> dict:
    """Audit persisted forecasts using their recorded data/config, without requiring the original GPU."""
    directory = resolve_reference(root, directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Run incomplete")
    config = ExperimentConfig(**manifest["config"])
    for battery, digest in manifest["evaluation_signature"]["data_sha256"].items():
        if file_sha256(root / "data" / "raw" / f"{battery}.mat") != digest:
            raise ValueError(f"Changed data: {battery}")
    for relative, digest in manifest["artifact_sha256"].items():
        if file_sha256(directory / relative) != digest:
            raise ValueError(f"Changed artifact: {relative}")
    for fold in manifest.get("folds", []) if verify_adapters else []:
        for name, digest in fold["adapter_sha256"].items():
            if file_sha256(resolve_reference(root, fold["adapter_path"]) / name) != digest:
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
                diagnostics = distribution_metrics(windows["target"], saved["samples"])
                for key, value in diagnostics.items():
                    if hasattr(row, key) and np.isfinite(getattr(row, key)):
                        np.testing.assert_allclose(value, getattr(row, key), rtol=0, atol=1e-12)
        calculated = point_metrics(windows["target"], pred)
        np.testing.assert_allclose([calculated["rmse"], calculated["mae"]], [row.rmse, row.mae], rtol=0, atol=1e-12)
        checked += 1
    return {"run_id": manifest["run_id"], "metric_rows_verified": checked,
            "artifacts_verified": len(manifest["artifact_sha256"]),
            "adapters_verified": verify_adapters, "status": "passed"}


def verify_saved_run(directory: Path, root: Path, config: ExperimentConfig) -> dict:
    """Strict cache reuse check, separate from hardware-independent artifact audit."""
    directory = resolve_reference(root, directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["kind"] not in ("zeroshot", "lora"):
        raise ValueError("Special experiment kinds require their own protocol verifier; use audit_saved_run for historical artifacts")
    if manifest["evaluation_signature"] != evaluation_signature(root, config):
        raise ValueError("Evaluation signature incompatible")
    if manifest["kind"] == "lora":
        if manifest["config"] != json.loads(json.dumps(asdict(config))):
            raise ValueError("LoRA training configuration differs")
        verify_zeroshot(resolve_reference(root, manifest["zeroshot_run"]), root, config)
    return audit_saved_run(directory, root, verify_adapters=True)


def find_lobo(root: Path, config: ExperimentConfig) -> Path | None:
    for path in sorted((root / "results" / "runs").glob("*_lora_*/manifest.json"), reverse=True):
        try:
            if json.loads(path.read_text(encoding="utf-8"))["kind"] != "lora":
                continue
            verify_saved_run(path.parent, root, config)
            return path.parent
        except (ValueError, KeyError, FileNotFoundError, AssertionError):
            continue
    return None
