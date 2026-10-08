"""Dataset-independent Chronos zero-shot evaluation of common SOH bundles."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd

from src.soh_data import load_bundle, observed_windows, sha256
from src.soh_experiment import (
    fresh_pipeline, predict_samples, summarize_samples, clear_device_cache,
    resolve_device, write_json, _environment, _baselines,
)
from src.soh_evaluation import point_scores
from src.soh_checkpoint import distribution_scores

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SCORES = ["rmse", "r2", "wql", "picp80"]
PROBABILITY_SCORES = ["wql", "picp80", "width80", "crps", "negative_rate", "above2_rate"]
SCORES = [*PRIMARY_SCORES, "width80", "mae", "crps", "negative_rate", "above2_rate"]
WQL_QUANTILES = tuple(i / 10 for i in range(1, 10))
EVALUATION_PROTOCOL = {
    "primary_metrics": PRIMARY_SCORES, "primary_point_estimator": "median",
    "quantile_levels": WQL_QUANTILES, "quantile_method": "linear",
    "wql": "2 * mean_q(sum(max(q * (y - forecast_q), (q - 1) * (y - forecast_q)))) / sum(abs(y))",
    "wql_zero_denominator": "NaN", "picp80": "mean(q10 <= y <= q90); fraction, target 0.8",
    "aggregation": "Calculate within each battery/seed; average seeds and give battery records equal weights",
    "auxiliary_metrics": ["width80", "mae", "crps", "negative_rate", "above2_rate"],
}
IDENTITY = ["dataset", "battery_id", "physical_battery_id", "model", "estimator", "inference_seed"]
TABLES = ["metrics.csv", "horizon_metrics.csv", "window_metrics.csv", "battery_summary.csv",
          "macro_by_seed.csv", "summary.csv", "paired_comparison.csv"]


@dataclass(frozen=True)
class ZeroShotConfig:
    model_id: str = "amazon/chronos-t5-base"
    revision: str = "ad294eaacead15db499b740ea4122266dd2a81a2"
    context_length: int = 64
    prediction_length: int = 16
    device: str = "auto"
    inference_batch_size: int = 16
    num_samples: int = 100
    inference_seeds: tuple[int, ...] = (42, 43, 44)
    temperature: float = 1.
    top_k: int = 50
    top_p: float = 1.
    stride: int = 1
    battery_ids: tuple[str, ...] | None = None
    max_windows_per_battery: int | None = None
    calce_cross_file_windows: bool = False

    def __post_init__(self):
        if not isinstance(self.calce_cross_file_windows, bool):
            raise ValueError("calce_cross_file_windows must be boolean")
        for name in ("context_length", "prediction_length", "inference_batch_size", "num_samples", "stride", "top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.model_id or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Use a model ID and a pinned 40-character revision")
        if not np.isfinite(self.temperature) or self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid sampling temperature/top_p")
        if self.device not in {"auto", "cpu", "xpu", "cuda"} and not re.fullmatch(r"(?:xpu|cuda):\d+", self.device):
            raise ValueError("Unsupported inference device")
        object.__setattr__(self, "inference_seeds", tuple(self.inference_seeds))
        if (not self.inference_seeds or len(set(self.inference_seeds)) != len(self.inference_seeds)
                or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in self.inference_seeds)):
            raise ValueError("Inference seeds must be unique integers in [0, 2**32)")
        if self.battery_ids is not None:
            object.__setattr__(self, "battery_ids", tuple(self.battery_ids))
            if (not self.battery_ids or len(set(self.battery_ids)) != len(self.battery_ids)
                    or any(not isinstance(b, str) or not b for b in self.battery_ids)):
                raise ValueError("battery_ids must contain unique nonempty strings")
        limit = self.max_windows_per_battery
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("max_windows_per_battery must be a positive integer or None")


def load_zeroshot_config(path=ROOT / "configs/t5-battery.json"):
    return ZeroShotConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def prepare_battery_windows(bundle, config):
    """Retain source cycles and masks; sampling depends only on window indices."""
    data, batteries, manifest = load_bundle(bundle)
    if config.calce_cross_file_windows and data.dataset.eq("calce").any():
        from src.calce_windows import observation_segments
        data = data.copy()
        if "source_segment_id" not in data:
            data["source_segment_id"] = data.segment_id
        for _, calce in data[data.dataset.eq("calce")].groupby("battery_id", sort=False):
            data.loc[calce.index, "segment_id"] = observation_segments(calce)
    if config.battery_ids is not None:
        missing = set(config.battery_ids) - set(batteries.battery_id)
        if missing:
            raise ValueError(f"Unknown batteries: {sorted(missing)}")
        batteries = batteries[batteries.battery_id.isin(config.battery_ids)]
    records, coverage = [], []
    for index, metadata in enumerate(batteries.to_dict("records"), 1):
        rows = data[(data.dataset == metadata["dataset"]) & (data.battery_id == metadata["battery_id"])]
        windows = observed_windows(rows, config.context_length, config.prediction_length)
        eligible = len(windows["target"])
        indices = np.arange(eligible)[::config.stride]
        if config.max_windows_per_battery is not None and len(indices) > config.max_windows_per_battery:
            indices = indices[np.linspace(0, len(indices) - 1, config.max_windows_per_battery, dtype=int)]
        windows = {k: v[indices] for k, v in windows.items()}
        windows["context_cycle"] = windows["forecast_start"][:, None] - np.arange(config.context_length, 0, -1)
        record = {"record_id": f"{index:04d}", "dataset": str(metadata["dataset"]), "battery_id": str(metadata["battery_id"]),
                  "physical_battery_id": str(metadata.get("physical_battery_id") or metadata["battery_id"])}
        coverage.append({**record, "n_rows": len(rows), "eligible_windows": eligible,
                         "evaluated_windows": len(indices), "status": "ready" if len(indices) else "skipped_no_eligible_windows"})
        if len(indices):
            records.append({**record, "windows": windows})
    return records, pd.DataFrame(coverage), data, manifest


def baseline_predictions(windows):
    if windows["context"].shape[1] == 1:
        last = np.repeat(windows["context"], windows["target"].shape[1], axis=1)
        return {"LastValue": last, "Linear": last.copy()}
    return _baselines(windows)


def quantile_loss_values(target, samples, quantile_levels=WQL_QUANTILES):
    """Return mean twice-pinball loss at each window/horizon, before normalization."""
    target, samples = np.asarray(target, dtype=np.float64), np.asarray(samples, dtype=np.float64)
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if (target.ndim != 2 or not target.size or samples.ndim != 3 or not samples.shape[1]
            or samples.shape[0::2] != target.shape or not np.isfinite(target).all() or not np.isfinite(samples).all()):
        raise ValueError("Expected finite targets [windows, horizon] and samples [windows, samples, horizon]")
    if levels.ndim != 1 or not levels.size or not np.isfinite(levels).all() or np.any((levels <= 0) | (levels >= 1)) or np.any(np.diff(levels) <= 0):
        raise ValueError("Quantile levels must be strictly increasing within (0, 1)")
    quantiles = np.quantile(samples, levels, axis=1, method="linear")
    error = target[None] - quantiles
    q = levels[:, None, None]
    return 2 * np.maximum(q * error, (q - 1) * error).mean(axis=0)


def normalized_quantile_loss(target, loss):
    denominator = np.abs(target).sum()
    return float(loss.sum() / denominator) if denominator else float("nan")


def weighted_quantile_loss(target, samples, quantile_levels=WQL_QUANTILES):
    return normalized_quantile_loss(np.asarray(target), quantile_loss_values(target, samples, quantile_levels))


def interval_metrics(target, loss, coverage, width):
    return {"wql": normalized_quantile_loss(target, loss), "picp80": float(coverage.mean()),
            "width80": float(width.mean())}


def score_prediction(record, prediction, model, estimator, seed=-1, samples=None):
    windows, tags = record["windows"], {k: record[k] for k in IDENTITY[:3]}
    tags.update(model=model, estimator=estimator, inference_seed=seed)
    target = windows["target"]
    row = {**tags, **point_scores(target, prediction), "n_windows": len(target),
           **{k: float("nan") for k in PROBABILITY_SCORES}}
    probability = None
    if samples is not None:
        distribution = distribution_scores(target, samples)
        summary = summarize_samples(np.asarray(samples, dtype=np.float64))
        probability = (quantile_loss_values(target, samples),
                       (target >= summary["q10"]) & (target <= summary["q90"]), summary["q90"] - summary["q10"])
        row.update(interval_metrics(target, *probability))
        row.update({k: distribution[k] for k in ["crps", "negative_rate", "above2_rate"]})
    def probability_block(index):
        return interval_metrics(target[index], *(value[index] for value in probability)) if probability is not None else {
            k: float("nan") for k in ["wql", "picp80", "width80"]}
    horizons = [{**tags, "horizon": h + 1, **point_scores(target[:, h:h+1], prediction[:, h:h+1]),
                 **probability_block((slice(None), slice(h, h+1)))} for h in range(target.shape[1])]
    window_rows = [{**tags, "forecast_start": int(cycle), **point_scores(target[i:i+1], prediction[i:i+1]),
                    **probability_block((slice(i, i+1), slice(None)))} for i, cycle in enumerate(windows["forecast_start"])]
    return row, horizons, window_rows


def evaluation_tables(records, directory):
    """Compute every metric from the same saved targets and raw samples."""
    rows, horizons, windows = [], [], []
    for record in records:
        forecasts = [(name, "point", -1, values, None) for name, values in baseline_predictions(record["windows"]).items()]
        for file in sorted((directory / "predictions").glob(f"{record['record_id']}_infer*.npz")):
            with np.load(file, allow_pickle=False) as saved:
                samples = saved["samples"]
                seed = int(saved["inference_seed"])
                summary = summarize_samples(samples)
                forecasts.extend(("Chronos", estimator, seed, summary[estimator], samples) for estimator in ("median", "mean"))
        for model, estimator, seed, values, samples in forecasts:
            row, h, w = score_prediction(record, values, model, estimator, seed, samples)
            rows.append(row); horizons.extend(h); windows.extend(w)
    metrics = pd.DataFrame(rows)
    battery = metrics.groupby(IDENTITY[:-1], as_index=False, sort=False)[SCORES + ["n_windows"]].mean()
    macro = metrics.groupby(["model", "estimator", "inference_seed"], as_index=False)[SCORES].mean()
    summary = macro.groupby(["model", "estimator"]).agg(
        **{f"{s}_{stat}": (s, stat) for s in SCORES for stat in ("mean", "std")},
        n_seeds=("inference_seed", "size")).reset_index()
    comparisons = []
    for keys, group in battery.groupby(IDENTITY[:3], sort=False):
        for estimator in ("median", "mean"):
            chronos = group[(group.model == "Chronos") & (group.estimator == estimator)].iloc[0]
            for baseline in ("LastValue", "Linear"):
                reference = group[group.model == baseline].iloc[0]
                comparisons.append({**dict(zip(IDENTITY[:3], keys)), "estimator": estimator, "baseline": baseline,
                    "rmse_chronos": chronos.rmse, "rmse_baseline": reference.rmse,
                    "rmse_improvement_pct": 100 * (1 - chronos.rmse / reference.rmse) if reference.rmse else float("nan"),
                    "r2_chronos": chronos.r2, "r2_baseline": reference.r2, "r2_delta": chronos.r2 - reference.r2})
    return dict(zip(TABLES, [metrics, pd.DataFrame(horizons), pd.DataFrame(windows), battery, macro, summary,
                            pd.DataFrame(comparisons)]))


def run_signature(bundle, config):
    source = ["battery_zeroshot.py", "battery_zeroshot_plots.py", "soh_data.py", "soh_experiment.py",
              "soh_evaluation.py", "soh_checkpoint.py"]
    environment = _environment(ROOT, resolve_device(config.device))
    return {"schema_version": 1, "config": json.loads(json.dumps(asdict(config))),
            "bundle_manifest_sha256": sha256(Path(bundle) / "manifest.json"),
            "implementation_sha256": {name: sha256(ROOT / "src" / name) for name in source},
            "versions": environment["versions"], "device": environment["device"]}


def verified_forecasts(directory, manifest, bundle, config, allow_metric_update=False):
    """Check saved inputs/samples before scoring or upgrading evaluation tables."""
    if manifest["schema_version"] != 1 or manifest["kind"] != "battery_zeroshot" or manifest["status"] != "complete":
        raise ValueError("Unsupported or incomplete battery zero-shot run")
    records, coverage, _, _ = prepare_battery_windows(bundle, config)
    signature = run_signature(bundle, config)
    if allow_metric_update:
        for key in ["schema_version", "config", "bundle_manifest_sha256", "versions", "device"]:
            if manifest["signature"][key] != signature[key]:
                raise ValueError(f"Cannot update metrics: forecast identity changed ({key})")
        for name in ["soh_data.py", "soh_experiment.py"]:
            if manifest["signature"]["implementation_sha256"][name] != signature["implementation_sha256"][name]:
                raise ValueError(f"Cannot update metrics: inference/data implementation changed ({name})")
    elif manifest["signature"] != signature:
        raise ValueError("Zero-shot data, code, configuration or environment changed")
    required = {*TABLES, "coverage.csv", "report.md"}
    expected_predictions = {f"predictions/{r['record_id']}_infer{s}.npz" for r in records for s in config.inference_seeds}
    if not required.union(expected_predictions).issubset(manifest["artifact_sha256"]):
        raise ValueError("Missing required zero-shot artifacts")
    for name, expected in manifest["artifact_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Zero-shot artifact changed or missing: {name}")
    actual_predictions = {p.relative_to(directory).as_posix() for p in (directory / "predictions").glob("*.npz")}
    if actual_predictions != expected_predictions:
        raise ValueError("Prediction inventory differs from the configured batteries/seeds")
    provenance = manifest.get("prediction_provenance")
    if provenance:
        original = (directory / provenance["original_manifest"]).resolve()
        if not original.is_relative_to(directory) or sha256(original) != provenance["original_manifest_sha256"]:
            raise ValueError("Original inference provenance changed")
        original_manifest = json.loads(original.read_text(encoding="utf-8"))
        for name in expected_predictions:
            if manifest["artifact_sha256"][name] != original_manifest["artifact_sha256"][name]:
                raise ValueError("Predictions differ from their original inference provenance")
    for record in records:
        for seed in config.inference_seeds:
            with np.load(directory / f"predictions/{record['record_id']}_infer{seed}.npz", allow_pickle=False) as saved:
                for key, value in record["windows"].items():
                    np.testing.assert_array_equal(saved[key], value)
                for key in IDENTITY[:3]:
                    if str(saved[key]) != record[key]:
                        raise ValueError("Saved forecast battery identity differs")
                if int(saved["inference_seed"]) != seed or saved["samples"].shape != (len(record["windows"]["target"]), config.num_samples, config.prediction_length):
                    raise ValueError("Saved forecast shape or seed differs")
                for key, value in summarize_samples(saved["samples"]).items():
                    np.testing.assert_allclose(saved[key], value, rtol=1e-12, atol=1e-12)
    return records, coverage


def verify_battery_zeroshot(directory, bundle=None, config=None):
    directory = Path(directory).resolve()
    if (directory / "run.json").is_file():
        from src.battery_zeroshot_minimal import verify_compact_result, saved_config
        result = verify_compact_result(directory)
        if config is not None and config != saved_config(result):
            raise ValueError("Saved compact configuration differs")
        if bundle is not None and Path(result["bundle"]).resolve() != Path(bundle).resolve():
            raise ValueError("Saved compact bundle differs")
        return {key: result[key] for key in ["status", "scope", "batteries_evaluated", "batteries_skipped", "evaluated_windows"]}
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    config = config or ZeroShotConfig(**manifest["config"])
    bundle = Path(bundle).resolve() if bundle else (ROOT / manifest["bundle"]).resolve()
    records, coverage = verified_forecasts(directory, manifest, bundle, config)
    if manifest.get("evaluation_protocol") != json.loads(json.dumps(EVALUATION_PROTOCOL)):
        raise ValueError("Evaluation metric definitions differ")
    tables = {"coverage.csv": coverage, **evaluation_tables(records, directory)}
    for name, expected in tables.items():
        pd.testing.assert_frame_equal(pd.read_csv(directory / name, keep_default_na=False, na_values=[""],
                                      dtype={k: str for k in ["record_id", *IDENTITY[:3]]}),
                                      expected, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)
    return {"batteries_evaluated": len(records), "batteries_skipped": int((coverage.evaluated_windows == 0).sum()),
            "windows": int(coverage.evaluated_windows.sum()), "prediction_files": len(records) * len(config.inference_seeds),
            "scope": manifest["scope"], "artifacts_verified": True, "metrics_recomputed": True}


def refresh_battery_zeroshot(directory, bundle=None):
    """Upgrade metrics without inference; preserve original prediction provenance."""
    directory = Path(directory).resolve()
    from src.battery_zeroshot_minimal import GROUPS
    if any((directory / group / "run.json").is_file() for group in GROUPS):
        if bundle is not None:
            raise ValueError("Suite refresh uses each group's saved input bundle")
        from src.battery_zeroshot_minimal import refresh_all_compact_r2
        refresh_all_compact_r2(directory)
        return {"groups_updated": len(GROUPS), "r2_metric": "mean_window_r2", "inference_repeated": False}
    if (directory / "run.json").is_file():
        from src.battery_zeroshot_minimal import refresh_compact_r2
        refresh_compact_r2(directory, bundle)
        return verify_battery_zeroshot(directory, bundle)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = ZeroShotConfig(**manifest["config"])
    bundle = Path(bundle).resolve() if bundle else (ROOT / manifest["bundle"]).resolve()
    records, coverage = verified_forecasts(directory, manifest, bundle, config, allow_metric_update=True)
    predictions = {p.name: sha256(p) for p in (directory / "predictions").glob("*.npz")}
    tables = evaluation_tables(records, directory)
    original = directory / "inference_manifest.json"
    if not original.exists():
        shutil.copy2(manifest_path, original)
    manifest.setdefault("prediction_provenance", {"original_manifest": original.name,
        "original_manifest_sha256": sha256(original), "note": "Predictions retain their original producer; metric updates do not repeat inference"})
    manifest.update(status="updating_metrics")
    write_json(manifest_path, manifest)
    try:
        for name, frame in tables.items():
            frame.to_csv(directory / name, index=False)
        from src.battery_zeroshot_plots import write_report
        write_report(directory, records, coverage, config, manifest["scope"], tables, reuse_figures=True)
        if predictions != {p.name: sha256(p) for p in (directory / "predictions").glob("*.npz")}:
            raise ValueError("Raw forecasts changed during the metric update")
        manifest.update(status="complete", signature=run_signature(bundle, config), evaluation_protocol=EVALUATION_PROTOCOL,
            metrics_updated_at=datetime.now(timezone.utc).isoformat(),
            artifact_sha256={p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*"))
                             if p.is_file() and p.name != "manifest.json"})
        write_json(manifest_path, manifest)
        return verify_battery_zeroshot(directory, bundle, config)
    except Exception as error:
        manifest.update(status="metric_update_failed", error=f"{type(error).__name__}: {error}")
        write_json(manifest_path, manifest)
        raise


def battery_zeroshot(bundle, config=None, output_root=ROOT / "results/t5-battery", force=False, minimal=True):
    """Swap only the common-format bundle path; reuse matching, verified runs."""
    if minimal:
        from src.battery_zeroshot_minimal import battery_zeroshot_minimal
        return battery_zeroshot_minimal(bundle, config, output_root, force)
    bundle, output_root = Path(bundle).resolve(), Path(output_root).resolve()
    config = config or load_zeroshot_config()
    if output_root == bundle or bundle in output_root.parents or output_root in bundle.parents:
        raise ValueError("Forecast outputs must be separate from the input data bundle")
    records, coverage, _, _ = prepare_battery_windows(bundle, config)
    if not records:
        raise ValueError("No eligible forecast windows; check context/horizon, quality flags and segment lengths")
    signature = run_signature(bundle, config)
    if not force and output_root.exists():
        for path in sorted(output_root.glob("*/manifest.json"), reverse=True):
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("status") == "complete" and previous.get("signature") == signature:
                verify_battery_zeroshot(path.parent, bundle, config)
                return path.parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_batteryzeroshot_" + uuid4().hex[:6]
    directory = output_root / run_id
    (directory / "predictions").mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "kind": "battery_zeroshot", "run_id": run_id, "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
                "bundle": str(bundle.relative_to(ROOT).as_posix()) if bundle.is_relative_to(ROOT) else str(bundle),
                "signature": signature, "evaluation_protocol": EVALUATION_PROTOCOL,
                "environment": _environment(ROOT, resolve_device(config.device)),
                "scope": "smoke" if config.max_windows_per_battery is not None else ("strided" if config.stride != 1 else "full"),
                "protocol": "Observed rolling contexts; no training, clipping, interpolation or renormalization. Equal battery-record weights, seeds averaged. Overlapping windows are not independent."}
    write_json(directory / "manifest.json", manifest)
    coverage.to_csv(directory / "coverage.csv", index=False)
    print("RUN=" + str(directory), flush=True)
    pipeline = None
    try:
        pipeline = fresh_pipeline(config, config.inference_seeds[0])
        for record in records:
            windows = record["windows"]
            for seed in config.inference_seeds:
                print(f"PREDICT {record['dataset']}/{record['battery_id']} seed={seed} windows={len(windows['target'])}", flush=True)
                samples = predict_samples(pipeline, windows["context"], config, seed)
                np.savez_compressed(directory / f"predictions/{record['record_id']}_infer{seed}.npz",
                    **windows, samples=samples, **summarize_samples(samples), inference_seed=seed,
                    **{k: record[k] for k in IDENTITY[:3]})
        if sha256(bundle / "manifest.json") != signature["bundle_manifest_sha256"]:
            raise ValueError("Input bundle changed during forecasting")
        load_bundle(bundle)
        tables = evaluation_tables(records, directory)
        for name, frame in tables.items():
            frame.to_csv(directory / name, index=False)
        from src.battery_zeroshot_plots import write_report
        write_report(directory, records, coverage, config, manifest["scope"], tables)
        manifest.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
            artifact_sha256={p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*"))
                             if p.is_file() and p.name != "manifest.json"})
        write_json(directory / "manifest.json", manifest)
        verify_battery_zeroshot(directory, bundle, config)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", finished_at=datetime.now(timezone.utc).isoformat())
        write_json(directory / "manifest.json", manifest)
        raise
    finally:
        del pipeline
        clear_device_cache()
    return directory
