"""Dataset-independent Chronos zero-shot evaluation of common SOH bundles."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
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
SCORES = ["rmse", "mae", "r2", "crps", "coverage80", "width80", "negative_rate", "above2_rate"]
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

    def __post_init__(self):
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


def load_zeroshot_config(path=ROOT / "configs/battery_zeroshot.json"):
    return ZeroShotConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def prepare_battery_windows(bundle, config):
    """Retain source cycles and masks; sampling depends only on window indices."""
    data, batteries, manifest = load_bundle(bundle)
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


def score_prediction(record, prediction, model, estimator, seed=-1, samples=None):
    windows, tags = record["windows"], {k: record[k] for k in IDENTITY[:3]}
    tags.update(model=model, estimator=estimator, inference_seed=seed)
    target = windows["target"]
    row = {**tags, **point_scores(target, prediction), "n_windows": len(target),
           **{k: float("nan") for k in SCORES[3:]}}
    if samples is not None:
        distribution = distribution_scores(target, samples)
        row.update({k: distribution[k] for k in SCORES[3:]})
    horizons = [{**tags, "horizon": h + 1, **point_scores(target[:, h:h+1], prediction[:, h:h+1])}
                for h in range(target.shape[1])]
    window_rows = [{**tags, "forecast_start": int(cycle), **point_scores(target[i:i+1], prediction[i:i+1])}
                   for i, cycle in enumerate(windows["forecast_start"])]
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


def verify_battery_zeroshot(directory, bundle=None, config=None):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["kind"] != "battery_zeroshot" or manifest["status"] != "complete":
        raise ValueError("Unsupported or incomplete battery zero-shot run")
    config = config or ZeroShotConfig(**manifest["config"])
    bundle = Path(bundle).resolve() if bundle else (ROOT / manifest["bundle"]).resolve()
    records, coverage, _, _ = prepare_battery_windows(bundle, config)
    if manifest["signature"] != run_signature(bundle, config):
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
    tables = {"coverage.csv": coverage, **evaluation_tables(records, directory)}
    for name, expected in tables.items():
        pd.testing.assert_frame_equal(pd.read_csv(directory / name, keep_default_na=False, na_values=[""],
                                      dtype={k: str for k in ["record_id", *IDENTITY[:3]]}),
                                      expected, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)
    return {"batteries_evaluated": len(records), "batteries_skipped": int((coverage.evaluated_windows == 0).sum()),
            "windows": int(coverage.evaluated_windows.sum()), "prediction_files": len(expected_predictions),
            "scope": manifest["scope"], "artifacts_verified": True, "metrics_recomputed": True}


def battery_zeroshot(bundle, config=None, output_root=ROOT / "results/battery_zeroshot", force=False):
    """Swap only the common-format bundle path; reuse matching, verified runs."""
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
                "signature": signature, "environment": _environment(ROOT, resolve_device(config.device)),
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
