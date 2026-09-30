"""Derived evaluation tables with R²; original training artifacts stay immutable."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.soh_experiment import (
    ExperimentConfig, load_soh, make_windows, temporal_split, _baselines,
    summarize_samples, point_metrics, file_sha256, write_json, verify_saved_run,
)

R2_DEFINITION = (
    "R2 = 1 - sum((target - prediction)^2) / sum((target - mean(target))^2). "
    "Flatten all windows and horizons within each battery/seed; average seed scores "
    "then give batteries equal weight. Horizon/window tables use their own target mean. "
    "Negative values are valid; constant targets or fewer than 2 values return NaN. "
    "R2 is a reporting metric only; checkpoint selection remains unchanged."
)


def r2_score(target, prediction):
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or not target.size:
        raise ValueError("Targets and predictions must have matching nonempty shapes")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite target or prediction")
    denominator = np.square(target - target.mean()).sum()
    if target.size < 2 or denominator == 0:
        return float("nan")
    return float(1 - np.square(target - prediction).sum() / denominator)


def point_scores(target, prediction):
    return {**point_metrics(target, prediction), "r2": r2_score(target, prediction)}


def distribution_r2(target, samples):
    summary = summarize_samples(samples)
    return {f"{estimator}_r2": r2_score(target, summary[estimator]) for estimator in ("median", "mean")}


def _r2_from_rmse(target, rmse):
    target = np.asarray(target, dtype=np.float64)
    variance = np.square(target - target.mean()).mean()
    return float(1 - float(rmse)**2 / variance) if target.size > 1 and variance > 0 else float("nan")


def evaluation_directory(root, source):
    return Path(root) / "results/evaluation_r2" / Path(source).resolve().relative_to(Path(root).resolve() / "results")


def export_run(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**manifest["config"])
    verify_saved_run(directory, root, cfg)
    out = evaluation_directory(root, directory)
    out.mkdir(parents=True, exist_ok=True)
    source_paths = set()
    series = load_soh(root, cfg)
    cache = {}

    def arrays(row):
        key = (row.battery, row.model, row.train_seed, row.inference_seed)
        if key not in cache:
            if row.model in ("LastValue", "Linear"):
                windows = make_windows(series[row.battery], cfg.context_length, cfg.prediction_length)
                summary = {"median": _baselines(windows)[row.model]}
            else:
                filename = (f"{row.battery}_infer{row.inference_seed}.npz" if row.model == "ZeroShot" else
                            f"{row.battery}_train{row.train_seed}_infer{row.inference_seed}.npz")
                path = directory / "predictions" / filename
                source_paths.add(path)
                with np.load(path, allow_pickle=False) as saved:
                    windows = {k: saved[k] for k in ("context", "target", "forecast_start")}
                    summary = summarize_samples(saved["samples"])
            cache[key] = (windows, summary)
        return cache[key]

    tables = {}
    for filename in ("metrics.csv", "horizon_metrics.csv", "window_metrics.csv"):
        frame = pd.read_csv(directory / filename)
        values = []
        for row in frame.itertuples(index=False):
            windows, summary = arrays(row)
            target, pred = windows["target"], summary[row.estimator]
            if filename == "horizon_metrics.csv":
                target, pred = target[:, row.horizon-1:row.horizon], pred[:, row.horizon-1:row.horizon]
            elif filename == "window_metrics.csv":
                index = int(np.flatnonzero(windows["forecast_start"] == row.forecast_start)[0])
                target, pred = target[index:index+1], pred[index:index+1]
            scores = point_scores(target, pred)
            np.testing.assert_allclose([scores["rmse"], scores["mae"]], [row.rmse, row.mae], rtol=0, atol=1e-12)
            values.append(scores["r2"])
        frame["r2"] = values
        tables[filename] = frame
        frame.to_csv(out / filename, index=False)

    metrics = tables["metrics.csv"]
    keys = ["model", "estimator", "train_seed", "inference_seed"]
    macro = metrics.groupby(keys, as_index=False)[["rmse", "mae", "r2"]].mean()
    macro.to_csv(out / "macro_by_run.csv", index=False)
    summary = pd.read_csv(directory / "summary.csv").merge(
        macro.groupby(["model", "estimator"]).agg(r2_mean=("r2", "mean"), r2_std=("r2", "std"),
                                                  r2_n_runs=("r2", "count")).reset_index(),
        on=["model", "estimator"], validate="one_to_one")
    summary.to_csv(out / "summary.csv", index=False)
    metrics.groupby(["model", "estimator", "battery"])[["rmse", "mae", "r2"]].mean().to_csv(out / "battery_summary.csv")

    if manifest.get("zeroshot_run"):
        baseline = Path(manifest["zeroshot_run"])
        zero_out = export_run(root, baseline)
        zero = pd.read_csv(zero_out / "metrics.csv").query("model == 'ZeroShot'")
        join = ["battery", "inference_seed", "estimator"]
        paired = metrics.merge(zero[join + ["rmse", "mae", "r2"]], on=join,
                               suffixes=("_lora", "_zero"), validate="many_to_one")
        paired["rmse_improvement"] = paired.rmse_zero - paired.rmse_lora
        paired["mae_improvement"] = paired.mae_zero - paired.mae_lora
        paired["r2_improvement"] = paired.r2_lora - paired.r2_zero
        paired.to_csv(out / "paired_comparison.csv", index=False)
        columns = ["rmse_lora", "rmse_zero", "mae_lora", "mae_zero", "r2_lora", "r2_zero",
                   "rmse_improvement", "mae_improvement", "r2_improvement"]
        paired_macro = paired.groupby(["train_seed", "inference_seed", "estimator"], as_index=False)[columns].mean()
        paired_macro["rmse_improvement_pct"] = 100 * paired_macro.rmse_improvement / paired_macro.rmse_zero
        paired_macro.to_csv(out / "paired_macro.csv", index=False)
        zw = pd.read_csv(zero_out / "window_metrics.csv").query("model == 'ZeroShot'")
        wkeys = join + ["forecast_start"]
        joined = tables["window_metrics.csv"].merge(zw[wkeys + ["rmse", "r2"]], on=wkeys,
                                                    suffixes=("_lora", "_zero"), validate="many_to_one")
        joined["delta_rmse"] = joined.rmse_zero - joined.rmse_lora
        joined["delta_r2"] = joined.r2_lora - joined.r2_zero
        joined.to_csv(out / "paired_windows.csv", index=False)

    if (directory / "distribution_metrics.csv").exists():
        frame = pd.read_csv(directory / "distribution_metrics.csv")
        def distribution_path(row):
            if row.model == "ZeroShot":
                return baseline / "predictions" / f"{row.battery}_infer{row.inference_seed}.npz"
            return directory / "predictions" / f"{row.battery}_train{row.train_seed}_infer{row.inference_seed}.npz"
        frame = enrich_distribution_table(frame, distribution_path)
        frame.to_csv(out / "distribution_metrics.csv", index=False)

    # Every saved validation check retains per-battery, per-seed RMSE. This exact
    # algebraic conversion avoids pretending an averaged RMSE is a pooled RMSE.
    history_rows = []
    for path in sorted((directory / "trials").glob("*/complete.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for check in record["history"]:
            for row in check["validation"]["by_seed_and_battery"]:
                target = temporal_split(series[row["battery"]], cfg)[1]["target"]
                history_rows.append({"test_battery": record["test_battery"], "train_seed": record["train_seed"],
                                     "learning_rate": record["learning_rate"], "step": check["step"],
                                     "selected_checkpoint": check["step"] == record["selected_step"], **row,
                                     "median_r2": _r2_from_rmse(target, row["median_rmse"]),
                                     "mean_r2": _r2_from_rmse(target, row["mean_rmse"])})
    if history_rows:
        frame = pd.DataFrame(history_rows)
        frame.to_csv(out / "validation_history_by_battery_seed.csv", index=False)
        group = ["test_battery", "train_seed", "learning_rate", "step", "selected_checkpoint"]
        macro_history = frame.groupby(group, as_index=False)[["median_rmse", "mean_rmse", "crps", "median_r2", "mean_r2"]].mean()
        macro_history.to_csv(out / "validation_history.csv", index=False)
        candidate = pd.read_csv(directory / "validation_candidates.csv")
        chosen = macro_history[macro_history.selected_checkpoint].rename(columns={"step": "selected_step"})
        candidate.merge(chosen[["test_battery", "train_seed", "learning_rate", "selected_step", "median_r2", "mean_r2"]],
                        on=["test_battery", "train_seed", "learning_rate", "selected_step"],
                        validate="one_to_one").to_csv(out / "validation_candidates.csv", index=False)

    write_json(out / "provenance.json", {"source_run": str(directory.relative_to(root)), "r2_definition": R2_DEFINITION,
        "metric_source_sha256": file_sha256(Path(__file__)), "original_manifest_sha256": file_sha256(directory / "manifest.json"),
        "prediction_sha256": {str(p.relative_to(root)): file_sha256(p) for p in sorted(source_paths)},
        "original_metrics_recalculated": len(metrics), "training_or_selection_changed": False})
    return out


def enrich_distribution_table(frame, resolve_path):
    frame = frame.copy()
    rows = []
    for row in frame.itertuples(index=False):
        with np.load(resolve_path(row), allow_pickle=False) as data:
            # Historical diagnose_lora casts to float64 before quantiles, unlike
            # the original run tables. Preserve that exact estimator protocol.
            samples = np.asarray(data["samples"], dtype=np.float64)
            summary = summarize_samples(samples)
            for estimator in ("median", "mean"):
                np.testing.assert_allclose(point_metrics(data["target"], summary[estimator])["rmse"],
                                           getattr(row, f"{estimator}_rmse"), rtol=0, atol=1e-12)
            rows.append(distribution_r2(data["target"], samples))
    for key in ("median_r2", "mean_r2"):
        frame[key] = [row[key] for row in rows]
    return frame


def export_diagnosis(root, diagnosis=None):
    root = Path(root).resolve()
    diagnosis = Path(diagnosis or root / "results/diagnostics/20260929").resolve()
    out = evaluation_directory(root, diagnosis)
    # This historical diagnosis used fixed runs, even if newer runs are added.
    original = {
        "zeroshot": root / "results/runs/20260923T063348Z_zeroshot_b57856",
        "lora": root / "results/runs/20260923T063628Z_lora_2df6cb",
    }

    def confirmation_path(row):
        if row.model in ("zero", "original_lora"):
            directory = original["zeroshot" if row.model == "zero" else "lora"]
            matches = list((directory / "predictions").glob(f"{row.battery}_*infer{row.inference_seed}.npz"))
            if len(matches) != 1:
                raise ValueError("Ambiguous diagnostic source")
            return matches[0]
        if row.inference_seed == 42:
            part = "training" if row.battery == "B0005" else "confirm_eos"
            return diagnosis / part / f"{row.battery}_without_eos/final.npz"
        return diagnosis / "seed_confirmation" / f"{row.battery}_without_eos_infer{row.inference_seed}.npz"

    frame = enrich_distribution_table(pd.read_csv(diagnosis / "seed_confirmation/metrics.csv"), confirmation_path)
    folder = out / "seed_confirmation"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "metrics.csv", index=False)
    columns = [c for c in frame.select_dtypes("number").columns if c != "inference_seed"]
    frame.groupby(["model", "battery"])[columns].mean().to_csv(folder / "battery_summary.csv")
    frame.groupby(["model", "inference_seed"])[columns].mean().to_csv(folder / "macro_by_seed.csv")
    frame.groupby(["model", "battery"])[columns].mean().groupby("model").mean().to_csv(folder / "macro_summary.csv")
    for path in diagnosis.rglob("*controls.csv"):
        def resolve(row):
            if "training_controls" in path.name:
                return path.parent / f"{row.battery}_{row.variant}" / f"{row.checkpoint}.npz"
            return path.parent / f"{row.battery}_{row.variant}.npz"
        extended = enrich_distribution_table(pd.read_csv(path), resolve)
        target = out / path.relative_to(diagnosis)
        target.parent.mkdir(parents=True, exist_ok=True)
        extended.to_csv(target, index=False)
    write_json(out / "provenance.json", {"source": str(diagnosis.relative_to(root)),
        "r2_definition": R2_DEFINITION, "metric_source_sha256": file_sha256(Path(__file__)),
        "method": "Original RMSE rechecked against raw sample arrays; R2 computed from the same arrays."})
    return out
