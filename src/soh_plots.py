"""Plots derived only from persisted experiment artifacts."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def forecast_grid(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    batteries = config["battery_ids"]
    infer_seed = config["inference_seeds"][0]
    train_seed = config["train_seeds"][0]
    figure, axes = plt.subplots(len(batteries), 3, figsize=(17, 3.4 * len(batteries)), squeeze=False)
    for row, battery in enumerate(batteries):
        name = (f"{battery}_infer{infer_seed}.npz" if manifest["kind"] == "zeroshot" else
                f"{battery}_train{train_seed}_infer{infer_seed}.npz")
        with np.load(directory / "predictions" / name, allow_pickle=False) as saved:
            indices = [0, len(saved["target"]) // 2, len(saved["target"]) - 1]
            for ax, idx in zip(axes[row], indices):
                start = saved["forecast_start"][idx]
                future = np.arange(start, start + config["prediction_length"])
                ax.plot(np.arange(start - config["context_length"], start), saved["context"][idx], label="Observed context")
                ax.plot(future, saved["target"][idx], label="Observed future")
                ax.plot(future, saved["median"][idx], label="Forecast median")
                ax.fill_between(future, saved["q10"][idx], saved["q90"][idx], alpha=0.2, label="10-90% interval")
                ax.axvline(start - 1, color="gray", linestyle="--")
                ax.set(title=f"{battery} | forecast starts at cycle {start}", xlabel="Discharge cycle", ylabel="SOH (Qk/Q1)")
                ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    return figure


def horizon_comparison(zeroshot: Path, lora: Path | None = None):
    frames = [pd.read_csv(zeroshot / "horizon_metrics.csv")]
    if lora:
        frames.append(pd.read_csv(lora / "horizon_metrics.csv"))
    data = pd.concat(frames)
    data = data[data.estimator == "median"]
    batteries = data.battery.unique()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, battery in zip(axes.flat, batteries):
        for model, group in data[data.battery == battery].groupby("model"):
            curve = group.groupby("horizon").rmse.mean()
            ax.plot(curve.index, curve.values, marker=".", label=model)
        ax.set(title=battery, xlabel="Horizon", ylabel="RMSE (averaged across runs)")
        ax.grid(alpha=0.2)
    axes.flat[0].legend()
    figure.tight_layout()
    return figure


def validation_curves(directory: Path):
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((directory / "selection").glob("*.json"))]
    paths = sorted((directory / "selection").glob("*.json"))
    figure, axes = plt.subplots(len(paths), 1, figsize=(10, 2.8 * len(paths)), squeeze=False)
    for ax, path, record in zip(axes.flat, paths, records):
        history = record["history"]
        ax.plot([x["epoch"] for x in history], [x["validation_macro_rmse"] for x in history], marker="o", label="Median RMSE")
        if all("validation" in x for x in history):
            ax.plot([x["epoch"] for x in history], [x["validation"]["crps"] for x in history],
                    marker=".", label="CRPS")
            ax.plot([x["epoch"] for x in history], [x["validation"]["mean_rmse"] for x in history],
                    linestyle=":", label="Mean forecast RMSE")
        ax.axvline(record["selected_epoch"], linestyle="--", label=f"Selected epoch {record['selected_epoch']}")
        ax.set(title=path.stem, xlabel="Epoch", ylabel="Validation macro error")
        ax.legend()
        ax.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def window_comparison(directory: Path):
    data = pd.read_csv(directory / "paired_windows.csv")
    data = data[data.estimator == "median"]
    batteries = data.battery.unique()
    figure, axes = plt.subplots(len(batteries), 1, figsize=(12, 3 * len(batteries)), squeeze=False)
    for ax, battery in zip(axes.flat, batteries):
        curve = data[data.battery == battery].groupby("forecast_start").delta_rmse.mean()
        ax.plot(curve.index, curve.values)
        ax.axhline(0, color="black", linestyle="--")
        ax.set(title=battery, xlabel="Forecast start cycle", ylabel="Zero-shot RMSE - LoRA RMSE")
        ax.grid(alpha=0.2)
    figure.tight_layout()
    return figure
