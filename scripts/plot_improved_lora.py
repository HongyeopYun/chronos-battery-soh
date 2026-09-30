"""Export SOH curves from the fixed EOS-loss ablation; no model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_evaluation import r2_score
BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
STAGES = ["Early", "Middle", "Late"]
OBSERVED = "#1f77b4"
TRUTH = "#ff7f0e"
PREDICTED = "#2ca02c"


def load_cases(diagnosis):
    cases, provenance = {}, {}
    for battery in BATTERIES:
        directory = diagnosis / ("training" if battery == "B0005" else "confirm_eos") / f"{battery}_without_eos"
        record = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
        assert record["variant"] == "without_eos" and record["test_battery"] == battery
        assert battery not in record["train_batteries"]
        path = directory / "final.npz"
        with np.load(path, allow_pickle=False) as saved:
            data = {key: saved[key] for key in ("context", "target", "forecast_start", "samples")}
        context, target, samples = data["context"], data["target"], data["samples"]
        n = len(target)
        assert context.shape == (n, 64) and target.shape == (n, 16) and samples.shape == (n, 100, 16)
        assert all(np.isfinite(values).all() for values in data.values())
        assert np.all(np.diff(data["forecast_start"]) == 1)
        assert data["forecast_start"][0] == 65
        data["q10"], data["median"], data["q90"] = np.quantile(samples, [.1, .5, .9], axis=1)
        data["indices"] = [0, n // 2, n - 1]
        data["series"] = np.concatenate([context[0], target[0], target[1:, -1]])
        # Check the reconstructed full curve against every saved window.
        for i, start in enumerate(data["forecast_start"]):
            np.testing.assert_array_equal(context[i], data["series"][start - 65:start - 1])
            np.testing.assert_array_equal(target[i], data["series"][start - 1:start + 15])
        cases[battery] = data
        provenance[battery] = {"prediction_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "train_batteries": record["train_batteries"], "epochs": record["epochs"], "steps": record["steps"]}
    return cases, provenance


def style_axis(ax):
    ax.set(xlabel="Discharge cycle", ylabel="SOH (Qk/Q1)")
    ax.grid(alpha=.2)


def legend(ax):
    handles = [Line2D([], [], color=OBSERVED, label="Observed context"),
               Line2D([], [], color=TRUTH, label="Observed future"),
               Line2D([], [], color=PREDICTED, label="Forecast median"),
               Patch(facecolor=OBSERVED, alpha=.2, label="10-90% interval")]
    ax.legend(handles=handles, fontsize=8)


def forecast(ax, data, idx, context=True):
    start = int(data["forecast_start"][idx])
    future = np.arange(start, start + 16)
    if context:
        ax.plot(np.arange(start - 64, start), data["context"][idx], color=OBSERVED, label="Observed context")
    ax.plot(future, data["target"][idx], color=TRUTH, label="Observed future")
    ax.plot(future, data["median"][idx], color=PREDICTED, label="Forecast median")
    ax.fill_between(future, data["q10"][idx], data["q90"][idx], color=OBSERVED, alpha=.2, label="10-90% interval")
    if context:
        ax.axvline(start - 1, color="gray", linestyle="--")


def export(fig, destination, stem):
    for extension in ("png", "pdf", "svg"):
        path = destination / f"{stem}.{extension}"
        fig.savefig(path, dpi=240, facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis", type=Path, default=ROOT / "results/diagnostics/20260929")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures/improved_lora_soh")
    args = parser.parse_args()
    cases, sources = load_cases(args.diagnosis.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    # Match src.soh_plots.forecast_grid, which produced zeroshot_0.png.
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})

    fig, axes = plt.subplots(4, 3, figsize=(17, 3.4 * 4), squeeze=False)
    selected_rows, forecast_rows = [], []
    for row, battery in enumerate(BATTERIES):
        data = cases[battery]
        for col, (stage, idx) in enumerate(zip(STAGES, data["indices"])):
            ax = axes[row, col]
            start = int(data["forecast_start"][idx])
            forecast(ax, data, idx)
            ax.set_title(f"{battery} | forecast starts at cycle {start}")
            style_axis(ax)
            rmse = float(np.sqrt(np.mean((data["median"][idx] - data["target"][idx]) ** 2)))
            selected_rows.append({"battery": battery, "stage": stage, "window_index": idx,
                "context_start": start - 64, "context_end": start - 1, "forecast_start": start,
                "forecast_end": start + 15, "median_rmse": rmse,
                "median_r2": r2_score(data["target"][idx], data["median"][idx])})
            for h in range(16):
                forecast_rows.append({"battery": battery, "stage": stage, "cycle": start + h,
                    "actual": data["target"][idx, h], "median": data["median"][idx, h],
                    "q10": data["q10"][idx, h], "q90": data["q90"][idx, h]})
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    export(fig, args.output, "improved_lora_4x3_windows")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, battery in zip(axes.flat, BATTERIES):
        data = cases[battery]
        ax.plot(np.arange(1, len(data["series"]) + 1), data["series"], color=OBSERVED)
        for stage, idx in zip(STAGES, data["indices"]):
            start = int(data["forecast_start"][idx])
            forecast(ax, data, idx, context=False)
            ax.axvline(start - 1, color="gray", linestyle="--", alpha=.5)
        ax.set_title(battery)
        style_axis(ax)
    legend(axes[0, 0])
    fig.tight_layout()
    export(fig, args.output, "improved_lora_4battery_overview")
    pd.DataFrame(selected_rows).to_csv(args.output / "selected_windows.csv", index=False)
    pd.DataFrame(forecast_rows).to_csv(args.output / "selected_forecasts.csv", index=False)
    metadata = {"model": "amazon/chronos-t5-base + q/v LoRA, EOS target loss excluded",
        "training_seed": 42, "inference_seed": 42, "num_samples": 100,
        "context_length": 64, "prediction_length": 16, "normalization": "Q_k / Q_1",
        "selection": "Indices [0, n//2, n-1], independent of errors; n=89 except B0018 n=53",
        "summary": "Linear interpolated median and 10th/90th percentiles from saved samples; no clipping or smoothing",
        "scope": "Post-hoc EOS-loss ablation; one training seed and one inference seed shown",
        "style_reference": "results/figures/zeroshot_0.png; src.soh_plots.forecast_grid",
        "source_predictions": sources}
    (args.output / "figure_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(selected_rows).to_string(index=False))
    print(f"Figures and source tables written to {args.output}")


if __name__ == "__main__":
    main()
