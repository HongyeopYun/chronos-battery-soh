"""Paired NASA SOH-only / voltage / current / voltage+current zero-shot test."""
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, WQL_QUANTILES, write_json, sha256
from src.battery_model_zeroshot import load_model, predict_quantiles, phase_labels, PHASES, SCORES, quantile_scores
from src.nasa_covariates import extract_nasa_features, covariate_inputs, FEATURES
from src.soh_evaluation import R2_DEFINITION, window_r2_values
from src.soh_experiment import clear_device_cache, resolve_device

CONDITIONS = {"SOH": (), "SOH+V": FEATURES[:1], "SOH+I": FEATURES[1:], "SOH+V+I": FEATURES}
OUTPUT = ROOT / "results/c2-battery/covariates/nasa_aging"
PREFIX = "c2-battery_nasa_vi"


def covariate_predictions(pipeline, windows, features, names, config):
    with torch.inference_mode():
        quantiles, points = pipeline.predict_quantiles(
            inputs=covariate_inputs(windows, features, names),
            prediction_length=config.prediction_length, context_length=config.context_length,
            quantile_levels=list(WQL_QUANTILES), batch_size=config.inference_batch_size,
            cross_learning=False)
    point = np.stack([p.cpu().numpy()[0] for p in points])
    values = np.stack([q.cpu().numpy()[0] for q in quantiles])
    if point.shape != windows["target"].shape or values.shape != (*point.shape, 9):
        raise ValueError("C2 covariate output axes differ")
    return point, values


def comparison_figure(records, data, predictions, directory, dataset_name="NASA", prefix=PREFIX,
                      first_eligible=True):
    fig, axes = plt.subplots(len(records), 3, figsize=(15, 3.2 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        battery_id, w = record["battery_id"], record["windows"]
        cycles = data[(data.battery_id == battery_id) & data.is_observed].cycle.to_numpy()
        labels = phase_labels(w["forecast_start"], cycles)
        centers = cycles.min() + (np.arange(3) + .5) * (cycles.max() - cycles.min() + 1) / 3
        indices = [0] if first_eligible else []
        phases = PHASES[1:] if first_eligible else PHASES
        phase_centers = centers[1:] if first_eligible else centers
        for phase, center in zip(phases, phase_centers):
            eligible = np.flatnonzero(labels == phase)
            indices.append(eligible[np.abs(w["forecast_start"][eligible] - center).argmin()])
        titles = ["First eligible" if first_eligible else "Early", "Middle", "Late"]
        for col, (index, title) in enumerate(zip(indices, titles)):
            ax = axes[row, col]
            future = w["target_cycle"][index]
            ax.plot(w["context_cycle"][index], w["context"][index], color=".5", lw=1.2, label="Observed context")
            ax.plot(future, w["target"][index], color="black", lw=1.6, label="Actual SOH")
            annotations = []
            for condition, color in [("SOH", "#2878b5"), ("SOH+V+I", "#d55e00")]:
                point, q = predictions[(battery_id, condition)]
                ax.plot(future, point[index], color=color, lw=1.6, label=condition)
                ax.fill_between(future, q[index, :, 0], q[index, :, -1], color=color, alpha=.12)
                scores = quantile_scores(w["target"][index:index+1], point[index:index+1])
                annotations.append(f"{condition}: RMSE {scores['rmse']:.4f}, R2 {scores['r2']:.2f}")
            ax.text(.02, .03, "\n".join(annotations), fontsize=8, transform=ax.transAxes)
            ax.axvline(future[0], color=".6", ls=":", lw=1)
            ax.set(title=f"{battery_id} | {title} | {future[0]}-{future[-1]}", xlabel="Cycle", ylabel="SOH (Q / Q_ref)")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    description = ("first eligible is not the observed-period early third" if first_eligible
                   else "representatives from observed-period thirds")
    fig.suptitle(f"Chronos-2 {dataset_name} zero-shot | 64 -> 16 | SOH vs SOH + past voltage/current\n"
                 f"Shaded: 10-90% prediction intervals; {description}")
    fig.tight_layout(rect=(0, 0, 1, .94))
    name = f"{prefix}_forecast_windows.png"
    fig.savefig(directory / name, dpi=150)
    plt.close(fig)
    return name


def report(directory, summary, battery, features, figure):
    overall = summary[summary.phase == "all"].set_index("condition")
    base = overall.loc["SOH"]
    vi = overall.loc["SOH+V+I"]
    lines = ["# NASA: Chronos-2에 과거 전압·전류 추가", "",
             "NASA 네 배터리의 동일한 320개 rolling window에서 비교했다. 과거 64 cycle로 다음 16 cycle의 SOH를 예측하며 stride는 1이다. 모델 추가 학습은 하지 않았다.", "",
             "전압·전류는 각 방전 시작 후 0~600초의 시간 가중 평균이다. 600초 경계의 값을 선형 보간해 적분하며, 방전 전체 길이·전류 적분 capacity는 특징에 사용하지 않았다. 과거 cycle의 측정값만 past_covariates로 입력하고 미래 센서값은 입력하지 않았다.", "",
             "단위는 전압 V, 전류 A다. 원래 부호를 유지하고 외부 정규화·클리핑·평활화는 하지 않았다. C2의 내부 정규화와 group attention을 사용하며 서로 다른 예측창 사이의 cross_learning은 껐다.", "",
             "| 입력 | RMSE ↓ | 평균 window R² ↑ | WQL ↓ | PICP@80 | RMSE 변화 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, row in overall.iterrows():
        change = (row.rmse / base.rmse - 1) * 100
        lines.append(f"| {name} | {row.rmse:.6f} | {row.r2:.4f} | {row.wql:.6f} | {row.picp80:.1%} | {change:+.2f}% |")
    lines += ["", "## 배터리별 SOH 대비 전압·전류 추가", "",
              "| 배터리 | SOH RMSE | SOH+V+I RMSE | 변화 | SOH R² | SOH+V+I R² |",
              "|---|---:|---:|---:|---:|---:|"]
    for battery_id, rows in battery[battery.phase == "all"].groupby("battery_id", sort=False):
        rows = rows.set_index("condition")
        a, b = rows.loc["SOH"], rows.loc["SOH+V+I"]
        lines.append(f"| {battery_id} | {a.rmse:.6f} | {b.rmse:.6f} | {(b.rmse/a.rmse-1)*100:+.2f}% | {a.r2:.4f} | {b.r2:.4f} |")
    lines += ["", "## 특징 범위", "", "| 배터리 | 평균 전압 범위 (V) | 평균 전류 범위 (A) |",
              "|---|---:|---:|"]
    for battery_id, rows in features.groupby("battery_id", sort=False):
        lines.append(f"| {battery_id} | {rows[FEATURES[0]].min():.6f}~{rows[FEATURES[0]].max():.6f} | "
                     f"{rows[FEATURES[1]].min():.6f}~{rows[FEATURES[1]].max():.6f} |")
    lines += ["", f"전압·전류 동시 추가의 전체 RMSE 변화는 {(vi.rmse/base.rmse-1)*100:+.2f}%다. "
              "이는 이 두 특징과 현재 문맥 길이에 대한 결과이며, 모든 다변량 입력의 효과나 배터리 물리 관계를 학습했다는 증거로 일반화하지 않는다.", "",
              "R²는 각 16-cycle 예측창에서 표준 1−SSE/SST로 계산한 뒤 배터리 안에서 단순 평균하고 네 배터리를 동일 가중 평균한다. 다른 지표도 배터리별 계산 후 동일 가중 평균한다. 중·후반은 기존 관측 cycle 범위의 1/3 구분을 유지한다. NASA 초반 1/3에는 64개 과거 관측을 확보한 창이 없으므로 점수는 없다. 그림의 첫 유효 창은 cycle 65~80이다.", "",
              "변수 선택은 결과를 보기 전에 고정했다. 겹치는 rolling window를 독립 표본으로 취급하는 유의성 검정은 하지 않았다. SOH-only 재실행 결과는 기존 저장된 네 배터리의 지표와 교차 확인했다.", "",
              f"![동일 예측창 비교]({figure})", "",
              f"전체/구간별 점수: [{PREFIX}_summary.csv]({PREFIX}_summary.csv). "
              f"배터리별 점수: [{PREFIX}_battery_metrics.csv]({PREFIX}_battery_metrics.csv)."]
    (directory / f"{PREFIX}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    original = ROOT / "results/c2-battery/full/nasa_aging"
    manifest = json.loads((original / "c2-battery_run.json").read_text(encoding="utf-8"))
    config = ZeroShotConfig(**manifest["config"])
    records, _, data, _ = prepare_battery_windows(manifest["bundle"], config)
    features = extract_nasa_features(ROOT / "data/raw", data)
    expected = pd.read_csv(original / "c2-battery_battery_metrics.csv")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
             "config": manifest["config"], "device": resolve_device(config.device),
             "conditions": CONDITIONS, "feature_interval_seconds": [0, 600],
             "feature_aggregation": "Time-weighted mean of Voltage_measured/Current_measured; linear boundary interpolation",
             "sensor_source": "Discharge records aligned using source_record_index and capacity/reference checks",
             "input": "Past covariates only; one target (SOH); cross_learning=False; no fine-tuning",
             "r2": R2_DEFINITION, "bundle_fingerprint": manifest["bundle_fingerprint"],
             "raw_sha256": {b: sha256(ROOT / f"data/raw/{b}.mat") for b in features.battery_id.unique()},
             "completed_tasks": 0}
    state_path = OUTPUT / f"{PREFIX}_run.json"
    write_json(state_path, state)
    features.to_csv(OUTPUT / f"{PREFIX}_cycle_features.csv", index=False)
    rows, window_rows, predictions, saved = [], [], {}, {}
    pipeline = load_model(config)
    try:
        for record in records:
            battery_id, w = record["battery_id"], record["windows"]
            f = features[features.battery_id == battery_id]
            cycles = data[(data.battery_id == battery_id) & data.is_observed].cycle.to_numpy()
            labels = phase_labels(w["forecast_start"], cycles)
            saved[f"{battery_id}_target"] = w["target"]
            saved[f"{battery_id}_forecast_start"] = w["forecast_start"]
            for index, (condition, names) in enumerate(CONDITIONS.items()):
                print(f"RUN {battery_id} {condition}: {len(w['target'])} windows", flush=True)
                point, q = (predict_quantiles(pipeline, w["context"], config) if not names else
                            covariate_predictions(pipeline, w, f, names, config))
                predictions[(battery_id, condition)] = (point, q)
                saved[f"{battery_id}_condition{index}_quantiles"] = q
                for phase in ["all", *PHASES]:
                    mask = np.ones(len(labels), dtype=bool) if phase == "all" else labels == phase
                    if not mask.any():
                        continue
                    score = quantile_scores(w["target"][mask], point[mask], q[mask])
                    rows.append({"battery_id": battery_id, "condition": condition, "phase": phase,
                                 "n_windows": int(mask.sum()), **score})
                    if not names:
                        old = expected[(expected.battery_id == battery_id) & (expected.model == "c2-battery") & (expected.phase == phase)].iloc[0]
                        np.testing.assert_allclose([score[key] for key in SCORES], old[SCORES].to_numpy(dtype=float),
                                                   rtol=2e-5, atol=5e-8)
                for start, phase, r2, rmse in zip(w["forecast_start"], labels, window_r2_values(w["target"], point),
                                                np.sqrt(np.square(w["target"] - point).mean(axis=1))):
                    window_rows.append({"battery_id": battery_id, "condition": condition, "forecast_start": int(start),
                                        "phase": phase, "rmse": float(rmse), "r2": float(r2)})
                state["completed_tasks"] += 1
                write_json(state_path, state)
        battery = pd.DataFrame(rows)
        summary = battery.groupby(["condition", "phase"], sort=False, as_index=False)[SCORES].mean()
        counts = battery.groupby(["condition", "phase"], sort=False, as_index=False).agg(
            n_batteries=("battery_id", "size"), n_windows=("n_windows", "sum"))
        summary = summary.merge(counts, on=["condition", "phase"], validate="one_to_one")
        # Verify that every reported R2 is the arithmetic mean of saved window R2 values.
        windows = pd.DataFrame(window_rows)
        for row in battery.to_dict("records"):
            mask = windows.battery_id.eq(row["battery_id"]) & windows.condition.eq(row["condition"])
            if row["phase"] != "all":
                mask &= windows.phase.eq(row["phase"])
            np.testing.assert_allclose(row["r2"], windows.loc[mask, "r2"].mean(), rtol=1e-10, atol=1e-10)
        battery.to_csv(OUTPUT / f"{PREFIX}_battery_metrics.csv", index=False)
        summary.to_csv(OUTPUT / f"{PREFIX}_summary.csv", index=False)
        windows.to_csv(OUTPUT / f"{PREFIX}_window_metrics.csv", index=False)
        np.savez_compressed(OUTPUT / f"{PREFIX}_predictions.npz", **saved)
        figure = comparison_figure(records, data, predictions, OUTPUT)
        report(OUTPUT, summary, battery, features, figure)
        state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
                     evaluated_windows_per_condition=sum(len(r["windows"]["target"]) for r in records),
                     figures=[figure], baseline_verified=True)
        write_json(state_path, state)
        print(summary[summary.phase == "all"].to_string(index=False), flush=True)
        print(f"COMPLETE {OUTPUT}", flush=True)
    except Exception as error:
        state.update(status="failed", error=str(error))
        write_json(state_path, state)
        raise
    finally:
        del pipeline
        clear_device_cache()


if __name__ == "__main__":
    main()
