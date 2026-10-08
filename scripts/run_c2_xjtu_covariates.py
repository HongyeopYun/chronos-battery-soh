"""Small fixed-selection XJTU 2C/3C zero-shot voltage/current comparison."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd

from scripts.run_c2_nasa_covariates import CONDITIONS, covariate_predictions, comparison_figure
from src.battery_model_zeroshot import load_model, predict_quantiles, quantile_scores, phase_labels, PHASES, SCORES
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, sha256, write_json
from src.nasa_covariates import FEATURES
from src.xjtu_covariates import extract_xjtu_features
from src.soh_evaluation import R2_DEFINITION, window_r2_values
from src.soh_experiment import clear_device_cache, resolve_device


def write_report(directory, prefix, group, summary, battery, features, config, figure):
    overall = summary[summary.phase == "all"].set_index("condition")
    base = overall.loc["SOH"]
    lines = [f"# Chronos-2 {group}: 과거 전압·전류 추가", "",
             "각 충전율에서 파일 번호 1~3번 배터리를 예측 결과와 무관하게 미리 선택했다. "
             "선택한 배터리의 유효 rolling window를 전부 평가했으며, XJTU 전체 배터리 결과는 아니다.", "",
             f"과거 {config.context_length} cycle → 미래 {config.prediction_length} cycle SOH, stride {config.stride}. "
             "SOH-only / +전압 / +전류 / +전압·전류를 같은 창에서 비교한다. "
             "기존 SOH 정규화·마스크·구간 구분을 유지하며 추가 학습은 하지 않았다.", "",
             "원본 cycle에서 current_A < −0.1 A인 단일 연속 방전 구간을 분리하고 relative_time_min을 초로 변환했다. "
             "각 방전의 시작 후 0~600초에서 시간 가중 평균 전압(V)·전류(A)를 추출한다. "
             "600초 경계는 선형 보간한다. 충전·휴지 구간이나 방전 전체 길이·capacity 적분은 특징에 사용하지 않는다.", "",
             f"동일 방전 내 중복 시각 {int(features.duplicate_discharge_timestamps.sum())}개는 해당 시각의 마지막 센서 표본을 유지했다. "
             "시간 역전이나 방전 구간의 모호성은 자동 연결하지 않고 오류로 처리한다.", "",
             "SOH와 센서 특징은 같은 과거 cycle에 맞춰 past_covariates로 입력한다. "
             "미래 센서값은 입력하지 않는다. C2 cross_learning=False로 서로 다른 예측창은 독립된 그룹이다. "
             "외부 스케일링·클리핑·평활화 없이 C2 내부 정규화를 사용한다.", "",
             "| 입력 | RMSE ↓ | 평균 window R² ↑ | WQL ↓ | PICP@80 | RMSE 변화 |",
             "|---|---:|---:|---:|---:|---:|"]
    for condition, row in overall.iterrows():
        lines.append(f"| {condition} | {row.rmse:.6f} | {row.r2:.4f} | {row.wql:.6f} | {row.picp80:.1%} | "
                     f"{(row.rmse/base.rmse-1)*100:+.2f}% |")
    lines += ["", "## 배터리별 비교", "",
              "| 배터리 | 창 수 | SOH RMSE | SOH+V+I RMSE | 변화 | SOH R² | SOH+V+I R² |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for battery_id, rows in battery[battery.phase == "all"].groupby("battery_id", sort=False):
        rows = rows.set_index("condition")
        a, b = rows.loc["SOH"], rows.loc["SOH+V+I"]
        lines.append(f"| {battery_id} | {int(a.n_windows)} | {a.rmse:.6f} | {b.rmse:.6f} | "
                     f"{(b.rmse/a.rmse-1)*100:+.2f}% | {a.r2:.4f} | {b.r2:.4f} |")
    lines += ["", "## 정상 프로토콜에서 특징 범위", "",
              "초기 기준 capacity 측정 cycle 1은 정상 방전과 전류 조건이 다르므로 아래 범위에서 제외한다. "
              "실제 예측창은 기존 유효 관측 마스크를 사용해 초기 기준 측정을 제외한다.", "",
              "| 배터리 | 평균 전압 범위 (V) | 평균 전류 범위 (A) | 전류 표준편차 (A) |",
              "|---|---:|---:|---:|"]
    for battery_id, rows in features[features.cycle > 1].groupby("battery_id", sort=False):
        lines.append(f"| {battery_id} | {rows[FEATURES[0]].min():.6f}~{rows[FEATURES[0]].max():.6f} | "
                     f"{rows[FEATURES[1]].min():.6f}~{rows[FEATURES[1]].max():.6f} | {rows[FEATURES[1]].std():.8f} |")
    lines += ["", "R²는 각 16-cycle 창에서 표준 1−SSE/SST로 계산 후 배터리 내 유효 창 단순 평균, "
              "선택한 세 배터리 동일 가중 평균이다. pooled R²는 사용하지 않는다. "
              "다른 지표도 배터리별 계산 후 동일 가중 평균한다. "
              "초·중·후반은 기존 관측 cycle 범위의 1/3을 예측 시작 cycle로 나눈다.", "",
              "과거 SOH-only 점수와 모든 조건의 window R² 평균을 교차 확인했다. "
              "다른 센서 특징이나 fine-tuning의 가능성으로 일반화하지 않는다. "
              "거의 일정한 전류의 미세한 변동이 입력에서 영향을 줄 수 있으며 개선 자체가 물리 관계 학습의 증거는 아니다. "
              "겹치는 창에 독립 표본 가정의 유의성 검정은 하지 않았다.", "",
              f"![초·중·후반 비교]({figure})", "",
              f"[구간별 점수]({prefix}_summary.csv) · [배터리별 점수]({prefix}_battery_metrics.csv)"]
    (directory / f"{prefix}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_group(group, pipeline):
    source = ROOT / f"results/c2-battery/full/{group}"
    original = json.loads((source / "c2-battery_run.json").read_text(encoding="utf-8"))
    batch, rate = ("batch1", "2C") if group == "xjtu_2c" else ("batch2", "3C")
    ids = tuple(f"{batch}_{rate}_battery-{i}" for i in [1, 2, 3])
    config = replace(ZeroShotConfig(**original["config"]), battery_ids=ids)
    records, _, all_data, _ = prepare_battery_windows(original["bundle"], config)
    data = all_data[all_data.battery_id.isin(ids)]
    if len(records) != 3:
        raise ValueError("Missing selected battery")
    directory = ROOT / f"results/c2-battery/covariates/{group}"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"c2-battery_{group}_vi"
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
             "group": group, "battery_ids": ids, "selection": "File numbers 1,2,3; fixed independently of performance",
             "scope": "Selected batteries; all eligible rolling windows; not full XJTU batch",
             "config": {**original["config"], "battery_ids": ids}, "device": resolve_device(config.device),
             "conditions": CONDITIONS, "feature_interval_seconds": [0, 600],
             "feature_aggregation": "Time-weighted voltage_V/current_A mean in a single current_A<-0.1 discharge segment; minutes converted to seconds; linear boundary interpolation; duplicate times keep last sample",
             "input": "Past covariates only; one SOH target; cross_learning=False; no fine-tuning",
             "r2": R2_DEFINITION, "bundle_fingerprint": original["bundle_fingerprint"],
             "raw_sha256": {b: sha256(ROOT / "data/raw/XJTU" / (b.split("_", 1)[1] + ".mat")) for b in ids},
             "completed_tasks": 0}
    state_path = directory / f"{prefix}_run.json"
    write_json(state_path, state)
    try:
        features = extract_xjtu_features(ROOT / "data/raw/XJTU", data)
        state["duplicate_discharge_timestamps"] = int(features.duplicate_discharge_timestamps.sum())
        features.to_csv(directory / f"{prefix}_cycle_features.csv", index=False)
        expected = pd.read_csv(source / "c2-battery_battery_metrics.csv")
        rows, window_rows, predictions, saved = [], [], {}, {}
        for record in records:
            b, w = record["battery_id"], record["windows"]
            if np.any(w["context_cycle"] <= 1):
                raise ValueError("Initial capacity reference cycle should not enter forecasting contexts")
            labels = phase_labels(w["forecast_start"], data[(data.battery_id == b) & data.is_observed].cycle.to_numpy())
            f = features[features.battery_id == b]
            saved[f"{b}_target"], saved[f"{b}_forecast_start"] = w["target"], w["forecast_start"]
            for index, (condition, names) in enumerate(CONDITIONS.items()):
                print(f"RUN {b} {condition}: {len(w['target'])} windows", flush=True)
                point, q = (predict_quantiles(pipeline, w["context"], config) if not names else
                            covariate_predictions(pipeline, w, f, names, config))
                predictions[(b, condition)] = (point, q)
                saved[f"{b}_condition{index}_quantiles"] = q
                for phase in ["all", *PHASES]:
                    mask = np.ones(len(labels), dtype=bool) if phase == "all" else labels == phase
                    if not mask.any():
                        continue
                    score = quantile_scores(w["target"][mask], point[mask], q[mask])
                    rows.append({"battery_id": b, "condition": condition, "phase": phase,
                                 "n_windows": int(mask.sum()), **score})
                    if not names:
                        old = expected[(expected.battery_id == b) & (expected.model == "c2-battery") & (expected.phase == phase)].iloc[0]
                        np.testing.assert_allclose([score[k] for k in SCORES], old[SCORES].to_numpy(dtype=float), rtol=2e-5, atol=5e-8)
                for start, phase, r2, rmse in zip(w["forecast_start"], labels, window_r2_values(w["target"], point),
                                                np.sqrt(np.square(w["target"] - point).mean(axis=1))):
                    window_rows.append({"battery_id": b, "condition": condition, "forecast_start": int(start),
                                        "phase": phase, "rmse": float(rmse), "r2": float(r2)})
                state["completed_tasks"] += 1
                write_json(state_path, state)
        battery, windows = pd.DataFrame(rows), pd.DataFrame(window_rows)
        summary = battery.groupby(["condition", "phase"], sort=False, as_index=False)[SCORES].mean()
        counts = battery.groupby(["condition", "phase"], sort=False, as_index=False).agg(
            n_batteries=("battery_id", "size"), n_windows=("n_windows", "sum"))
        summary = summary.merge(counts, on=["condition", "phase"], validate="one_to_one")
        for row in battery.to_dict("records"):
            mask = windows.battery_id.eq(row["battery_id"]) & windows.condition.eq(row["condition"])
            if row["phase"] != "all":
                mask &= windows.phase.eq(row["phase"])
            np.testing.assert_allclose(row["r2"], windows.loc[mask, "r2"].mean(), rtol=1e-10, atol=1e-10)
        for name, table in [("battery_metrics", battery), ("summary", summary), ("window_metrics", windows)]:
            table.to_csv(directory / f"{prefix}_{name}.csv", index=False)
        np.savez_compressed(directory / f"{prefix}_predictions.npz", **saved)
        figure = comparison_figure(records, data, predictions, directory, dataset_name=group, prefix=prefix, first_eligible=False)
        write_report(directory, prefix, group, summary, battery, features, config, figure)
        state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
                     evaluated_windows_per_condition=sum(len(r["windows"]["target"]) for r in records),
                     figures=[figure], baseline_verified=True)
        write_json(state_path, state)
        print(summary[summary.phase == "all"].to_string(index=False), flush=True)
        return summary.assign(group=group)
    except Exception as error:
        state.update(status="failed", error=str(error))
        write_json(state_path, state)
        raise


def main():
    config = ZeroShotConfig(**json.loads((ROOT / "results/c2-battery/full/xjtu_2c/c2-battery_run.json").read_text(encoding="utf-8"))["config"])
    print("Loading local Chronos-2; selected XJTU 2C/3C batteries 1,2,3", flush=True)
    pipeline = load_model(config)
    try:
        summaries = [run_group(group, pipeline) for group in ["xjtu_2c", "xjtu_3c"]]
        pd.concat(summaries, ignore_index=True).to_csv(ROOT / "results/c2-battery/covariates/c2-battery_xjtu_vi_summary.csv", index=False)
        print("COMPLETE selected XJTU voltage/current comparison", flush=True)
    finally:
        del pipeline
        clear_device_cache()


if __name__ == "__main__":
    main()
