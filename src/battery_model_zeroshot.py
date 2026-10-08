"""Independent T5/Chronos-2 zero-shot with standard R2 and observed-period thirds."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from src.battery_zeroshot import (
    ROOT, prepare_battery_windows, baseline_predictions,
    WQL_QUANTILES, sha256, write_json,
)
from src.soh_data import load_bundle
from src.soh_evaluation import mean_window_r2, window_r2_values, R2_DEFINITION
from src.soh_experiment import fresh_pipeline, predict_samples, resolve_device, clear_device_cache

SCORES = ["rmse", "r2", "wql", "picp80", "width80"]
PHASES = ["early", "middle", "late"]
PILOT = {"mit_batch3": ("b3c0", "b3c1", "b3c3", "b3c4"),
         "xjtu_3c": ("batch2_3C_battery-1", "batch2_3C_battery-2")}
PROTOCOL = {
    "r2": R2_DEFINITION,
    "phase": "Three equal cycle spans from first to last observed cycle of each measurement record. Each window belongs to the phase of its forecast start; its full horizon is scored. Boundary-crossing horizons stay in their start phase.",
    "aggregation": "Arithmetic mean of per-window R2 within battery/phase, then equal battery-record macro. Other metrics retain their original aggregation. Overlapping windows are not independent; no pooled score across datasets.",
    "prediction": "Median (q0.5); WQL at q0.1,...,q0.9; PICP@80 at q0.1/q0.9. No training, clipping, interpolation, smoothing or future target input. Chronos-2 cross_learning=False; one univariate task per window.",
}


def model_tag(config):
    if config.model_id == "amazon/chronos-t5-base":
        return "t5-battery"
    if config.model_id == "amazon/chronos-2":
        return "c2-battery"
    raise ValueError("This comparison supports chronos-t5-base and chronos-2")


def phase_labels(starts, cycles):
    """Cycle spans, independent of targets, forecasts and eligible-window density."""
    starts, cycles = np.asarray(starts), np.asarray(cycles)
    if not cycles.size or not starts.size or cycles.min() > starts.min() or cycles.max() < starts.max():
        raise ValueError("Forecast starts must lie inside the observed measurement period")
    positions = np.floor(3 * (starts - cycles.min()) / (cycles.max() - cycles.min() + 1)).astype(int)
    return np.asarray(PHASES)[np.minimum(positions, 2)]


def quantile_scores(target, median, quantiles=None):
    target, median = np.asarray(target, dtype=np.float64), np.asarray(median, dtype=np.float64)
    if target.shape != median.shape or not target.size or not np.isfinite(target).all() or not np.isfinite(median).all():
        raise ValueError("Targets and median predictions must match and be finite")
    r2_values = window_r2_values(target, median)
    result = {"rmse": float(np.sqrt(np.square(target - median).mean())), "r2": mean_window_r2(target, median),
              "r2_n_windows": int(np.isfinite(r2_values).sum()), "r2_undefined_windows": int(np.isnan(r2_values).sum()),
              **{key: float("nan") for key in ["wql", "picp80", "width80"]}}
    if quantiles is not None:
        values = np.asarray(quantiles, dtype=np.float64)
        if values.shape != (*target.shape, 9) or not np.isfinite(values).all():
            raise ValueError("Expected nine finite quantiles [windows, horizon, quantiles]")
        # Do not sort, clip or repair crossing quantiles: expose an invalid interval.
        if np.any(values[..., 0] > values[..., -1]):
            raise ValueError("Crossed 10-90% prediction interval")
        q = np.asarray(WQL_QUANTILES)
        error = target[..., None] - values
        denominator = np.abs(target).sum()
        loss = 2 * np.maximum(q * error, (q - 1) * error).sum(axis=(0, 1)).mean()
        result.update(wql=float(loss / denominator) if denominator else float("nan"),
                      picp80=float(((target >= values[..., 0]) & (target <= values[..., -1])).mean()),
                      width80=float((values[..., -1] - values[..., 0]).mean()))
    return result


def load_model(config):
    if model_tag(config) == "t5-battery":
        return fresh_pipeline(config, config.inference_seeds[0])
    # Standardize input and restore forecasts in float32; avoid raw-SOH bf16 rounding.
    pipeline = Chronos2Pipeline.from_pretrained(
        config.model_id, revision=config.revision, device_map=resolve_device(config.device),
        dtype=torch.float32, cache_dir=str(ROOT / "models/huggingface"), local_files_only=True,
    )
    pipeline.model.eval()
    return pipeline


def predict_quantiles(pipeline, context, config):
    if model_tag(config) == "t5-battery":
        samples = predict_samples(pipeline, context, config, config.inference_seeds[0])
        values = np.quantile(samples.astype(np.float64), WQL_QUANTILES, axis=1, method="linear")
        # Preserve the original float32 sample-median point estimator.
        return np.median(samples, axis=1), values.transpose(1, 2, 0)
    with torch.inference_mode():
        # A list of 1D contexts gives each rolling window its own univariate group.
        values, median = pipeline.predict_quantiles(
            inputs=[np.asarray(row, dtype=np.float32) for row in context],
            prediction_length=config.prediction_length, context_length=config.context_length,
            quantile_levels=list(WQL_QUANTILES), batch_size=config.inference_batch_size,
            cross_learning=False,
        )
    quantiles = np.stack([value.cpu().numpy()[0] for value in values])
    point = np.stack([value.cpu().numpy()[0] for value in median])
    if quantiles.shape != (len(context), config.prediction_length, 9) or point.shape != (len(context), config.prediction_length):
        raise ValueError("Unexpected Chronos-2 quantile axes or task count")
    return point, quantiles


def battery_rows(record, labels, median, quantiles, tag):
    windows = record["windows"]
    tags = {key: record[key] for key in ["dataset", "battery_id", "physical_battery_id"]}
    rows = []
    forecasts = [(name, point, None) for name, point in baseline_predictions(windows).items()]
    forecasts.append((tag, median, quantiles))
    for phase in ["all", *PHASES]:
        mask = np.ones(len(labels), dtype=bool) if phase == "all" else labels == phase
        if not mask.any():
            continue
        starts = windows["forecast_start"][mask]
        for name, point, values in forecasts:
            rows.append({**tags, "model": name, "phase": phase, "n_windows": int(mask.sum()),
                         "first_forecast_cycle": int(starts.min()), "last_forecast_cycle": int(starts.max()),
                         **quantile_scores(windows["target"][mask], point[mask], None if values is None else values[mask])})
    return rows


def write_tables(directory, tag, rows):
    battery = pd.DataFrame(rows)
    summary = battery.groupby(["model", "phase"], sort=False, as_index=False)[SCORES].mean()
    counts = battery.groupby(["model", "phase"], sort=False).agg(n_batteries=("battery_id", "size"),
                                                               n_windows=("n_windows", "sum"), n_r2=("r2", "count")).reset_index()
    summary = summary.merge(counts, on=["model", "phase"], validate="one_to_one")
    if "r2_n_windows" in battery:
        valid = battery.groupby(["model", "phase"], sort=False, as_index=False)[["r2_n_windows", "r2_undefined_windows"]].sum()
        summary = summary.merge(valid, on=["model", "phase"], validate="one_to_one")
    battery.to_csv(directory / f"{tag}_battery_metrics.csv", index=False)
    summary.to_csv(directory / f"{tag}_summary.csv", index=False)
    return summary


def verify_result(directory, tag):
    directory = Path(directory)
    manifest = json.loads((directory / f"{tag}_run.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["protocol"] != PROTOCOL:
        raise ValueError("Incomplete or incompatible phase evaluation")
    battery = pd.read_csv(directory / f"{tag}_battery_metrics.csv", keep_default_na=False, na_values=[""])
    summary = pd.read_csv(directory / f"{tag}_summary.csv")
    expected = battery.groupby(["model", "phase"], sort=False, as_index=False)[SCORES].mean()
    pd.testing.assert_frame_equal(summary[["model", "phase", *SCORES]], expected, check_dtype=False, rtol=1e-10, atol=1e-12)
    all_rows = battery[battery.phase == "all"]
    if any(set(group.model) != {tag, "LastValue", "Linear"} or len(group) != 3
           for _, group in all_rows.groupby(["dataset", "battery_id"])):
        raise ValueError("Missing or duplicate model scores")
    counts = all_rows[all_rows.model == tag]
    if len(counts) != manifest["batteries_evaluated"] or counts.n_windows.sum() != manifest["evaluated_windows"]:
        raise ValueError("Evaluation coverage differs")
    if not np.isfinite(battery.rmse).all() or np.isinf(battery.r2).any() or (battery.r2.dropna() > 1).any():
        raise ValueError("Invalid point metrics")
    probability = battery[battery.model == tag]
    if (not np.isfinite(probability[["wql", "picp80", "width80"]]).all().all()
            or not probability.picp80.between(0, 1).all() or (probability.width80 < 0).any()):
        raise ValueError("Invalid probability metrics")
    for battery_id, group in battery[battery.model == tag].groupby("battery_id"):
        if group[group.phase != "all"].n_windows.sum() != group[group.phase == "all"].n_windows.iloc[0]:
            raise ValueError(f"Phase window counts do not partition {battery_id}")
    if "r2_n_windows" in battery:
        if not ((battery.r2_n_windows + battery.r2_undefined_windows) == battery.n_windows).all():
            raise ValueError("R2 valid/undefined window counts differ")
        for _, group in battery.groupby(["dataset", "battery_id", "model"]):
            phases, total = group[group.phase != "all"], group[group.phase == "all"].iloc[0]
            if phases.r2_n_windows.sum() != total.r2_n_windows:
                raise ValueError("R2 phase valid counts do not partition all windows")
            if total.r2_n_windows:
                expected_r2 = (phases.r2.fillna(0) * phases.r2_n_windows).sum() / total.r2_n_windows
                np.testing.assert_allclose(total.r2, expected_r2, rtol=1e-10, atol=1e-10)
    if not manifest["figures"] or any(not (directory / name).is_file() for name in manifest["figures"]):
        raise ValueError("Missing representative curves")
    if not (directory / f"{tag}_report.md").is_file():
        raise ValueError("Missing report")
    if manifest.get("run_scope") == "full":
        coverage = pd.read_csv(directory / f"{tag}_coverage.csv", keep_default_na=False)
        if (coverage.evaluated_windows.sum() != manifest["evaluated_windows"]
                or (coverage.evaluated_windows == 0).sum() != manifest["batteries_skipped"]):
            raise ValueError("Full-suite evaluated/skipped coverage differs")
        expected_counts = probability.pivot(index="battery_id", columns="phase", values="n_windows").fillna(0)
        for row in coverage[coverage.evaluated_windows > 0].to_dict("records"):
            for phase in PHASES:
                if row[f"{phase}_windows"] != expected_counts.loc[row["battery_id"]].get(phase, 0):
                    raise ValueError("Phase coverage differs from evaluated metrics")
    return manifest


def write_report(directory, tag, config, summary, figures, scope="pilot", coverage=None):
    lines = [f"# {tag}: {directory.name} zero-shot", "",
             f"모델 `{config.model_id}`, 관측 {config.context_length} → 예측 {config.prediction_length}, stride {config.stride}.", "",
             "각 배터리 측정 기록의 관측 cycle 범위를 3등분한다. 예측 시작 cycle로 초·중·후반을 배정하고 해당 window의 전체 horizon을 평가한다. "
             "경계를 넘는 horizon도 시작 구간에 포함한다. 실제 전체 수명 3등분을 뜻하지 않는다.", "",
             "Mean window R²: 한 예측창의 미래 target 평균을 기준으로 R² = 1 − SSE/SST를 계산한 뒤 "
             "각 배터리·구간 또는 전체에서 유효한 창별 R²를 단순평균한다. 이후 배터리 점수를 동일 가중 평균한다. "
             "음수는 유지하고 상수 target 창은 NaN으로 제외한다. r2_n_windows는 유효 창 수, r2_undefined_windows는 제외 창 수다. "
             "전체 점수는 전체 창의 평균이며 초·중·후반 평균을 단순평균하지 않는다. "
             "실제값이 거의 일정하면 R²가 크게 음수여도 계산 오류가 아니므로 RMSE·곡선을 함께 본다.", "",
             "| 모델 | 구간 | RMSE ↓ | R² ↑ | WQL ↓ | PICP@80 | 80% 구간 폭 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        values = ["—" if not np.isfinite(row[key]) else (f"{row[key]:.1%}" if key == "picp80" else f"{row[key]:.6f}") for key in SCORES]
        lines.append(f"| {row['model']} | {row['phase']} | " + " | ".join(values) + " |")
    lines += ["", "WQL은 q0.1~q0.9의 9개 twice-pinball loss를 평균하고 구간 target 절댓값 합으로 정규화한다. "
              "PICP@80은 q0.1~q0.9 포함률이며 명목값은 80%다. RMSE와 구간 폭은 SOH 비율 단위다.", "",
              "최근 실제 관측으로 매번 예측하는 rolling zero-shot이다. 학습·clipping·평활화 없이 같은 입력과 window를 사용한다. "
              "C2는 독립 단변량 입력과 cross_learning=False를 사용한다. T5는 100 samples, seed 42; C2는 직접 분위수 예측으로 sampling 설정을 사용하지 않는다.", "",
              "대표 곡선은 각 구간 중앙에 가장 가까운 유효 예측창이다. 그래프의 R²는 표시된 window 한 개의 점수이며 구간 전체 점수와 다르다. "
              "겹치는 window는 독립 표본이 아니다. " +
              ("입력 묶음의 모든 평가 가능한 배터리와 window를 평가했다."
               if scope == "full" else "일부 배터리만 평가한 pilot이며 전체 데이터셋 성능으로 해석하지 않는다."), "",
              f"[설정]({tag}_run.json) · [배터리별 점수]({tag}_battery_metrics.csv)", ""]
    if scope == "full":
        skipped = coverage[coverage.evaluated_windows == 0]
        lines += [f"평가 {int((coverage.evaluated_windows > 0).sum())}개 기록, 제외 {len(skipped)}개 기록, "
                  f"총 {int(coverage.evaluated_windows.sum()):,}개 window. [평가 가능 구간]({tag}_coverage.csv).", "",
                  "일부 기록은 초반 1/3 안에서 context 길이를 확보할 수 없어 초반 점수가 없다. "
                  "표에서 해당 구간의 n_batteries가 줄어들며 그림에는 평가 가능한 창이 없다고 표시한다.", ""]
        if len(skipped):
            lines += ["유효 창이 없어 제외한 기록: " + ", ".join(skipped.battery_id.astype(str)), ""]
    if config.calce_cross_file_windows and directory.name == "calce_cs2_1c":
        lines += ["## CALCE 관측 순서와 파일 경계", "",
                  "같은 배터리의 시간순 관측 기록을 파일 경계를 넘어 사용한다. 원본 파일·시각·source cycle은 보존하며 "
                  "누락 cycle, 파일 내부 source cycle 누락, protocol 변경, 불완전 방전 및 품질 제외 관측을 건너 창을 만들지 않는다. "
                  "가로축은 기존 추출본의 관측 기록 순번이다. 파일 사이 달력 시간의 휴지는 별도 입력하지 않으며 "
                  "연속된 실제 운전 시간이나 미기록 물리 cycle이 없음을 입증한 결과는 아니다. "
                  "기존 파일별 32→8 결과와는 horizon과 평가 창 집합이 달라 직접적인 성능 개선 실험으로 해석하지 않는다.", ""]
    lines += [f"![대표 SOH 곡선]({name})\n" for name in figures]
    (directory / f"{tag}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_model_battery(bundle, config, output_root=None, pipeline=None, scope="pilot"):
    tag = model_tag(config)
    if scope not in {"pilot", "full"}:
        raise ValueError("Choose pilot or full scope")
    if scope == "full" and (config.battery_ids is not None or config.max_windows_per_battery is not None):
        raise ValueError("Full evaluation must not restrict batteries or windows")
    if len(config.inference_seeds) != 1:
        raise ValueError("This pilot uses one T5 seed or deterministic C2 quantiles")
    bundle = Path(bundle).resolve()
    root = Path(output_root or ROOT / "results" / tag / scope).resolve()
    if root == bundle or root in bundle.parents or bundle in root.parents:
        raise ValueError("Forecast outputs must be separate from input data")
    directory = root / bundle.name
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / f"{tag}_run.json"
    records, coverage, data, _ = prepare_battery_windows(bundle, config)
    if not records:
        raise ValueError("No eligible forecast windows in the input bundle")
    if scope == "pilot" and (coverage.evaluated_windows == 0).any():
        raise ValueError("Every selected pilot battery must have eligible windows")
    labels_by_id, cycles_by_id = {}, {}
    for phase in PHASES:
        coverage[f"{phase}_windows"] = 0
    for record in records:
        series = data[(data.dataset == record["dataset"]) & (data.battery_id == record["battery_id"]) & data.is_observed]
        cycles = series.cycle.to_numpy()
        labels = phase_labels(record["windows"]["forecast_start"], cycles)
        labels_by_id[record["record_id"]], cycles_by_id[record["record_id"]] = labels, cycles
        mask = (coverage.dataset == record["dataset"]) & (coverage.battery_id == record["battery_id"])
        for phase in PHASES:
            coverage.loc[mask, f"{phase}_windows"] = int((labels == phase).sum())
    identity = {"config": json.loads(json.dumps(asdict(config))), "bundle": str(bundle),
                "bundle_fingerprint": sha256(bundle / "manifest.json"), "protocol": PROTOCOL}
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if previous and any(previous.get(key) != value for key, value in identity.items()):
        raise ValueError("Output has different inputs/settings; use a separate output root")
    if previous and previous.get("run_scope", "pilot") != scope:
        raise ValueError("Output has a different evaluation scope")
    if previous and previous["status"] == "complete":
        verify_result(directory, tag)
        print(f"REUSE {tag}/{bundle.name}", flush=True)
        return directory
    manifest = {**identity, "status": "running", "model": config.model_id, "device": resolve_device(config.device),
                "started_at": (previous or {}).get("started_at", datetime.now(timezone.utc).isoformat()),
                "run_scope": scope,
                "scope": f"{scope}; eligible windows at stride {config.stride}" if config.max_windows_per_battery is None else "window-limited smoke",
                "prediction_method": "sample median" if tag == "t5-battery" else "direct quantiles, float32, no sampling",
                "batteries_completed": 0}
    write_json(manifest_path, manifest)
    if scope == "full":
        coverage.to_csv(directory / f"{tag}_coverage.csv", index=False)
    rows, plots = [], []
    completed = set()
    metric_path = directory / f"{tag}_battery_metrics.csv"
    if previous and metric_path.is_file():
        partial = pd.read_csv(metric_path, keep_default_na=False, na_values=[""])
        for record in records:
            checkpoint = directory / f".forecast_{record['record_id']}.npz"
            scores = partial[(partial.dataset == record["dataset"]) & (partial.battery_id == record["battery_id"])]
            if checkpoint.is_file() and set(scores.model) == {tag, "LastValue", "Linear"}:
                with np.load(checkpoint, allow_pickle=False) as saved:
                    plots.append({**record, "forecast": {key: saved[key].copy() for key in saved.files}})
                rows.extend(scores.to_dict("records"))
                completed.add(record["record_id"])
    owned = pipeline is None
    try:
        if owned:
            pipeline = load_model(config)
        for record in records:
            if record["record_id"] in completed:
                continue
            windows = record["windows"]
            labels = labels_by_id[record["record_id"]]
            cycles = cycles_by_id[record["record_id"]]
            tick = time.perf_counter()
            print(f"PREDICT {tag}/{bundle.name}/{record['battery_id']} windows={len(labels)}", flush=True)
            median, quantiles = predict_quantiles(pipeline, windows["context"], config)
            rows.extend(battery_rows(record, labels, median, quantiles, tag))
            centers = cycles.min() + (np.arange(3) + .5) * (cycles.max() - cycles.min() + 1) / 3
            indices, available = [], []
            for phase, center in zip(PHASES, centers):
                eligible = np.flatnonzero(labels == phase)
                available.append(bool(len(eligible)))
                # The placeholder row is never plotted or scored as this phase.
                indices.append(eligible[np.abs(windows["forecast_start"][eligible] - center).argmin()] if len(eligible) else 0)
            forecast = {**{key: value[indices].copy() for key, value in windows.items()},
                        "median": median[indices].copy(), "q10": quantiles[indices, :, 0].copy(),
                        "q90": quantiles[indices, :, -1].copy(), "phase_available": np.asarray(available)}
            np.savez_compressed(directory / f".forecast_{record['record_id']}.npz", **forecast)
            plots.append({**record, "forecast": forecast})
            write_tables(directory, tag, rows)
            manifest["batteries_completed"] = len(plots)
            write_json(manifest_path, manifest)
            print(f"DONE {tag}/{bundle.name}/{record['battery_id']} elapsed={time.perf_counter()-tick:.1f}s", flush=True)
        if sha256(bundle / "manifest.json") != identity["bundle_fingerprint"]:
            raise ValueError("Input bundle changed during inference")
        load_bundle(bundle)
        summary = write_tables(directory, tag, rows)
        from src.battery_zeroshot_plots import forecast_figures
        plots.sort(key=lambda record: record["record_id"])
        figures = forecast_figures(directory, plots, config, show_window_scores=True, model_label=tag,
                                   filename_prefix=tag + "_", representative_description="Window nearest each observed-period third midpoint")
        write_report(directory, tag, config, summary, figures, scope, coverage)
        manifest.update(status="complete", batteries_evaluated=len(records),
                        batteries_skipped=int((coverage.evaluated_windows == 0).sum()),
                        evaluated_windows=int(coverage.evaluated_windows.sum()), figures=figures,
                        finished_at=datetime.now(timezone.utc).isoformat())
        write_json(manifest_path, manifest)
        verify_result(directory, tag)
        for record in records:
            (directory / f".forecast_{record['record_id']}.npz").unlink()
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(manifest_path, manifest)
        raise
    finally:
        if owned:
            del pipeline
            clear_device_cache()
    return directory


def write_model_suite(output_root, tag, groups):
    """Index completed full groups without pooling their different conditions."""
    root = Path(output_root)
    frames, links = [], []
    for group in groups:
        manifest = verify_result(root / group, tag)
        frame = pd.read_csv(root / group / f"{tag}_summary.csv")
        frame.insert(0, "group", group)
        for position, key in enumerate(["context_length", "prediction_length", "stride"], 1):
            frame.insert(position, key, manifest["config"][key])
        frames.append(frame)
        links.append(f"- [{group}]({group}/{tag}_report.md)")
    if not frames:
        return
    pd.concat(frames, ignore_index=True).to_csv(root / f"{tag}_summary.csv", index=False)
    (root / f"{tag}_report.md").write_text(f"# {tag}: 전체 공통 SOH 데이터 평가\n\n" + "\n".join(links) +
        "\n\n완료된 그룹만 표시한다. 전체·초/중/후반 R²는 창별 표준 R²의 단순평균 후 배터리 동일 가중 평균이다. "
        "NASA 등 짧은 기록은 일부 구간 점수가 없으며 n_batteries/n_r2를 확인한다. "
        "CALCE는 32→8, 나머지는 64→16, 모두 stride 1이다. 데이터셋 사이 점수를 하나로 합치지 않는다.\n", encoding="utf-8")


def write_comparison(root=ROOT / "results"):
    summaries, completed, manifests = [], [], {}
    for tag in ["t5-battery", "c2-battery"]:
        for group in PILOT:
            directory = Path(root) / tag / "pilot" / group
            if not (directory / f"{tag}_run.json").is_file():
                continue
            manifest = json.loads((directory / f"{tag}_run.json").read_text(encoding="utf-8"))
            if manifest["status"] != "complete":
                continue
            verify_result(directory, tag)
            manifests[tag, group] = manifest
            completed.append((tag, group))
            frame = pd.read_csv(directory / f"{tag}_summary.csv")
            frame.insert(0, "group", group)
            frame.insert(1, "run_model", tag)
            summaries.append(frame)
    if not summaries:
        return
    for group in PILOT:
        if all((tag, group) in manifests for tag in ["t5-battery", "c2-battery"]):
            left, right = (manifests[tag, group] for tag in ["t5-battery", "c2-battery"])
            common = ["context_length", "prediction_length", "stride", "battery_ids", "max_windows_per_battery"]
            if (left["bundle"] != right["bundle"] or left["bundle_fingerprint"] != right["bundle_fingerprint"]
                    or any(left["config"][key] != right["config"][key] for key in common)):
                raise ValueError(f"Unmatched T5/C2 inputs or window settings: {group}")
            baseline = []
            for tag in ["t5-battery", "c2-battery"]:
                frame = pd.read_csv(Path(root) / tag / "pilot" / group / f"{tag}_battery_metrics.csv", keep_default_na=False, na_values=[""])
                baseline.append(frame[frame.model.isin(["LastValue", "Linear"])].sort_values(["battery_id", "model", "phase"]).reset_index(drop=True))
            pd.testing.assert_frame_equal(*baseline, check_dtype=False, rtol=1e-10, atol=1e-12)
    directory = Path(root) / "t5-c2-battery-comparison"
    directory.mkdir(parents=True, exist_ok=True)
    # Baselines remain associated with their run, making input parity reviewable.
    pd.concat(summaries, ignore_index=True).to_csv(directory / "t5-c2-battery_summary.csv", index=False)
    links = [f"- [{tag}/{group}](../{tag}/pilot/{group}/{tag}_report.md)" for tag, group in completed]
    pending = [f"{tag}/{group}" for tag in ["t5-battery", "c2-battery"] for group in PILOT if (tag, group) not in completed]
    (directory / "README.md").write_text("# T5 / Chronos-2 battery zero-shot pilot\n\n" + "\n".join(links) +
        "\n\n동일한 일부 배터리·64→16·stride 1 비교다. Mean window R²와 관측 기간 초·중·후반 점수를 사용한다. "
        "창별 R²를 단순평균한 뒤 배터리를 동일 가중 평균한다. 전체 T5 결과도 같은 정의로 갱신한다. 완료된 그룹만 CSV에 포함되며 데이터셋 사이 점수를 합치지 않는다.\n\n"
        + ("진행 예정/진행 중: " + ", ".join(pending) if pending else "두 모델의 모든 pilot 그룹이 완료됐고 입력 조건과 단순 기준 점수의 일치 여부를 확인했다.") + "\n", encoding="utf-8")
