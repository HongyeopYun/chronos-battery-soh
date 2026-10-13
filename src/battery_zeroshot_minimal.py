"""Compact zero-shot results: score every window, keep tables and figures only."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import src.battery_zeroshot as core
from src.soh_evaluation import mean_window_r2, window_r2_values, R2_DEFINITION
from src.soh_experiment import fresh_pipeline, predict_samples, clear_device_cache, resolve_device

SCORES = ["rmse", "r2", "wql", "picp80", "width80"]
GROUPS = ["nasa_aging", "calce_cs2_1c", "xjtu_2c", "xjtu_3c", "mit_batch2", "mit_batch1", "mit_batch3"]


def compact_point_scores(target, prediction):
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Point predictions must match finite targets")
    rmse = float(np.sqrt(np.square(target - prediction).mean()))
    values = window_r2_values(target, prediction)
    return {"rmse": rmse, "r2": mean_window_r2(target, prediction),
            "r2_n_windows": int(np.isfinite(values).sum()), "r2_undefined_windows": int(np.isnan(values).sum())}


def compact_scores(target, samples):
    """Normalize WQL within a battery/seed, using all its selected targets."""
    summary = core.summarize_samples(samples)
    intervals = core.summarize_samples(np.asarray(samples, dtype=np.float64))
    scores = {
        **compact_point_scores(target, summary["median"]),
        "wql": core.weighted_quantile_loss(target, samples),
        "picp80": float(((target >= intervals["q10"]) & (target <= intervals["q90"])).mean()),
        "width80": float((intervals["q90"] - intervals["q10"]).mean()),
    }
    return scores, summary


def saved_config(manifest):
    return core.ZeroShotConfig(**manifest["config"])


def verify_compact_result(directory, require_r2=True):
    """Validate the retained result package; raw samples are intentionally absent."""
    directory = Path(directory)
    manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Compact zero-shot result is incomplete")
    battery = pd.read_csv(directory / "battery_metrics.csv", keep_default_na=False, na_values=[""])
    summary = pd.read_csv(directory / "summary.csv")
    ready = battery[battery.status == "evaluated"]
    models = {"Chronos", "LastValue", "Linear"}
    counts = ready.groupby(list(core.IDENTITY[:3])).model.agg(list)
    if any(len(items) != 3 or set(items) != models for items in counts):
        raise ValueError("Missing or duplicate model scores for an evaluated battery")
    score_columns = SCORES if require_r2 else [key for key in SCORES if key != "r2"]
    expected = ready.groupby("model", as_index=False)[score_columns].mean()
    pd.testing.assert_frame_equal(summary[["model", *score_columns]], expected,
                                  check_dtype=False, rtol=1e-10, atol=1e-12)
    if ready.empty or not np.isfinite(ready.rmse).all():
        raise ValueError("Missing or nonfinite point scores")
    if require_r2 and (manifest.get("r2_definition") != R2_DEFINITION
            or "r2" not in ready or np.isinf(ready.r2).any() or (ready.r2.dropna() > 1).any()):
        raise ValueError("Invalid mean window R2 definition or values")
    probability = ready[ready.model == "Chronos"]
    if (len(probability) != manifest["batteries_evaluated"]
            or probability.n_windows.sum() != manifest["evaluated_windows"]
            or not (probability.n_windows > 0).all()):
        raise ValueError("Battery or window counts differ from the completed run")
    if (not np.isfinite(probability[["rmse", "wql", "picp80", "width80"]]).all().all()
            or not probability.picp80.between(0, 1).all() or (probability.wql < 0).any()
            or not np.isfinite(probability.width80).all() or (probability.width80 < 0).any()):
        raise ValueError("Invalid probability scores")
    for name in manifest["figures"]:
        path = directory / name
        if not path.is_file() or path.stat().st_size < 100:
            raise ValueError(f"Missing forecast figure: {name}")
    if not manifest["figures"] or not (directory / "report.md").is_file():
        raise ValueError("Missing compact report or figures")
    return manifest


def aggregate_batteries(rows):
    return pd.DataFrame(rows).groupby([*core.IDENTITY[:3], "model", "status", "n_windows"],
                                     as_index=False)[[*SCORES, "r2_n_windows", "r2_undefined_windows"]].mean()


def write_compact_report(directory, config, coverage, summary, figures):
    manifest_path = Path(directory) / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    representative = manifest.get("r2_scope") == "representative_windows"
    lines = [f"# {directory.name}: Chronos zero-shot", "",
             f"관측 {config.context_length} → 예측 {config.prediction_length} cycles, stride {config.stride}, "
             f"{config.num_samples} samples, seed {list(config.inference_seeds)}.", "",
             f"평가 배터리 기록 {int((coverage.evaluated_windows > 0).sum())}개, "
             f"평가 window {int(coverage.evaluated_windows.sum()):,}개 / "
             f"유효 window {int(coverage.eligible_windows.sum()):,}개. "
             f"제외 기록 {int((coverage.evaluated_windows == 0).sum())}개.", "",
             "| 모델 | RMSE ↓ | Mean window R² ↑ | WQL ↓ | PICP@80 | 80% 구간 폭 |", "|---|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        values = ["—" if not np.isfinite(row[key]) else
                  (f"{row[key]:.2%}" if key == "picp80" else f"{row[key]:.6f}") for key in SCORES]
        lines.append(f"| {row['model']} | " + " | ".join(values) + " |")
    lines += ["", "Chronos 점예측은 sample 중앙값이다. RMSE·확률 지표는 배터리·seed 안의 모든 window/horizon으로 계산하고, "
              "seed를 평균한 뒤 배터리 기록에 같은 가중치를 준다. WQL은 0.1~0.9의 9개 분위수 손실을 평균하고 "
              "target 절댓값 합으로 정규화한다. PICP@80은 q10~q90의 포함률이며 목표는 80%다. "
              "구간 폭과 RMSE는 SOH 비율 단위다. LastValue/Linear는 같은 context와 target을 사용한다.", "",
              "Mean window R² = mean_w[1 − SSE_w / SST_w]. 각 창의 실제 미래 평균으로 SST_w를 계산한다. "
              "유효 창의 R²를 단순평균한 뒤 seed·배터리 기록을 동일 가중 평균한다. 음수도 그대로 보고한다. "
              "0은 각 window의 실제 미래 평균을 예측한 수준이므로 LastValue의 기준과 다르다. "
              "상수 target 창은 NaN이며 창별 평균에서 제외한다. 작은 분모에 epsilon을 추가하지 않는다. "
              "그래프의 R²는 표시된 1개 window 내부 점수이고 같은 중심화 원칙을 사용한다.", "",
              "그림은 각 배터리의 첫·중간·마지막 유효 평가 window이며 첫 번째 seed를 표시한다. "
              "매 window에 최근 실제 관측을 제공하는 rolling 평가다. 겹치는 window는 독립 표본이 아니다. "
              "다른 context/horizon 설정의 결과는 직접 순위를 매기지 않는다.", "",
              "[배터리별 점수와 제외 기록](battery_metrics.csv) · [설정](run.json)", "",
              "원시 samples·상세 window/horizon CSV·코드/환경 해시는 보관하지 않는다. "
              "따라서 이 결과 패키지로 원시 예측을 재검산하거나 다시 그릴 수는 없다.", ""]
    lines += [f"![대표 예측 {index}]({name})\n" for index, name in enumerate(figures, 1)]
    if representative:
        lines.insert(2, "**R²만 대표 창 평가:** 배터리별 관측 기간의 초·중·후반 중심에 가장 가까운 유효 창을 최대 3개 선택했다. "
                     "R²는 선택한 창별 점수의 단순평균 후 배터리 동일 가중 평균이며 전체 창 점수가 아니다. "
                     "RMSE·WQL·PICP@80·Width80은 기존 전체 평가값을 유지한다. 기존 그림은 이전 전체 실험의 대표 창이다. "
                     "새 대표 창의 점수와 예측은 representative_r2_windows.csv와 representative_r2_predictions.npz에 저장했다.\n")
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def refresh_compact_r2(directory, bundle=None):
    """Never infer arithmetic mean window R2 from aggregate RMSE."""
    directory = Path(directory).resolve()
    original = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if original.get("r2_definition") != R2_DEFINITION:
        raise ValueError("Mean window R2 requires raw forecasts or re-inference; run scripts/unify_window_r2.py")
    manifest = verify_compact_result(directory, require_r2=False)
    config = saved_config(manifest)
    original_bundle = Path(manifest["bundle"]).resolve()
    bundle = Path(bundle).resolve() if bundle is not None else original_bundle
    if bundle != original_bundle or core.sha256(bundle / "manifest.json") != manifest["bundle_fingerprint"]:
        raise ValueError("Saved compact input bundle differs")
    if manifest.get("r2_definition") == R2_DEFINITION:
        return verify_compact_result(directory)

def write_suite_index(output_root, groups=GROUPS):
    root = Path(output_root)
    summaries, reports = [], []
    for group in groups:
        directory = root / group
        manifest = verify_compact_result(directory)
        setting = saved_config(manifest)
        summary = pd.read_csv(directory / "summary.csv")
        summary.insert(0, "group", group)
        summary.insert(1, "context_length", setting.context_length)
        summary.insert(2, "prediction_length", setting.prediction_length)
        summary.insert(3, "stride", setting.stride)
        summary["r2_scope"] = manifest.get("r2_scope", "all_windows")
        summaries.append(summary)
        reports.append(f"- [{group}: {setting.context_length}→{setting.prediction_length}]({group}/report.md)")
    pd.concat(summaries, ignore_index=True).to_csv(root / "summary.csv", index=False)
    (root / "README.md").write_text("# 모든 공통 SOH 데이터의 zero-shot\n\n" + "\n".join(reports) +
        "\n\n그룹별 모델 점수는 summary.csv에 있다. 배터리 기록에 같은 가중치를 주며, 그룹 간 점수를 하나로 합치지 않는다. "
        "R²는 한 창의 실제 미래 평균을 기준으로 계산하고 창별 점수를 단순평균한 뒤 seed·배터리를 동일 가중 평균한다. "
        "r2_scope=representative_windows인 T5 MIT batch1·3은 대표 창만 평가했으며 나머지 지표는 기존 전체 값이다. "
        "C2는 전체 창 평가를 유지한다. CALCE는 결과 경로별 context/horizon을 확인한다.\n", encoding="utf-8")


def refresh_all_compact_r2(output_root):
    """Upgrade all seven completed groups and their combined index without inference."""
    root = Path(output_root).resolve()
    for group in GROUPS:
        manifest = verify_compact_result(root / group, require_r2=False)
        if manifest.get("r2_definition") != R2_DEFINITION and len(saved_config(manifest).inference_seeds) != 1:
            raise ValueError(f"Cannot recover {group} R2 from multiple seeds without raw forecasts")
    for group in GROUPS:
        refresh_compact_r2(root / group)
    write_suite_index(root)
    return root


def battery_zeroshot_minimal(bundle, config=None, output_root=core.ROOT / "results/t5-battery",
                            force=False, pipeline=None):
    bundle, output_root = Path(bundle).resolve(), Path(output_root).resolve()
    config = config or core.load_zeroshot_config()
    if output_root == bundle or bundle in output_root.parents or output_root in bundle.parents:
        raise ValueError("Forecast outputs must be separate from the input data bundle")
    records, coverage, _, _ = core.prepare_battery_windows(bundle, config)
    if not records:
        raise ValueError("No eligible forecast windows; adjust context/horizon for this bundle")
    directory = output_root / (bundle.name + ("_smoke" if config.max_windows_per_battery else ""))
    manifest_path = directory / "run.json"
    fingerprint = core.sha256(bundle / "manifest.json")
    previous = None
    if manifest_path.is_file() and not force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (previous["status"] == "complete" and saved_config(previous) == config and previous["bundle"] == str(bundle)
                and previous.get("bundle_fingerprint") == fingerprint):
            refresh_compact_r2(directory, bundle)
            return directory
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "bundle": str(bundle), "model": config.model_id,
                "bundle_fingerprint": fingerprint,
                "config": asdict(config), "device": resolve_device(config.device),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "r2_definition": R2_DEFINITION,
                "scope": "smoke" if config.max_windows_per_battery else ("full" if config.stride == 1 else "strided"),
                "retention": "Summary, battery metrics, forecast figures and settings only; no raw predictions"}
    core.write_json(manifest_path, manifest)
    owned_pipeline = pipeline is None
    rows, plots = [], []
    completed_ids = set()
    if (previous and saved_config(previous) == config and previous.get("bundle_fingerprint") == fingerprint
            and previous["status"] in {"running", "failed"} and (directory / "battery_metrics.csv").is_file()):
        partial = pd.read_csv(directory / "battery_metrics.csv", keep_default_na=False, na_values=[""])
        for record in records:
            saved = directory / f".forecast_{record['record_id']}.npz"
            scores = partial[(partial.dataset == record["dataset"]) & (partial.battery_id == record["battery_id"])]
            if saved.is_file() and set(scores.model) == {"Chronos", "LastValue", "Linear"}:
                if "r2" not in scores or previous.get("r2_definition") != R2_DEFINITION:
                    continue
                with np.load(saved, allow_pickle=False) as forecast:
                    plots.append({**record, "forecast": {key: forecast[key].copy() for key in forecast.files}})
                rows.extend(scores.to_dict("records"))
                completed_ids.add(record["record_id"])
        if completed_ids:
            print(f"RESUME {bundle.name} batteries={len(completed_ids)}/{len(records)}", flush=True)
    try:
        if owned_pipeline:
            pipeline = fresh_pipeline(config, config.inference_seeds[0])
        for record in records:
            if record["record_id"] in completed_ids:
                continue
            windows = record["windows"]
            tags = {key: record[key] for key in core.IDENTITY[:3]}
            tags.update(n_windows=len(windows["target"]), status="evaluated")
            for model, prediction in core.baseline_predictions(windows).items():
                rows.append({**tags, "model": model, "inference_seed": -1,
                    **compact_point_scores(windows["target"], prediction),
                    **{key: float("nan") for key in ["wql", "picp80", "width80"]}})
            for seed_index, seed in enumerate(config.inference_seeds):
                print(f"PREDICT {bundle.name}/{record['battery_id']} seed={seed} windows={tags['n_windows']}", flush=True)
                samples = predict_samples(pipeline, windows["context"], config, seed)
                scores, summary = compact_scores(windows["target"], samples)
                rows.append({**tags, "model": "Chronos", "inference_seed": seed, **scores})
                if seed_index == 0:
                    indices = np.array([0, len(windows["target"]) // 2, len(windows["target"]) - 1])
                    plots.append({**record, "forecast": {
                        **{key: value[indices].copy() for key, value in windows.items()},
                        **{key: value[indices].copy() for key, value in summary.items()}}})
                del samples
            np.savez_compressed(directory / f".forecast_{record['record_id']}.npz", **plots[-1]["forecast"])
            aggregate_batteries(rows).to_csv(directory / "battery_metrics.csv", index=False)
            completed = len(plots)
            manifest["batteries_completed"] = completed
            core.write_json(manifest_path, manifest)
            print(f"DONE {bundle.name} battery={completed}/{len(records)} rmse={rows[-1]['rmse']:.6f}", flush=True)
        battery = aggregate_batteries(rows)
        summary = battery.groupby("model", as_index=False)[SCORES].mean()
        for item in coverage[coverage.evaluated_windows == 0].to_dict("records"):
            battery.loc[len(battery)] = {key: (item[key] if key in item else
                ("Chronos" if key == "model" else (0 if key == "n_windows" else float("nan")))) for key in battery.columns}
        battery.to_csv(directory / "battery_metrics.csv", index=False)
        summary.to_csv(directory / "summary.csv", index=False)
        from src.battery_zeroshot_plots import forecast_figures
        plots.sort(key=lambda record: record["record_id"])
        figures = forecast_figures(directory, plots, config, show_window_scores=True)
        write_compact_report(directory, config, coverage, summary, figures)
        manifest.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
                        batteries_evaluated=len(records), batteries_skipped=int((coverage.evaluated_windows == 0).sum()),
                        evaluated_windows=int(coverage.evaluated_windows.sum()), figures=figures)
        core.write_json(manifest_path, manifest)
        verify_compact_result(directory)
        for record in records:
            checkpoint = directory / f".forecast_{record['record_id']}.npz"
            if checkpoint.parent == directory and checkpoint.is_file():
                checkpoint.unlink()
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        core.write_json(manifest_path, manifest)
        raise
    finally:
        if owned_pipeline:
            del pipeline
            clear_device_cache()
    return directory


def run_all_battery_zeroshot(config=None, output_root=core.ROOT / "results/t5-battery", force=False,
                           calce_context=64, calce_horizon=16):
    """Evaluate every cleaned bundle, sharing one model and retaining compact results."""
    config = config or core.load_zeroshot_config()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    groups = GROUPS
    pipeline = fresh_pipeline(config, config.inference_seeds[0])
    completed_groups = []
    try:
        for group in groups:
            setting = replace(config, context_length=calce_context, prediction_length=calce_horizon,
                              stride=1) if group == "calce_cs2_1c" else config
            directory = battery_zeroshot_minimal(core.ROOT / "data/processed/clean" / group, setting,
                                                output_root, force, pipeline)
            completed_groups.append(group)
            write_suite_index(output_root, completed_groups)
        retire_replaced_smoke_results(output_root)
    finally:
        del pipeline
        clear_device_cache()
    return output_root


def retire_replaced_smoke_results(output_root):
    """Remove the two superseded XJTU checks only after the suite is complete."""
    root = Path(output_root).resolve()
    for group in ["nasa_aging", "calce_cs2_1c", "xjtu_2c", "xjtu_3c", "mit_batch1", "mit_batch2", "mit_batch3"]:
        verify_compact_result(root / group)
    for name in ["20261006T030616Z_batteryzeroshot_bf1d52", "20261006T030722Z_batteryzeroshot_fecc95"]:
        directory = (root / name).resolve()
        if directory.is_dir():
            if not directory.is_relative_to(root) or directory.parent != root:
                raise ValueError("Legacy result cleanup escaped the result directory")
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("kind") != "battery_zeroshot" or manifest.get("scope") != "smoke":
                raise ValueError("Refusing to retire an unexpected experiment")
            shutil.rmtree(directory)
            print(f"REMOVED superseded smoke result {name}", flush=True)
