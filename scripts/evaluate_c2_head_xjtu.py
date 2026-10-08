"""Evaluate locked SOH-only heads on every XJTU cell outside training/validation."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, sha256, write_json
from src.battery_model_zeroshot import load_model, predict_quantiles, phase_labels, PHASES
from src.battery_fewshot_tuning import check_battery_split
from scripts.run_c2_partial_tuning import restore, summarize
from scripts.report_r2_distribution import distribution

OUT = ROOT / "results/c2-battery/head-only-xjtu-all-test"
SOURCE = ROOT / "results/c2-battery/fewshot-partial-tuning"
CONDITIONS = ("SOH/zero-shot", "SOH/head")


def figures(group, records, data, predictions, directory):
    files = []
    for start in range(0, len(records), 4):
        page = records[start:start + 4]
        fig, axes = plt.subplots(len(page), 3, figsize=(15, 3.1 * len(page) + .6), squeeze=False)
        for row, record in enumerate(page):
            b, w = record["battery_id"], record["windows"]
            cycles = data[(data.battery_id == b) & data.is_observed].cycle.to_numpy()
            phases = phase_labels(w["forecast_start"], cycles)
            centers = cycles.min() + (np.arange(3) + .5) * (cycles.max() - cycles.min() + 1) / 3
            cohort = "previous test" if int(b.rsplit("-", 1)[1]) <= 3 else "additional test"
            for col, (phase, center) in enumerate(zip(PHASES, centers)):
                ax = axes[row, col]
                candidates = np.flatnonzero(phases == phase)
                if not len(candidates):
                    ax.text(.5, .5, f"Cell {b.rsplit('-', 1)[1]} | {phase}\nNo eligible window", ha="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                i = candidates[np.abs(w["forecast_start"][candidates] - center).argmin()]
                x = w["target_cycle"][i]
                ax.plot(w["context_cycle"][i], w["context"][i], color=".65", lw=1.1, label="Past SOH")
                ax.plot(x, w["target"][i], color="black", lw=1.7, label="Actual")
                for condition, color in zip(CONDITIONS, ("#2878b5", "#8e44ad")):
                    q = predictions[b, condition][i]
                    ax.plot(x, q[:, 4], color=color, lw=1.4, label=condition)
                    if condition == "SOH/head":
                        ax.fill_between(x, q[:, 0], q[:, -1], where=q[:, 0] <= q[:, -1], color=color, alpha=.14)
                ax.axvline(x[0], color=".6", ls=":", lw=1)
                ax.set(title=f"Cell {b.rsplit('-', 1)[1]} | {phase} | {x[0]}-{x[-1]} ({cohort})", xlabel="Cycle", ylabel="SOH")
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Chronos-2 {group} | locked head-only vs zero-shot | 64 -> 16, stride 1\n"
                     "Training cells 4,5 and validation cell 6 are excluded from ALL test scores")
        fig.tight_layout(rect=(0, 0, 1, .94))
        name = f"c2-head_forecast_windows_{start // 4 + 1:02d}.png"
        fig.savefig(directory / name, dpi=150)
        plt.close(fig)
        files.append(name)
    return files


def cohort_summary(directory, previous_ids):
    battery = pd.read_csv(directory / "battery_metrics.csv")
    windows = pd.read_csv(directory / "window_metrics.csv.gz")
    rows = []
    for cohort in ("previous_test", "additional_test", "all_test"):
        b = battery if cohort == "all_test" else battery[battery.battery_id.isin(previous_ids) == (cohort == "previous_test")]
        w = windows[windows.battery_id.isin(b.battery_id.unique())]
        for (condition, phase), table in b.groupby(["condition", "phase"], sort=False):
            selected = w[w.condition == condition]
            if phase != "all":
                selected = selected[selected.phase == phase]
            stats = distribution(selected)
            np.testing.assert_allclose(stats["r2_mean"], table.r2.mean(), atol=1e-10)
            scores = {k: float(table[k].mean(skipna=False)) for k in
                      ("rmse", "wql", "picp80", "width80", "invalid_interval_fraction", "quantile_crossing_fraction")}
            rows.append(dict(dataset=directory.name, cohort=cohort, condition=condition, phase=phase,
                             n_batteries=len(table), n_windows=int(table.n_windows.sum()), **scores, **stats))
    result = pd.DataFrame(rows)
    result.to_csv(directory / "cohort_summary.csv", index=False)
    return result


def run_group(group, pipeline):
    directory = OUT / group
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "run.json"
    if state_path.exists():
        raise ValueError(f"Output already exists: {state_path}")
    source = json.loads((SOURCE / group / "run.json").read_text(encoding="utf-8"))
    selection = source["selected"]["SOH/head"]
    assert source["status"] == "complete" and source["selection_locked_at"]
    if sha256(Path(selection["selected_weights"])) != selection["selected_weights_sha256"]:
        raise ValueError("Locked head weights changed")
    original = json.loads((ROOT / f"results/c2-battery/full/{group}/c2-battery_run.json").read_text(encoding="utf-8"))
    if any(original["config"][key] != source[key] for key in ("model_id", "revision")):
        raise ValueError("Source head and foundation checkpoint differ")
    bundle = Path(original["bundle"])
    if sha256(bundle / "manifest.json") != original["bundle_fingerprint"]:
        raise ValueError("Dataset changed since the baseline")
    budget = pd.read_csv(SOURCE / group / "label_budget.csv")
    train = budget[budget.split == "train"].battery_id.unique().tolist()
    val = budget[budget.split == "validation"].battery_id.unique().tolist()
    all_ids = pd.read_csv(bundle / "batteries.csv").battery_id.tolist()
    test = [b for b in all_ids if b not in train + val]
    check_battery_split(train, val, test)
    config = replace(ZeroShotConfig(**original["config"]), battery_ids=tuple(test), max_windows_per_battery=None)
    assert config.context_length == 64 and config.prediction_length == 16 and config.stride == 1
    records, coverage, data, _ = prepare_battery_windows(bundle, config)
    coverage["cohort"] = np.where(coverage.battery_id.isin(source["test_battery_ids"]), "previous_test", "additional_test")
    coverage.to_csv(directory / "coverage.csv", index=False)
    names = selection["trainable_names"]
    if not names or any(not n.startswith("output_patch_embedding.") for n in names):
        raise ValueError("Checkpoint contains parameters outside the head")
    initial = {n: p.detach().cpu().clone() for n, p in pipeline.model.named_parameters() if n in names}
    weights = torch.load(selection["selected_weights"], map_location="cpu", weights_only=True)
    if set(weights) != set(names):
        raise ValueError("Head parameter keys differ")
    state = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
                 device=str(pipeline.model.device), training_performed=False, model_id=config.model_id,
                 revision=config.revision, context=64, horizon=16, stride=1, cross_learning=False,
                 source_selection=str(SOURCE / group / "run.json"), head_sha256=selection["selected_weights_sha256"],
                 locked_learning_rate=selection["learning_rate"], locked_step=selection["step"],
                 train_ids_excluded=train, validation_ids_excluded=val,
                 previous_test_ids=source["test_battery_ids"], all_test_ids=test,
                 additional_test_ids=[b for b in test if b not in source["test_battery_ids"]])
    write_json(state_path, state)
    try:
        predictions = {}
        with np.load(SOURCE / group / "predictions.npz") as cached:
            for record in records:
                b, w = record["battery_id"], record["windows"]
                if b not in source["test_battery_ids"]:
                    continue
                np.testing.assert_array_equal(w["target"], cached[f"{b}_target"])
                np.testing.assert_array_equal(w["forecast_start"], cached[f"{b}_forecast_start"])
                for condition in CONDITIONS:
                    ci = source["prediction_condition_order"].index(condition)
                    predictions[b, condition] = cached[f"{b}_condition{ci}_quantiles"].copy()
        for condition in CONDITIONS:
            restore(pipeline.model, weights if condition == "SOH/head" else initial, names)
            pipeline.model.eval()
            for record in records:
                b, w = record["battery_id"], record["windows"]
                if (b, condition) in predictions:
                    continue
                print(f"EVAL {group} {condition} {b}: {len(w['target'])} windows", flush=True)
                predictions[b, condition] = predict_quantiles(pipeline, w["context"], config)[1]
        summary, conditions = summarize(group, records, data, predictions, directory)
        cohorts = cohort_summary(directory, source["test_battery_ids"])
        previous = pd.read_csv(SOURCE / group / "summary.csv")
        for c in CONDITIONS:
            a = cohorts[(cohorts.cohort == "previous_test") & (cohorts.phase == "all") & (cohorts.condition == c)].iloc[0]
            z = previous[(previous.phase == "all") & (previous.condition == c)].iloc[0]
            for metric in ("rmse", "wql", "picp80", "r2_mean", "r2_median", "r2_negative_fraction"):
                np.testing.assert_allclose(a[metric], z[metric], atol=1e-12)
        state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
                     evaluated_batteries=len(records), evaluated_windows=int(coverage.evaluated_windows.sum()),
                     skipped_ids=coverage[coverage.evaluated_windows == 0].battery_id.tolist(),
                     prediction_condition_order=conditions, existing_results_preserved=True,
                     figures=figures(group, records, data, predictions, directory))
        write_json(state_path, state)
        print(cohorts[cohorts.phase == "all"].to_string(index=False), flush=True)
        return cohorts
    except Exception as error:
        state.update(status="failed", error=repr(error))
        write_json(state_path, state)
        raise
    finally:
        restore(pipeline.model, initial, names)


def main():
    config = ZeroShotConfig(**json.loads((ROOT / "configs/c2-battery.json").read_text(encoding="utf-8")))
    pipeline = load_model(config)
    print(f"INFERENCE ONLY on {pipeline.model.device}", flush=True)
    tables = [run_group(group, pipeline) for group in ("xjtu_2c", "xjtu_3c")]
    all_scores = pd.concat(tables, ignore_index=True)
    all_scores.to_csv(OUT / "cohort_summary.csv", index=False)
    lines = ["# C2 head-only: XJTU 전체 미학습 배터리 테스트", "",
             "기존 검증으로 선택한 SOH-only head를 고정해 추가 추론만 수행했다. 학습·검증 배터리 4·5·6번은 모든 테스트 집계에서 제외했다. "
             "각 조건의 head를 같은 조건의 배터리에 적용했으며, 다른 데이터셋으로의 전이 평가는 아니다.", "",
             "2C: 이전 1·2·3번 + 추가 7·8번. 3C: 이전 1·2·3번 + 추가 7~15번. 모든 유효 64→16 창, stride 1을 평가했다. "
             "이전 세 배터리의 예측은 동일한 정답·시작 cycle을 확인해 재사용했고 원래 점수와 일치함을 확인했다.", "",
             "R²는 창별 정석 계산 후 배터리 내 평균, 배터리 간 동등 가중 평균이다. 중앙값·음수 비율도 같은 배터리 가중치를 쓴다. "
             "상수 창은 R² 미정의로 제외한다. PICP는 q10>q90 역전이 있으면 N/A이며, 예측을 정렬하거나 보정하지 않는다.", "",
             "| 데이터 | 테스트 범위 | 모델 | 배터리 / 창 | RMSE | R² 평균 | R² 중앙값 | 음수 창 비율 | WQL | PICP@80 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in all_scores[all_scores.phase == "all"].itertuples():
        picp = f"{row.picp80:.1%}" if np.isfinite(row.picp80) else "N/A"
        lines.append(f"| {row.dataset} | {row.cohort} | {row.condition} | {row.n_batteries} / {row.n_windows} | "
                     f"{row.rmse:.6f} | {row.r2_mean:.3f} | {row.r2_median:.3f} | {row.r2_negative_fraction:.1%} | {row.wql:.6f} | {picp} |")
    lines += ["", "추가 테스트 배터리는 이번 head 학습·검증·체크포인트 선택에 쓰이지 않았다. "
              "다만 이미 순정 zero-shot 결과를 확인한 데이터이며, foundation 사전학습 데이터와의 중복은 검증하지 않았다. "
              "겹치는 창은 독립 표본이 아니며 단일 seed 결과이다. 초·중·후반 점수와 내부 분위수 역전율은 CSV에 보존했다.", ""]
    for group in ("xjtu_2c", "xjtu_3c"):
        state = json.loads((OUT / group / "run.json").read_text(encoding="utf-8"))
        for name in state["figures"]:
            lines += [f"![{group} 대표 예측창]({group}/{name})", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("ALL_XJTU_HEAD_TEST_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
