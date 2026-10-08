"""Common 8-train-window / 4-validation-window head adaptation across seven groups."""
import argparse
from dataclasses import asdict
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

from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, write_json, sha256
from src.battery_model_zeroshot import load_model, predict_quantiles, phase_labels, PHASES
from src.battery_fewshot_tuning import select_label_windows, fixed_training_inputs, adaptation_scores
from src.c2_head_protocol import GROUPS, WINDOWS_PER_SOURCE_CELL, make_folds
from src.soh_evaluation import window_r2_values, R2_DEFINITION
from scripts.run_c2_partial_tuning import train_selected, restore
from scripts.report_r2_distribution import distribution

OUT = ROOT / "results/c2-battery/head-only-common8"
WEIGHTS = ROOT / "models/c2-head-common8"
CONDITIONS = ("zero-shot", "head-only")
METRICS = ("rmse", "wql", "picp80", "width80", "invalid_interval_fraction", "quantile_crossing_fraction")


def now():
    return datetime.now(timezone.utc).isoformat()


def config():
    return ZeroShotConfig(**json.loads((ROOT / "configs/c2-battery.json").read_text(encoding="utf-8")))


def load_group(group, cfg):
    return prepare_battery_windows(ROOT / "data/processed/clean" / group, cfg)


def repeated_mit_ids():
    frames = [pd.read_csv(ROOT / "data/processed/clean" / g / "batteries.csv") for g in GROUPS if g.startswith("mit_")]
    physical = pd.concat(frames).physical_battery_id
    return sorted(physical[physical.duplicated(keep=False)].unique().tolist())


def label_budget(record_map, fold):
    rows, selections = [], {}
    for split, key in (("train", "train_ids"), ("validation", "validation_ids")):
        for b in fold[key]:
            w = record_map[b]["windows"]
            selections[b] = select_label_windows(w, WINDOWS_PER_SOURCE_CELL)
            for i in selections[b]:
                rows.append(dict(split=split, battery_id=b, context_start=int(w["context_cycle"][i, 0]),
                                 context_end=int(w["context_cycle"][i, -1]), label_start=int(w["target_cycle"][i, 0]),
                                 label_end=int(w["target_cycle"][i, -1])))
    return pd.DataFrame(rows), selections


def protocol():
    cfg = config()
    assert cfg.context_length == 64 and cfg.prediction_length == 16 and cfg.stride == 1 and cfg.calce_cross_file_windows
    repeated = repeated_mit_ids()
    plan = dict(schema_version=1, config=asdict(cfg), windows_per_source_cell=4,
                train_windows=8, validation_windows=4, train_future_labels=128, validation_future_labels=64,
                seed=42, tasks_per_step=4, max_steps=100, learning_rates=[1e-6, 1e-5],
                validation_steps=[0, 20, 40, 60, 80, 100], trainable="output_patch_embedding only",
                selection="Minimum validation normalized quantile loss among candidates whose decile and q10/q90 crossing rates do not exceed step 0; step 0 allowed",
                repeated_mit_physical_ids_reserved_for_test=repeated,
                source_selection="Natural ID order; no outcome or forecast-score selection. XJTU keeps cells 4,5 train and 6 validation.",
                small_dataset_folds="Four folds: each cell is test once; next cell in sorted cyclic order validates; remaining two train.",
                scope="Other-cell offline supervised adaptation from completed histories, single seed; each group/fold starts from the original foundation checkpoint; no refit or test-time updates",
                r2=R2_DEFINITION, groups={})
    all_source_physical, all_test_physical = set(), set()
    for group in GROUPS:
        records, coverage, _, _ = load_group(group, cfg)
        by_id = {r["battery_id"]: r for r in records}
        folds = make_folds(group, records, repeated)
        for fold in folds:
            budget, _ = label_budget(by_id, fold)
            fold["label_budget"] = budget.to_dict("records")
            fold["unique_soh_observations"] = {split: sum(len({c for r in rows.itertuples() for c in
                range(r.context_start, r.label_end + 1)}) for _, rows in budget[budget.split == split].groupby("battery_id"))
                for split in ("train", "validation")}
            fold["test_windows"] = sum(len(by_id[b]["windows"]["target"]) for b in fold["test_ids"])
            if group.startswith("mit_"):
                all_source_physical.update(by_id[b]["physical_battery_id"] for b in fold["train_ids"] + fold["validation_ids"])
                all_test_physical.update(by_id[b]["physical_battery_id"] for b in fold["test_ids"])
        plan["groups"][group] = dict(bundle_sha256=sha256(ROOT / "data/processed/clean" / group / "manifest.json"),
                                     n_test_records=sum(len(f["test_ids"]) for f in folds),
                                     n_test_windows=sum(f["test_windows"] for f in folds),
                                     skipped_records=coverage[coverage.evaluated_windows == 0].battery_id.tolist(), folds=folds)
    if all_source_physical & all_test_physical:
        raise ValueError("A physical MIT source cell leaks into another batch's test set")
    plan["mit_cross_batch_source_test_disjoint"] = True
    return plan


def prepare_plan():
    plan = json.loads(json.dumps(protocol()))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Prepared protocol changed; use a new result directory")
    else:
        write_json(path, plan)
    return plan


def update_status(state):
    write_json(OUT / "status.json", state)


def train_group(group, group_plan, cfg, pipeline, initial):
    records, _, _, _ = load_group(group, cfg)
    by_id = {r["battery_id"]: r for r in records}
    for fold in group_plan["folds"]:
        directory = OUT / group / fold["fold_id"]
        directory.mkdir(parents=True, exist_ok=True)
        selected_path = directory / "selected.json"
        if selected_path.exists():
            selection = json.loads(selected_path.read_text(encoding="utf-8"))
            if sha256(Path(selection["selected_weights"])) != selection["selected_weights_sha256"]:
                raise ValueError("Saved head checkpoint changed")
            print(f"REUSE {group} {fold['fold_id']} selected checkpoint", flush=True)
            continue
        budget, selected = label_budget(by_id, fold)
        if budget.to_dict("records") != fold["label_budget"]:
            raise ValueError("Training window selection changed")
        budget.to_csv(directory / "label_budget.csv", index=False)
        sets = []
        for key in ("train_ids", "validation_ids"):
            tasks = []
            for b in fold[key]:
                tasks.extend(fixed_training_inputs(by_id[b]["windows"], pd.DataFrame({"cycle": []}), selected[b]))
            sets.append(tasks)
        assert len(sets[0]) == 8 and len(sets[1]) == 4
        print(f"FIT {group} {fold['fold_id']}: 8 train / 4 validation windows", flush=True)
        selection = train_selected(pipeline, initial, sets, "head", 100, "SOH", directory,
                                   weights_path=WEIGHTS / group / f"{fold['fold_id']}.pt")
        selection.update(group=group, fold_id=fold["fold_id"], selected_at=now(),
                         source_ids=fold["train_ids"], validation_ids=fold["validation_ids"], test_ids=fold["test_ids"])
        write_json(selected_path, selection)


def representative_indices(record, data):
    b, w = record["battery_id"], record["windows"]
    cycles = data[(data.battery_id == b) & data.is_observed].cycle.to_numpy()
    phases = phase_labels(w["forecast_start"], cycles)
    result = [0]  # Explicitly 'First eligible', which need not lie in the early third.
    for j, phase in ((1, "middle"), (2, "late")):
        indices = np.flatnonzero(phases == phase)
        center = cycles.min() + (j + .5) * (cycles.max() - cycles.min() + 1) / 3
        result.append(int(indices[np.abs(w["forecast_start"][indices] - center).argmin()]) if len(indices) else -1)
    return result


def evaluate_record(record, data, zero, head, fold_id):
    b, w = record["battery_id"], record["windows"]
    cycles = data[(data.battery_id == b) & data.is_observed].cycle.to_numpy()
    phases = phase_labels(w["forecast_start"], cycles)
    battery_rows, frames = [], []
    for condition, q in zip(CONDITIONS, (zero, head)):
        r2 = window_r2_values(w["target"], q[..., 4])
        # Independent formula verification avoids silently changing the R2 definition.
        y, pred = w["target"].astype(float), q[..., 4].astype(float)
        sst = np.square(y - y.mean(axis=1, keepdims=True)).sum(axis=1)
        direct = np.full(len(y), np.nan)
        np.divide(np.square(y - pred).sum(axis=1), sst, out=direct, where=sst > 0)
        np.testing.assert_allclose(r2, 1 - direct, atol=1e-10, rtol=1e-10)
        frames.append(pd.DataFrame(dict(battery_id=b, physical_battery_id=record["physical_battery_id"],
                                       fold_id=fold_id, condition=condition, phase=phases,
                                       forecast_start=w["forecast_start"], r2=r2)))
        for phase in ("all", *PHASES):
            mask = np.ones(len(y), dtype=bool) if phase == "all" else phases == phase
            if mask.any():
                battery_rows.append(dict(battery_id=b, physical_battery_id=record["physical_battery_id"],
                                         fold_id=fold_id, condition=condition, phase=phase, n_windows=int(mask.sum()),
                                         **adaptation_scores(y[mask], q[mask])))
    return battery_rows, pd.concat(frames, ignore_index=True)


def group_summary(group, battery, windows):
    rows = []
    for (condition, phase), table in battery.groupby(["condition", "phase"], sort=False):
        w = windows[windows.condition == condition]
        if phase != "all":
            w = w[w.phase == phase]
        stats = distribution(w)
        np.testing.assert_allclose(stats["r2_mean"], table.r2.mean(), atol=1e-10)
        rows.append(dict(dataset=group, condition=condition, phase=phase, n_batteries=len(table),
                         n_windows=int(table.n_windows.sum()), **stats,
                         **{key: float(table[key].mean(skipna=False)) for key in METRICS}))
    return pd.DataFrame(rows)


def plot_group(group, representative, directory, adaptation_caption=None):
    ids = list(representative)
    fig, axes = plt.subplots(len(ids), 3, figsize=(15, 3.1 * len(ids) + .5), squeeze=False)
    for row, b in enumerate(ids):
        for col, title in enumerate(("First eligible", "Middle", "Late")):
            ax = axes[row, col]
            item = representative[b][col]
            if item is None:
                ax.text(.5, .5, f"{b} | {title}\nNo eligible window", transform=ax.transAxes, ha="center")
                ax.set_axis_off()
                continue
            x, y = item["target_cycle"], item["target"]
            ax.plot(item["context_cycle"], item["context"], color=".6", label="Past SOH", lw=1.1)
            ax.plot(x, y, color="black", label="Actual", lw=1.6)
            for condition, color in zip(CONDITIONS, ("#2878b5", "#8e44ad")):
                q = item[condition]
                ax.plot(x, q[:, 4], color=color, label=condition, lw=1.4)
                if condition == "head-only":
                    ax.fill_between(x, q[:, 0], q[:, -1], where=q[:, 0] <= q[:, -1], color=color, alpha=.14)
            ax.axvline(x[0], ls=":", color=".6", lw=1)
            ax.set(title=f"{b} | {title} | {x[0]}-{x[-1]}", xlabel="Cycle", ylabel="SOH")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    caption = adaptation_caption or "Head adaptation: 8 source training windows + 4 validation windows; ALL held-out windows scored"
    fig.suptitle(f"Chronos-2 {group} | SOH only | 64 -> 16, stride 1\n" + caption)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(directory / "c2-head_forecast_windows.png", dpi=150)
    plt.close(fig)


def evaluate_group(group, group_plan, cfg, pipeline, initial, state):
    directory = OUT / group
    done_path = directory / "evaluation.json"
    if done_path.exists() and json.loads(done_path.read_text(encoding="utf-8")).get("status") == "complete":
        print(f"REUSE complete evaluation {group}", flush=True)
        return pd.read_csv(directory / "summary.csv")
    records, coverage, data, _ = load_group(group, cfg)
    by_id = {r["battery_id"]: r for r in records}
    assigned = {b: f for f in group_plan["folds"] for b in f["test_ids"]}
    test_ids = [r["battery_id"] for r in records if r["battery_id"] in assigned]
    selected_ids = [test_ids[i] for i in np.linspace(0, len(test_ids) - 1, min(4, len(test_ids)), dtype=int)]
    coverage["role"] = ["test" if b in assigned else "source_or_validation" if b in by_id else "skipped" for b in coverage.battery_id]
    coverage.to_csv(directory / "coverage.csv", index=False)
    rows, frames, representative = [], [], {}
    runtime = directory / ".runtime"
    runtime.mkdir(exist_ok=True)
    names = list(initial)
    weights = {}
    original_root = ROOT / "results/c2-battery" / ("calce64-16" if group == "calce_cs2_1c" else "full") / group
    original = pd.read_csv(original_root / "c2-battery_battery_metrics.csv")
    for b in test_ids:
        fold = assigned[b]
        checkpoint = json.loads((directory / fold["fold_id"] / "selected.json").read_text(encoding="utf-8"))
        record, w = by_id[b], by_id[b]["windows"]
        print(f"EVAL {group} {b}: {len(w['target'])} windows, selected step {checkpoint['step']}", flush=True)
        cache_path = runtime / f"{b}.npz"
        if cache_path.exists():
            with np.load(cache_path) as cached:
                np.testing.assert_array_equal(cached["target"], w["target"])
                np.testing.assert_array_equal(cached["forecast_start"], w["forecast_start"])
                zero, head = cached["zero"].copy(), cached["head"].copy()
        else:
            restore(pipeline.model, initial, names)
            pipeline.model.eval()
            zero = predict_quantiles(pipeline, w["context"], cfg)[1]
            if checkpoint["step"] == 0:
                head = zero.copy()
            else:
                if fold["fold_id"] not in weights:
                    path = Path(checkpoint["selected_weights"])
                    if sha256(path) != checkpoint["selected_weights_sha256"]:
                        raise ValueError("Selected head weights changed")
                    weights[fold["fold_id"]] = torch.load(path, map_location="cpu", weights_only=True)
                restore(pipeline.model, weights[fold["fold_id"]], names)
                head = predict_quantiles(pipeline, w["context"], cfg)[1]
            np.savez_compressed(cache_path, zero=zero, head=head, target=w["target"], forecast_start=w["forecast_start"])
        scores, windows = evaluate_record(record, data, zero, head, fold["fold_id"])
        reference = original[(original.battery_id == b) & (original.model == "c2-battery") & (original.phase == "all")].iloc[0]
        actual = next(s for s in scores if s["condition"] == "zero-shot" and s["phase"] == "all")
        assert reference.n_windows == actual["n_windows"]
        np.testing.assert_allclose([actual[k] for k in ("rmse", "r2", "wql", "picp80")],
                                   reference[["rmse", "r2", "wql", "picp80"]].to_numpy(dtype=float), rtol=1e-6, atol=1e-8)
        rows.extend(scores)
        frames.append(windows)
        if b in selected_ids:
            representative[b] = [None if i < 0 else {**{key: w[key][i] for key in
                ("context", "context_cycle", "target", "target_cycle")}, "zero-shot": zero[i], "head-only": head[i]}
                for i in representative_indices(record, data)]
        state.update(current_group=group, current_battery=b, batteries_completed_in_group=len(frames),
                     batteries_total_in_group=len(test_ids))
        update_status(state)
    battery, windows = pd.DataFrame(rows), pd.concat(frames, ignore_index=True)
    summary = group_summary(group, battery, windows)
    battery.to_csv(directory / "battery_metrics.csv", index=False)
    windows.to_csv(directory / "window_metrics.csv.gz", index=False)
    summary.to_csv(directory / "summary.csv", index=False)
    compact = {}
    for b, items in representative.items():
        for i, item in enumerate(items):
            if item is not None:
                compact.update({f"{b}_{i}_{key}": value for key, value in item.items()})
    np.savez_compressed(directory / "representative_predictions.npz", **compact)
    plot_group(group, representative, directory)
    count = int(battery[(battery.phase == "all") & (battery.condition == "head-only")].n_windows.sum())
    assert count == group_plan["n_test_windows"]
    result = dict(status="complete", finished_at=now(), n_test_records=len(test_ids), n_test_windows=count,
                  zero_shot_matches_existing_results=True, per_window_r2_independently_verified=True,
                  interval_policy="Raw quantiles; q10>q90 invalidates group PICP/width and propagates NaN",
                  representative_batteries=selected_ids, representative_policy="Up to four evenly spaced test record IDs, not performance-selected",
                  predictions_retained="Per-window R2, per-cell aggregate metrics and representative predictions; full temporary quantiles removed")
    write_json(done_path, result)
    # Only remove this runner's verified temporary prediction files, inside its output directory.
    for b in test_ids:
        path = (runtime / f"{b}.npz").resolve()
        if not path.is_relative_to(runtime.resolve()):
            raise ValueError("Temporary prediction path escaped output directory")
        path.unlink(missing_ok=True)
    restore(pipeline.model, initial, names)
    print(summary[summary.phase == "all"].to_string(index=False), flush=True)
    return summary


def report(plan):
    tables, selections = [], []
    for group, group_plan in plan["groups"].items():
        path = OUT / group / "summary.csv"
        if path.exists():
            tables.append(pd.read_csv(path))
        for fold in group_plan["folds"]:
            path = OUT / group / fold["fold_id"] / "selected.json"
            if path.exists():
                s = json.loads(path.read_text(encoding="utf-8"))
                selections.append(dict(dataset=group, fold_id=fold["fold_id"], learning_rate=s["learning_rate"],
                                       step=s["step"], validation_loss=s["validation_loss"],
                                       baseline_loss=s["baseline"]["validation_loss"],
                                       trainable_parameters=s["trainable_parameters"],
                                       search_seconds=sum(r["training_and_validation_seconds"] for r in s["candidate_runtimes"])))
    pd.DataFrame(selections).to_csv(OUT / "selected_checkpoints.csv", index=False)
    if not tables:
        return
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    lines = ["# Chronos-2: 공통 8개 학습창으로 head-only 적응", "",
             "SOH 단독, context 64 → horizon 16, stride 1. 각 실험에서 원래 C2 가중치로 시작해 head 3.06%만 학습한다. "
             "학습 배터리 2개에서 각 4개 창(총 8개/미래 정답 128개), 검증 배터리 1개에서 4개 창(미래 정답 64개)을 사용한다. "
             "이는 고유 SOH 측정값 8개라는 의미가 아니며, source 전체 이력의 과거 입력도 필요하다.", "",
             "## 테스트 결과", "", "| 데이터셋 | 모델 | 테스트 기록/창 | RMSE | R² 평균 | R² 중앙값 | 음수 창 비율 | WQL | PICP@80 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary[summary.phase == "all"].itertuples():
        picp = f"{row.picp80:.1%}" if np.isfinite(row.picp80) else "N/A"
        lines.append(f"| {row.dataset} | {row.condition} | {row.n_batteries}/{row.n_windows:,} | {row.rmse:.6f} | "
                     f"{row.r2_mean:.3f} | {row.r2_median:.3f} | {row.r2_negative_fraction:.1%} | {row.wql:.6f} | {picp} |")
    lines += ["", "## 분할과 학습", "",
              "- NASA·CALCE: 배터리 4개를 test 1개씩 바꾸는 4개 fold. 정렬 ID의 다음 배터리가 검증, 나머지 2개가 학습이다. 각 test 배터리는 해당 fold의 학습·검증에서 제외되고 전체 창을 한 번 평가한다.",
              "- MIT: 자연 ID 순서로, 다른 batch에 이어지는 물리 배터리를 제외한 첫 2개를 학습, 다음 1개를 검증으로 고정한다. 나머지 유효 기록 전체를 테스트한다. 이어지는 동일 물리 배터리의 기록은 모두 test에만 남긴다.",
              "- XJTU: 기존과 동일한 학습 4·5번, 검증 6번, 나머지 전체 test. 기존 20개 학습창 결과를 덮어쓰지 않고 공통 8개 창 결과를 별도로 기록했다.",
              "- 각 source 배터리에서 미래 정답 구간이 겹치지 않는 후보 중 위치에 따라 균등하게 4개를 선택한다. 과거 입력끼리는 겹칠 수 있다. 완료된 다른 배터리 이력을 사용하는 offline adaptation이며 target 초기 prefix 적응이 아니다.",
              "- 두 학습률 1e-6·1e-5, 후보당 100 optimizer steps, step당 4개 창, seed 42, AdamW, 선형 LR 감소, gradient clip 1. 학습에 쓰는 분위수는 기존 C2의 21개이며 context로 계산한 정규화 기준을 사용한다.",
              "- 검증은 20 step마다 수행한다. q0.1~q0.9의 분위수 역전율과 q10>q90 비율이 step 0보다 늘지 않은 후보 중 검증 quantile loss 최소값을 선택한다. Step 0도 포함하며 선택되면 순정을 유지한다. 검증 데이터 refit은 없다.",
              "- 모든 group/fold의 선택을 먼저 고정한 후 테스트한다. 테스트 결과로 LR·step을 바꾸지 않는다. 모델이 서로 다른 fold끼리 가중치를 공유하거나 이어 학습하지 않는다.", "",
              "## 지표와 해석", "",
              "R²는 각 16개 정답의 평균으로 1−SSE/SST를 계산한다. 각 배터리 내 창 평균을 낸 뒤 배터리 기록 간 같은 가중치를 주며, 중앙값과 음수 비율도 동일한 가중치다. "
              "상수 창은 미정의로 제외하고 음수 값은 유지한다. RMSE/WQL/PICP 역시 기록별 계산 후 동등 가중 평균한다. 서로 다른 데이터셋 점수를 하나로 합치지 않는다. "
              "학습 loss와 평가 WQL은 분위수 개수와 정규화 방식이 다르다.", "",
              "매 cycle 실제 SOH 관측이 갱신되는 rolling 예측이다. q50을 점예측, q10~q90을 구간으로 쓰고 WQL은 9개 decile로 평가한다. "
              "예측의 clipping·평활화·분위수 정렬은 없다. 구간 역전 시 해당 그룹 PICP·width는 N/A이며 평균에도 전파한다. "
              "초·중·후반은 관측 cycle 범위의 3등분으로 예측 시작 cycle을 배정한다. 그림 첫 열은 초기 1/3이 아닌 첫 평가 가능 창이며 세부 점수의 early 정의와 구분한다.", "",
              "단일 seed·고정 분할의 탐색 결과이며, 작은 데이터셋 fold들도 source를 공유해 독립 실험이 아니다. MIT 연속 기록 5쌍을 독립 물리 배터리 10개로 해석하지 않는다. "
              "동일 라벨창 예산이라도 데이터 취득 비용·고유 과거 관측 수·열화 범위는 다르다. 테스트 데이터는 과거 zero-shot 실험에서 이미 검토한 데이터이고 foundation 사전학습 데이터 중복은 검증하지 않았다.", "",
              "## 선택한 체크포인트", "", "| 데이터셋 | fold | LR | step | 검증 loss: 초기 → 선택 | 두 후보 학습+검증 시간 |",
              "|---|---|---:|---:|---:|---:|"]
    for s in selections:
        lines.append(f"| {s['dataset']} | {s['fold_id']} | {s['learning_rate']:g} | {s['step']} | "
                     f"{s['baseline_loss']:.4f} → {s['validation_loss']:.4f} | {s['search_seconds']:.1f}초 |")
    for group in plan["groups"]:
        if (OUT / group / "c2-head_forecast_windows.png").exists():
            lines += ["", f"![{group}]({group}/c2-head_forecast_windows.png)"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan = prepare_plan()
    for group, p in plan["groups"].items():
        print(f"PLAN {group}: {len(p['folds'])} fold(s), {p['n_test_records']} test records, {p['n_test_windows']} windows", flush=True)
    if args.prepare_only:
        return
    state = dict(status="training", started_at=now(), groups_completed=[], selected_folds=0,
                 total_folds=sum(len(g["folds"]) for g in plan["groups"].values()), protocol_sha256=sha256(OUT / "protocol.json"))
    update_status(state)
    cfg = config()
    pipeline = load_model(cfg)
    state["device"] = str(pipeline.model.device)
    print(f"MODEL {cfg.model_id} on {pipeline.model.device}", flush=True)
    initial = {n: p.detach().cpu().clone() for n, p in pipeline.model.named_parameters() if n.startswith("output_patch_embedding.")}
    frozen_versions = {n: p._version for n, p in pipeline.model.named_parameters() if n not in initial}
    try:
        for group, p in plan["groups"].items():
            train_group(group, p, cfg, pipeline, initial)
            state.update(current_group=group, selected_folds=state["selected_folds"] + len(p["folds"]))
            update_status(state)
        if any(p._version != frozen_versions[n] for n, p in pipeline.model.named_parameters() if n in frozen_versions):
            raise ValueError("Frozen backbone changed")
        state.update(status="evaluating", all_selections_locked_at=now(), frozen_backbone_verified=True)
        update_status(state)
        for group, p in plan["groups"].items():
            evaluate_group(group, p, cfg, pipeline, initial, state)
            state["groups_completed"].append(group)
            update_status(state)
            report(plan)
        state.update(status="complete", finished_at=now())
        update_status(state)
        report(plan)
        print("ALL_HEAD_COMMON8_COMPLETE", flush=True)
    except Exception as error:
        state.update(status="failed", error=repr(error), failed_at=now())
        update_status(state)
        raise


if __name__ == "__main__":
    main()
