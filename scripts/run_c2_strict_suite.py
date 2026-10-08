"""Chronos-2 zero-shot and head-only on strict_v2; preserve all old results."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
import torch

from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, sha256, write_json
from src.battery_model_zeroshot import load_model, predict_quantiles
from src.battery_fewshot_tuning import select_label_windows, fixed_training_inputs
from src.c2_head_protocol import GROUPS, check_physical_split
from src.soh_evaluation import R2_DEFINITION
from scripts.run_c2_partial_tuning import train_selected, restore
from scripts.run_c2_head_suite import evaluate_record, group_summary, representative_indices, plot_group

OUT = ROOT / "results/c2-battery/strict-v2-zero-head"
WEIGHTS = ROOT / "models/c2-head-strict-v2"
BUNDLES = ROOT / "data/processed/strict_v2"
PRIOR = ROOT / "results/c2-battery/head-only-common8"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def available_labels(windows):
    end, count = -1, 0
    for start, target in zip(windows["forecast_start"], windows["target_cycle"]):
        if start > end:
            count += 1
            end = int(target[-1])
    return count


def prepare():
    cfg = ZeroShotConfig(**read(ROOT / "configs/c2-battery.json"))
    previous = read(PRIOR / "protocol.json")
    plan = dict(schema_version=1, config=asdict(cfg), bundle_root=str(BUNDLES),
                previous_split_protocol_sha256=sha256(PRIOR / "protocol.json"),
                seed=42, head_trainable="output_patch_embedding only", candidate_learning_rates=[1e-6, 1e-5],
                max_steps=100, validation_every=20, step_zero_allowed=True, r2=R2_DEFINITION,
                budget_policy="Up to four non-overlapping target windows per source cell; equal cell budgets within each dataset, reduced to availability only, never forecast scores",
                groups={})
    prepared = {}
    source_physical, test_physical = set(), set()
    for group in GROUPS:
        records, coverage, data, manifest = prepare_battery_windows(BUNDLES / group, cfg)
        by_id = {r["battery_id"]: r for r in records}
        folds = [{k: f[k] for k in ["fold_id", "train_ids", "validation_ids", "test_ids"]}
                 for f in previous["groups"][group]["folds"]]
        source_ids = {b for f in folds for key in ["train_ids", "validation_ids"] for b in f[key]}
        if source_ids - by_id.keys():
            raise ValueError(f"Locked source battery has no strict windows: {group}")
        count = min(4, min(available_labels(by_id[b]["windows"]) for b in source_ids))
        if count < 1:
            raise ValueError("No non-overlapping source label window")
        for fold in folds:
            fold["skipped_test_ids"] = [b for b in fold["test_ids"] if b not in by_id]
            fold["test_ids"] = [b for b in fold["test_ids"] if b in by_id]
            check_physical_split(fold, {b: r["physical_battery_id"] for b, r in by_id.items()})
            budget = []
            for split, key in [("train", "train_ids"), ("validation", "validation_ids")]:
                for b in fold[key]:
                    w = by_id[b]["windows"]
                    for i in select_label_windows(w, count):
                        budget.append(dict(split=split, battery_id=b, index=int(i),
                                           context_start=int(w["context_cycle"][i, 0]), context_end=int(w["context_cycle"][i, -1]),
                                           label_start=int(w["target_cycle"][i, 0]), label_end=int(w["target_cycle"][i, -1])))
            fold["label_budget"] = budget
            if group.startswith("mit_"):
                source_physical.update(by_id[b]["physical_battery_id"] for b in fold["train_ids"] + fold["validation_ids"])
                test_physical.update(by_id[b]["physical_battery_id"] for b in fold["test_ids"])
        tested = [b for f in folds for b in f["test_ids"]]
        if len(tested) != len(set(tested)):
            raise ValueError("Duplicate held-out test assignment")
        directory = OUT / group
        directory.mkdir(parents=True, exist_ok=True)
        coverage["role"] = ["held_out_test" if b in tested else "source_or_validation" if b in by_id else "no_valid_window"
                            for b in coverage.battery_id]
        coverage.to_csv(directory / "coverage.csv", index=False)
        plan["groups"][group] = dict(folds=folds, windows_per_source_cell=count,
                                    train_windows=2 * count, validation_windows=count,
                                    bundle_sha256=sha256(BUNDLES / group / "manifest.json"),
                                    n_test_records=len(tested), n_test_windows=sum(len(by_id[b]["windows"]["target"]) for b in tested),
                                    n_all_windows=sum(len(r["windows"]["target"]) for r in records),
                                    preprocessing_scope="Strict eligible cycles only, including exclusion of quarantined observations")
        prepared[group] = dict(by_id=by_id, records=records, data=data)
    if source_physical & test_physical:
        raise ValueError("MIT physical cell leaks across batches")
    path = OUT / "protocol.json"
    plan = json.loads(json.dumps(plan))
    if path.exists() and read(path) != plan:
        raise ValueError("Protocol changed: use a separate result directory")
    write_json(path, plan)
    return cfg, plan, prepared


def cache_path(group, b, condition):
    path = OUT / group / ".runtime" / f"{b}_{condition}.npz"
    path.parent.mkdir(exist_ok=True)
    return path


def cached_predict(pipeline, cfg, group, record, condition, weights_hash):
    w, b = record["windows"], record["battery_id"]
    path = cache_path(group, b, condition)
    fingerprint = sha256(OUT / "protocol.json")
    if path.exists():
        with np.load(path) as z:
            if str(z["protocol"]) != fingerprint or str(z["weights"]) != weights_hash:
                raise ValueError("Cached model or data changed")
            np.testing.assert_array_equal(z["forecast_start"], w["forecast_start"])
            np.testing.assert_array_equal(z["target"], w["target"])
            return z["quantiles"].copy()
    q = predict_quantiles(pipeline, w["context"], cfg)[1]
    if not np.isfinite(q).all():
        raise ValueError("Nonfinite prediction")
    np.savez_compressed(path, quantiles=q, target=w["target"], forecast_start=w["forecast_start"],
                        protocol=fingerprint, weights=weights_hash)
    return q


def status(state, **kwargs):
    state.update(**kwargs, updated_at=now())
    write_json(OUT / "status.json", state)


def zero_stage(pipeline, cfg, plan, prepared, initial, state):
    restore(pipeline.model, initial, list(initial))
    pipeline.model.eval()
    for group, spec in plan["groups"].items():
        ids = [b for f in spec["folds"] for b in f["test_ids"]]
        for i, b in enumerate(ids):
            status(state, status="zero_shot_inference", group=group, battery=b, completed_in_group=i, total_in_group=len(ids))
            r = prepared[group]["by_id"][b]
            print(f"ZERO {group} {b}: {len(r['windows']['target'])} windows", flush=True)
            cached_predict(pipeline, cfg, group, r, "zero", cfg.revision)
    status(state, status="zero_shot_ready", zero_shot_finished_at=now())


def train_stage(pipeline, cfg, plan, prepared, initial, state, head_mode):
    marker = OUT / "head_mode.json"
    choice = {"head_mode": head_mode}
    if marker.exists() and read(marker) != choice:
        raise ValueError("Head training mode changed")
    write_json(marker, choice)
    frozen = {n: p._version for n, p in pipeline.model.named_parameters() if n not in initial}
    for group, spec in plan["groups"].items():
        for fold in spec["folds"]:
            directory = OUT / group / fold["fold_id"]
            directory.mkdir(exist_ok=True)
            selected_path = directory / "selected.json"
            if selected_path.exists():
                s = read(selected_path)
                if sha256(s["selected_weights"]) != s["selected_weights_sha256"]:
                    raise ValueError("Selected weights changed")
                continue
            status(state, status="head_training", group=group, fold=fold["fold_id"], head_mode=head_mode)
            restore(pipeline.model, initial, list(initial))
            if head_mode == "existing":
                s = read(PRIOR / group / fold["fold_id"] / "selected.json")
                if sha256(s["selected_weights"]) != s["selected_weights_sha256"]:
                    raise ValueError("Historical head weights changed")
                s.update(training_performed=False, trained_on="original clean bundle", head_mode=head_mode)
            else:
                budget = pd.DataFrame(fold["label_budget"])
                budget.to_csv(directory / "label_budget.csv", index=False)
                tasks = []
                for split in ("train", "validation"):
                    values = []
                    for b, selected in budget[budget.split.eq(split)].groupby("battery_id", sort=False):
                        values.extend(fixed_training_inputs(prepared[group]["by_id"][b]["windows"],
                                                          pd.DataFrame({"cycle": []}), selected["index"].to_numpy()))
                    tasks.append(values)
                assert len(tasks[0]) == spec["train_windows"] and len(tasks[1]) == spec["validation_windows"]
                print(f"FIT {group} {fold['fold_id']}: {len(tasks[0])} train / {len(tasks[1])} validation windows", flush=True)
                s = train_selected(pipeline, initial, tasks, "head", 100, "SOH", directory,
                                   weights_path=WEIGHTS / group / f"{fold['fold_id']}.pt")
                s.update(training_performed=True, trained_on="strict_v2", head_mode=head_mode)
            s.update(fold_id=fold["fold_id"], source_ids=fold["train_ids"], validation_ids=fold["validation_ids"],
                     test_ids=fold["test_ids"], locked_at=now())
            write_json(selected_path, s)
    if any(p._version != frozen[n] for n, p in pipeline.model.named_parameters() if n in frozen):
        raise ValueError("Frozen backbone was modified")
    restore(pipeline.model, initial, list(initial))
    status(state, status="head_selection_locked", all_head_selections_locked_at=now(), frozen_backbone_verified=True)


def evaluate(pipeline, cfg, plan, prepared, initial, state):
    summaries = []
    head_mode = read(OUT / "head_mode.json")["head_mode"]
    for group, spec in plan["groups"].items():
        directory = OUT / group
        if (directory / "evaluation.json").exists():
            summaries.append(pd.read_csv(directory / "summary.csv"))
            continue
        assigned = {b: f for f in spec["folds"] for b in f["test_ids"]}
        ids = list(assigned)
        picture_ids = [ids[i] for i in np.linspace(0, len(ids) - 1, min(4, len(ids)), dtype=int)]
        rows, frames, representatives = [], [], {}
        last_weights = None
        for i, b in enumerate(ids):
            f = assigned[b]
            s = read(directory / f["fold_id"] / "selected.json")
            r = prepared[group]["by_id"][b]
            w = r["windows"]
            status(state, status="head_test_inference", group=group, battery=b, completed_in_group=i, total_in_group=len(ids))
            with np.load(cache_path(group, b, "zero")) as z:
                np.testing.assert_array_equal(z["target"], w["target"])
                np.testing.assert_array_equal(z["forecast_start"], w["forecast_start"])
                if str(z["protocol"]) != sha256(OUT / "protocol.json") or str(z["weights"]) != cfg.revision:
                    raise ValueError("Zero-shot cache mismatch")
                zero = z["quantiles"].copy()
            if s["step"] == 0:
                head = zero.copy()
            else:
                if last_weights != s["selected_weights_sha256"]:
                    if sha256(s["selected_weights"]) != s["selected_weights_sha256"]:
                        raise ValueError("Selected head changed")
                    weights = torch.load(s["selected_weights"], map_location="cpu", weights_only=True)
                    if set(weights) != set(initial):
                        raise ValueError("Checkpoint includes weights outside the output head")
                    restore(pipeline.model, weights, list(initial))
                    pipeline.model.eval()
                    last_weights = s["selected_weights_sha256"]
                head = cached_predict(pipeline, cfg, group, r, "head", last_weights)
            print(f"TEST {group} {b}: {len(w['target'])} windows, head step={s['step']}", flush=True)
            a, metrics = evaluate_record(r, prepared[group]["data"], zero, head, f["fold_id"])
            rows.extend(a)
            frames.append(metrics)
            if b in picture_ids:
                representatives[b] = [None if k < 0 else {**{key: w[key][k] for key in ("context", "context_cycle", "target", "target_cycle")},
                                                          "zero-shot": zero[k], "head-only": head[k]}
                                      for k in representative_indices(r, prepared[group]["data"])]
        battery, windows = pd.DataFrame(rows), pd.concat(frames, ignore_index=True)
        for condition in ("zero-shot", "head-only"):
            assert len(windows[windows.condition.eq(condition)]) == spec["n_test_windows"]
        summary = group_summary(group, battery, windows)
        battery.to_csv(directory / "battery_metrics.csv", index=False)
        windows.to_csv(directory / "window_metrics.csv.gz", index=False)
        summary.to_csv(directory / "summary.csv", index=False)
        saved = {f"{b}_{i}_{k}": val for b, items in representatives.items() for i, item in enumerate(items)
                 if item is not None for k, val in item.items()}
        np.savez_compressed(directory / "representative_predictions.npz", **saved)
        caption = (f"Strict data | {spec['train_windows']} training + {spec['validation_windows']} validation windows per fold | all held-out windows"
                   if head_mode == "retrain" else "Strict test data | historical head weights | all held-out windows")
        plot_group(group, representatives, directory, adaptation_caption=caption)
        write_json(directory / "evaluation.json", dict(status="complete", n_test_records=len(ids), n_test_windows=spec["n_test_windows"],
                   head_mode=head_mode, identical_model_test_windows=True, window_r2_independently_verified=True, finished_at=now()))
        summaries.append(summary)
        pd.concat(summaries, ignore_index=True).to_csv(OUT / "summary.csv", index=False)
    restore(pipeline.model, initial, list(initial))
    report(plan, head_mode)
    status(state, status="complete", finished_at=now(), head_mode=head_mode)


def report(plan, head_mode):
    summary = pd.read_csv(OUT / "summary.csv")
    lines = ["# Chronos-2: 정제 데이터 zero-shot / few-shot head-only", "",
             f"Head mode: **{head_mode}**. SOH 단독, 64→16, stride 1. 두 모델을 동일한 미학습 배터리·동일한 유효 창에서 평가했다.", "",
             "| 데이터셋 | 모델 | 배터리 / 창 | RMSE | R² 평균 | R² 중앙값 | 음수 창 | WQL | PICP80 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary[summary.phase.eq("all")].itertuples():
        picp = f"{r.picp80:.1%}" if np.isfinite(r.picp80) else "N/A (구간 역전)"
        lines.append(f"| {r.dataset} | {r.condition} | {r.n_batteries} / {r.n_windows:,} | {r.rmse:.6f} | {r.r2_mean:.3f} | {r.r2_median:.3f} | {r.r2_negative_fraction:.1%} | {r.wql:.6f} | {picp} |")
    lines += ["", "## 실험 조건", "",
              "- 기존 전체 데이터셋 common8 실험의 배터리 분할을 고정했다. XJTU는 4·5번 학습, 6번 검증, 나머지 테스트. NASA/CALCE는 배터리별 4-fold, MIT는 동일 물리 배터리의 배치 간 누수를 차단했다.",
              "- 재학습은 본체 고정, output head 3.06%만 변경한다. 기존 공통 설정 LR 1e-6/1e-5, 각 100 steps, seed 42, 20 steps마다 검증, 순정(step 0)도 후보이며 validation refit은 없다. 새 Optuna 탐색은 없다.",
              "- 기존 common8 비교를 재실행한 것으로, XJTU 전용 Optuna20 또는 LoRA와 다른 실험이다.",
              "- 재학습 시 가능한 비중복 정답 창 수에 따라 NASA/CALCE는 source별 1창(학습 2·검증 1), MIT/XJTU는 source별 4창(학습 8·검증 4)을 쓴다. 겹치는 정답을 복제해 예산을 채우지 않았다. 모든 정확한 cycle 범위는 protocol.json에 있다.",
              "- 다른 배터리의 완료된 이력을 사용하는 offline supervised few-shot이다. 테스트 배터리 정답은 학습과 체크포인트 선택에 사용하지 않았다. 모든 head 선택을 확정한 후 두 모델 점수 비교를 생성했다.",
              "- R²는 각 16개 예측 창 안에서 계산한다. 창별 값을 배터리별 평균 후 배터리 동등 가중 평균한다. 중앙값·음수 비율은 동일 배터리 가중치의 창별 분포다. 음수 유지, 상수 창만 미정의. 초·중·후반은 원래 관측 수명 범위의 3등분이며, 없는 구간은 점수를 만들지 않는다.",
              "- MIT 확인 보류 관측까지 제외한 조건부 평가다. 전체 수명 성능으로 일반화할 수 없다. NASA/CALCE는 정제 후 창이 적고 수명 범위도 좁으므로 결과의 불확실성이 크다.",
              "- 기존 결과보다 점수가 좋아져도 테스트 창이 달라 전처리 자체의 개선 효과로 단정할 수 없다. 기존 결과와 모델 가중치는 보존했다.", "",
              "- MIT 원작자가 noisy channel로 제외한 b3c2, b3c23, b3c32, b3c37, b3c42, b3c43은 학습·검증·테스트에서 모두 제외했다.",
              "- NASA head-only는 B0018 테스트에서 q10>q90 구간 역전이 발생하여 PICP80을 N/A로 표시했다. 검증에서 없던 문제가 테스트에서 발생한 것이며, 분위수 정렬로 숨기거나 테스트 성능을 보고 체크포인트를 바꾸지 않았다.", "",
              "## 선택한 head", "", "| 데이터 | fold | 학습/검증 창 | LR | 선택 step | 검증 loss: 순정→선택 |", "|---|---|---:|---:|---:|---:|"]
    selections = []
    for group, spec in plan["groups"].items():
        for fold in spec["folds"]:
            s = read(OUT / group / fold["fold_id"] / "selected.json")
            selections.append(dict(dataset=group, fold_id=fold["fold_id"], learning_rate=s["learning_rate"], step=s["step"],
                                   validation_loss=s["validation_loss"], baseline_loss=s["baseline"]["validation_loss"],
                                   train_windows=spec["train_windows"], validation_windows=spec["validation_windows"], head_mode=head_mode))
            lines.append(f"| {group} | {fold['fold_id']} | {spec['train_windows']}/{spec['validation_windows']} | {s['learning_rate']:g} | {s['step']} | {s['baseline']['validation_loss']:.4f} → {s['validation_loss']:.4f} |")
    lines += ["", "## 대표 예측 그래프", ""]
    for group in plan["groups"]:
        lines += [f"### {group}", "", f"![{group}]({group}/c2-head_forecast_windows.png)", ""]
    pd.DataFrame(selections).to_csv(OUT / "selected_checkpoints.csv", index=False)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "zero", "finish", "all"], default="all")
    parser.add_argument("--head-mode", choices=["retrain", "existing"], default="retrain")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, plan, prepared = prepare()
    for g, p in plan["groups"].items():
        print(f"PLAN {g}: {p['train_windows']}/{p['validation_windows']} train/val, {p['n_test_records']} test cells, {p['n_test_windows']} test windows", flush=True)
    if args.stage == "prepare":
        return
    state = read(OUT / "status.json") if (OUT / "status.json").exists() else dict(started_at=now())
    pipeline = load_model(cfg)
    initial = {n: p.detach().cpu().clone() for n, p in pipeline.model.named_parameters() if n.startswith("output_patch_embedding.")}
    status(state, status="loaded", device=str(pipeline.model.device), protocol_sha256=sha256(OUT / "protocol.json"))
    print(f"MODEL {pipeline.model.device}", flush=True)
    try:
        if args.stage in ("zero", "all"):
            zero_stage(pipeline, cfg, plan, prepared, initial, state)
        if args.stage in ("finish", "all"):
            train_stage(pipeline, cfg, plan, prepared, initial, state, args.head_mode)
            evaluate(pipeline, cfg, plan, prepared, initial, state)
    except Exception as error:
        status(state, status="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
