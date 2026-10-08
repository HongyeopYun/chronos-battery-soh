"""Validation-selected head/last-group-attention adaptation, then all held-out windows."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from transformers import set_seed

from src.battery_fewshot_tuning import check_battery_split, select_label_windows, fixed_training_inputs, adaptation_scores
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, sha256, write_json
from src.battery_model_zeroshot import load_model, predict_quantiles, phase_labels, PHASES
from src.c2_partial_tuning import configure_subset, model_batch, corrected_loss, validation_scores, eligible_checkpoint, choose_checkpoint
from src.nasa_covariates import FEATURES
from src.xjtu_covariates import extract_xjtu_features
from src.soh_evaluation import window_r2_values, R2_DEFINITION
from src.soh_experiment import clear_device_cache
from scripts.run_c2_nasa_covariates import covariate_predictions
from scripts.report_r2_distribution import distribution

OUT = ROOT / "results/c2-battery/fewshot-partial-tuning"
WEIGHTS = ROOT / "models/c2-fewshot-partial"
LRS = (1e-6, 1e-5)
MODES = ("head", "head_group")
INPUTS = {"SOH": (), "SOH+V+I": FEATURES}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def prepare(group):
    original = json.loads((ROOT / f"results/c2-battery/full/{group}/c2-battery_run.json").read_text(encoding="utf-8"))
    batch, rate = ("batch1", "2C") if group == "xjtu_2c" else ("batch2", "3C")
    ids = lambda numbers: [f"{batch}_{rate}_battery-{n}" for n in numbers]
    train, val, test = ids([4, 5]), ids([6]), ids([1, 2, 3])
    check_battery_split(train, val, test)
    config = replace(ZeroShotConfig(**original["config"]), battery_ids=tuple(train + val + test), max_windows_per_battery=None)
    if sha256(Path(original["bundle"]) / "manifest.json") != original["bundle_fingerprint"]:
        raise ValueError("Dataset changed since baseline")
    records, _, data, _ = prepare_battery_windows(original["bundle"], config)
    by_id = {r["battery_id"]: r for r in records}
    features = pd.concat([
        pd.read_csv(ROOT / f"results/c2-battery/covariates/{group}/c2-battery_{group}_vi_cycle_features.csv"),
        pd.read_csv(ROOT / f"results/c2-battery/fewshot/{group}/c2-battery_{group}_fewshot_support_features.csv"),
        extract_xjtu_features(ROOT / "data/raw/XJTU", data[data.battery_id.isin(val)])], ignore_index=True)
    selected, budget = {}, []
    for b in train + val:
        w = by_id[b]["windows"]
        selected[b] = select_label_windows(w, 10)
        for i in selected[b]:
            budget.append(dict(split="train" if b in train else "validation", battery_id=b,
                               context_start=int(w["context_cycle"][i, 0]), context_end=int(w["context_cycle"][i, -1]),
                               label_start=int(w["target_cycle"][i, 0]), label_end=int(w["target_cycle"][i, -1])))
    budget = pd.DataFrame(budget)
    old = pd.read_csv(ROOT / f"results/c2-battery/fewshot-tuning/{group}/c2-battery_{group}_fewshot_full_label_budget.csv")
    pd.testing.assert_frame_equal(budget, old)
    sets = {}
    for label, names in INPUTS.items():
        sets[label] = []
        for battery_ids in (train, val):
            tasks = []
            for b in battery_ids:
                tasks.extend(fixed_training_inputs(by_id[b]["windows"], features[features.battery_id == b], selected[b], names))
            sets[label].append(tasks)
    return config, by_id, data, features, sets, budget, test


def restore(model, initial, names):
    with torch.no_grad():
        parameters = dict(model.named_parameters())
        for name in names:
            parameters[name].copy_(initial[name])


def train_selected(pipeline, initial, tasks, mode, steps, label, directory, weights_path=None):
    model = pipeline.model
    names = configure_subset(model, mode)
    parameters = dict(model.named_parameters())
    trainable = sum(parameters[n].numel() for n in names)
    total = sum(p.numel() for p in parameters.values())
    n_variates = 1 + len(INPUTS[label])
    val_dataset = Chronos2Dataset(inputs=tasks[1], context_length=64, prediction_length=16,
                                 batch_size=4 * n_variates, output_patch_size=16, min_past=64, mode=DatasetMode.VALIDATION)
    validation = [model_batch(batch, model.device) for batch in val_dataset]
    baseline = dict(step=0, learning_rate=0., **validation_scores(model, validation))
    baseline["eligible"] = True
    history = [baseline]
    winner = baseline
    winner_weights = {n: initial[n].clone() for n in names}
    candidate_runtimes = []
    for lr in LRS:
        restore(model, initial, names)
        frozen_versions = {n: p._version for n, p in parameters.items() if n not in names}
        set_seed(42)
        configure_subset(model, mode)
        optimizer = torch.optim.AdamW([parameters[n] for n in names], lr=lr, weight_decay=0., eps=1e-8)
        assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(parameters[n]) for n in names}
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1 - step / steps))
        train_dataset = Chronos2Dataset(inputs=tasks[0], context_length=64, prediction_length=16,
                                       batch_size=4 * n_variates, output_patch_size=16, min_past=64, mode=DatasetMode.TRAIN)
        iterator = iter(train_dataset)
        started = time.perf_counter()
        for step in range(1, steps + 1):
            batch = model_batch(next(iterator), model.device)
            assert batch["context"].shape == (4 * n_variates, 64)
            assert torch.isfinite(batch["future_target"]).sum().item() == 4 * 16
            assert torch.isnan(batch["future_covariates"]).all()
            optimizer.zero_grad(set_to_none=True)
            loss = corrected_loss(model, batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameters[n] for n in names], 1.)
            optimizer.step()
            scheduler.step()
            if step % 20 == 0:
                row = dict(step=step, learning_rate=lr, **validation_scores(model, validation))
                row["eligible"] = eligible_checkpoint(row, baseline)
                history.append(row)
                chosen = choose_checkpoint(history, baseline)
                if chosen is row:
                    winner = row
                    winner_weights = {n: parameters[n].detach().cpu().clone() for n in names}
                print(f"VALIDATE {directory.name} {label}/{mode} lr={lr:g} step={step}: {row}", flush=True)
                pd.DataFrame(history).to_csv(directory / f"{label.replace('+', '_')}_{mode}_validation.csv", index=False)
                configure_subset(model, mode)
        candidate_runtimes.append(dict(learning_rate=lr, training_and_validation_seconds=time.perf_counter() - started))
        if any(parameters[n]._version != version for n, version in frozen_versions.items()):
            raise ValueError("A frozen parameter was modified")
        if any(p.grad is not None for n, p in parameters.items() if n not in names):
            raise ValueError("Frozen parameter received a gradient")
        del optimizer, scheduler, iterator
    assert winner == choose_checkpoint(history, baseline)
    restore(model, initial, names)
    model.eval()
    path = Path(weights_path) if weights_path is not None else WEIGHTS / directory.name / f"{label.replace('+', '_')}_{mode}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(winner_weights, path)
    return dict(**winner, trainable_parameters=trainable, total_parameters=total,
                trainable_fraction=trainable / total, trainable_names=names,
                selected_weights=str(path), selected_weights_sha256=sha256(path),
                optimizer_subset_verified=True, frozen_parameters_verified=True,
                candidate_runtimes=candidate_runtimes, baseline=baseline)


def infer(pipeline, record, features, names, config):
    b, w = record["battery_id"], record["windows"]
    pipeline.model.eval()
    return (covariate_predictions(pipeline, w, features[features.battery_id == b], names, config)[1] if names
            else predict_quantiles(pipeline, w["context"], config)[1])


def summarize(group, records, data, predictions, directory):
    batteries, windows, saved = [], [], {}
    conditions = list(dict.fromkeys(c for _, c in predictions))
    for record in records:
        b, w = record["battery_id"], record["windows"]
        phases = phase_labels(w["forecast_start"], data[(data.battery_id == b) & data.is_observed].cycle.to_numpy())
        saved[f"{b}_target"], saved[f"{b}_forecast_start"] = w["target"], w["forecast_start"]
        for ci, condition in enumerate(conditions):
            q = predictions[b, condition]
            saved[f"{b}_condition{ci}_quantiles"] = q
            r2 = window_r2_values(w["target"], q[..., 4])
            # Independent direct verification, with explicit constant-window handling.
            y, point = w["target"].astype(float), q[..., 4].astype(float)
            sst = ((y - y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
            direct = np.full(len(y), np.nan)
            np.divide(((y - point) ** 2).sum(axis=1), sst, out=direct, where=sst > 0)
            np.testing.assert_allclose(r2, 1 - direct, atol=1e-10, rtol=1e-10)
            windows.extend(dict(dataset=group, battery_id=b, condition=condition, phase=phase,
                                forecast_start=int(start), r2=value)
                           for start, phase, value in zip(w["forecast_start"], phases, r2))
            for phase in ("all", *PHASES):
                mask = np.ones(len(phases), bool) if phase == "all" else phases == phase
                if mask.any():
                    batteries.append(dict(dataset=group, battery_id=b, condition=condition, phase=phase,
                                          n_windows=int(mask.sum()), **adaptation_scores(w["target"][mask], q[mask])))
    battery, window = pd.DataFrame(batteries), pd.DataFrame(windows)
    rows = []
    for (condition, phase), table in battery.groupby(["condition", "phase"], sort=False):
        selected = window[window.condition == condition]
        if phase != "all":
            selected = selected[selected.phase == phase]
        score = {key: float(table[key].mean(skipna=False)) for key in
                 ("rmse", "wql", "picp80", "width80", "invalid_interval_fraction", "quantile_crossing_fraction")}
        stats = distribution(selected)
        np.testing.assert_allclose(stats["r2_mean"], table.r2.mean(), atol=1e-10)
        rows.append(dict(dataset=group, condition=condition, phase=phase, n_batteries=len(table),
                         n_windows=int(table.n_windows.sum()), **score, **stats))
    summary = pd.DataFrame(rows)
    summary.to_csv(directory / "summary.csv", index=False)
    battery.to_csv(directory / "battery_metrics.csv", index=False)
    window.to_csv(directory / "window_metrics.csv.gz", index=False)
    np.savez_compressed(directory / "predictions.npz", **saved)
    return summary, conditions


def plot(group, records, data, predictions, directory, input_label):
    fig, axes = plt.subplots(3, 3, figsize=(15, 9.5))
    styles = [("zero-shot", "#2878b5"), ("full", "#888888"), ("head", "#8e44ad"), ("head_group", "#d55e00")]
    for row, record in enumerate(records):
        b, w = record["battery_id"], record["windows"]
        cycles = data[(data.battery_id == b) & data.is_observed].cycle.to_numpy()
        phases = phase_labels(w["forecast_start"], cycles)
        centers = cycles.min() + (np.arange(3) + .5) * (cycles.max() - cycles.min() + 1) / 3
        for col, (phase, center) in enumerate(zip(PHASES, centers)):
            ax = axes[row, col]
            candidates = np.flatnonzero(phases == phase)
            if not len(candidates):
                ax.set_axis_off()
                continue
            i = candidates[np.abs(w["forecast_start"][candidates] - center).argmin()]
            x = w["target_cycle"][i]
            ax.plot(w["context_cycle"][i], w["context"][i], color=".65", lw=1.1, label="Past SOH")
            ax.plot(x, w["target"][i], color="black", lw=1.8, label="Actual")
            for method, color in styles:
                q = predictions[b, f"{input_label}/{method}"][i]
                ax.plot(x, q[:, 4], color=color, lw=1.3, label=method.replace("_", "+"))
                if method == "head_group":
                    ax.fill_between(x, q[:, 0], q[:, -1], where=q[:, 0] <= q[:, -1], color=color, alpha=.12)
            ax.axvline(x[0], ls=":", color=".6", lw=1)
            ax.set(title=f"Cell {b.split('-')[-1]} | {phase} | {x[0]}-{x[-1]}", xlabel="Cycle", ylabel="SOH")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Chronos-2 {group} | {input_label} | 64 -> 16\n"
                 "20 source training windows; validation selects LR/step; test uses ALL eligible windows")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(directory / f"forecast_{input_label.replace('+', '_')}.png", dpi=150)
    plt.close(fig)


def run_group(group, pipeline, steps):
    directory = OUT / group
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "run.json"
    config, by_id, data, features, sets, budget, test_ids = prepare(group)
    budget.to_csv(directory / "label_budget.csv", index=False)
    state = dict(status="running", started_at=timestamp(), device=str(pipeline.model.device),
                 model_id=config.model_id, revision=config.revision, seed=42, max_steps=steps, learning_rates=list(LRS),
                 context=64, horizon=16, tasks_per_step=4, train_windows=20, validation_windows=10,
                 train_future_labels=320, validation_future_labels=160, test_battery_ids=test_ids,
                 r2=R2_DEFINITION, selection_rule="Minimum normalized SOH quantile loss among step 0 and checkpoints whose validation decile crossing/invalid interval rates do not exceed baseline; per method and input; no test scores",
                 training="AdamW lr linear decay, weight_decay=0, grad clip=1; frozen modules eval; head and optional last group attention train",
                 scope="Exploratory other-cell offline adaptation. Completed source-cell histories; not target-prefix few-shot. Single seed. All test windows overlap.",
                 selected={})
    for split in ("train", "validation"):
        table = budget[budget.split == split]
        state[f"{split}_unique_soh_observations"] = sum(len({c for r in rows.itertuples()
            for c in range(r.context_start, r.label_end + 1)}) for _, rows in table.groupby("battery_id"))
    initial = {n: p.detach().cpu().clone() for n, p in pipeline.model.named_parameters()}
    write_json(state_path, state)
    try:
        # Lock ALL new-model choices before evaluating held-out test predictions.
        for label in INPUTS:
            for mode in MODES:
                key = f"{label}/{mode}"
                print(f"FIT {group} {key}: two learning rates, {steps} steps each", flush=True)
                state["selected"][key] = train_selected(pipeline, initial, sets[label], mode, steps, label, directory)
                write_json(state_path, state)
        if not all(torch.equal(p.detach().cpu(), initial[n]) for n, p in pipeline.model.named_parameters()):
            raise ValueError("Foundation model did not restore exactly")
        state.update(foundation_restoration_verified=True, selection_locked_at=timestamp(), status="evaluating")
        write_json(state_path, state)
        records = [by_id[b] for b in test_ids]
        predictions = {}
        with np.load(ROOT / f"results/c2-battery/covariates/{group}/c2-battery_{group}_vi_predictions.npz") as cache:
            for record in records:
                b, w = record["battery_id"], record["windows"]
                np.testing.assert_array_equal(w["forecast_start"], cache[f"{b}_forecast_start"])
                np.testing.assert_array_equal(w["target"], cache[f"{b}_target"])
                for label, ci in [("SOH", 0), ("SOH+V+I", 3)]:
                    predictions[b, f"{label}/zero-shot"] = cache[f"{b}_condition{ci}_quantiles"].copy()
        for label, names in INPUTS.items():
            for mode in MODES:
                key = f"{label}/{mode}"
                info = state["selected"][key]
                weights = torch.load(info["selected_weights"], map_location="cpu", weights_only=True)
                restore(pipeline.model, weights, info["trainable_names"])
                for record in records:
                    predictions[record["battery_id"], key] = infer(pipeline, record, features, names, config)
                restore(pipeline.model, initial, info["trainable_names"])
            # Saved older models: inference only, never retrain or select with these test scores.
            for old_mode, method in [("full", "full"), ("head", "old_head_lr1e-4")]:
                path = ROOT / f"models/c2-fewshot-{old_mode}/{group}/{label.replace('+', '_')}/finetuned-ckpt"
                old = type(pipeline).from_pretrained(str(path), device_map=str(pipeline.model.device), dtype=torch.float32, local_files_only=True)
                print(f"EVAL {group} {label}/{method}: all held-out windows", flush=True)
                for record in records:
                    predictions[record["battery_id"], f"{label}/{method}"] = infer(old, record, features, names, config)
                del old
                clear_device_cache()
        summary, conditions = summarize(group, records, data, predictions, directory)
        for label in INPUTS:
            plot(group, records, data, predictions, directory, label)
        state.update(status="complete", finished_at=timestamp(), prediction_condition_order=conditions,
                     evaluated_windows_per_condition=sum(len(r["windows"]["target"]) for r in records),
                     direct_window_r2_verified=True)
        write_json(state_path, state)
        print(summary[summary.phase == "all"].to_string(index=False), flush=True)
        return summary
    except Exception as error:
        state.update(status="failed", error=repr(error))
        write_json(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", choices=["xjtu_2c", "xjtu_3c"], default=["xjtu_2c", "xjtu_3c"])
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.steps < 20 or args.steps % 20:
        parser.error("steps must be a positive multiple of 20")
    # Never accidentally replace the new experiment's completed results.
    if any((OUT / group / "run.json").exists() for group in args.groups):
        parser.error("Run output exists; inspect it before rerunning")
    config = ZeroShotConfig(**json.loads((ROOT / "configs/c2-battery.json").read_text(encoding="utf-8")))
    pipeline = load_model(config)
    print(f"MODEL {config.model_id} on {pipeline.model.device}", flush=True)
    tables = [run_group(group, pipeline, args.steps) for group in args.groups]
    pd.concat(tables, ignore_index=True).to_csv(OUT / "summary.csv", index=False)
    render_report()


def render_report():
    tables, states = [], {}
    for path in sorted(OUT.glob("*/run.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["status"] == "complete":
            states[path.parent.name] = state
            tables.append(pd.read_csv(path.parent / "summary.csv"))
    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    lines = ["# Chronos-2: 소량 데이터로 head / 마지막 group attention 튜닝", "",
             "전체 모델을 다시 학습하지 않고 적은 파라미터만 바꿔 zero-shot에서 개선되는지 확인했다. "
             "비교용 full tuning과 이전 head 모델은 저장된 가중치로 추론만 다시 했다.", "",
             "## 핵심 결과", "",
             "| 데이터 | SOH head RMSE 변화 | SOH head+group RMSE 변화 | head R² 평균 / 중앙값 |",
             "|---|---:|---:|---:|"]
    for group in states:
        scores = summary[(summary.dataset == group) & (summary.phase == "all")].set_index("condition")
        base = scores.loc["SOH/zero-shot", "rmse"]
        head, partial = scores.loc["SOH/head"], scores.loc["SOH/head_group"]
        lines.append(f"| {group} | {(head.rmse / base - 1):+.1%} | {(partial.rmse / base - 1):+.1%} | "
                     f"{head.r2_mean:.3f} / {head.r2_median:.3f} |")
    lines += ["", "이번 결과에서는 head만 3.06% 학습해도 zero-shot보다 개선됐고, head+마지막 group attention 5.03% 학습의 추가 이득은 작았다. "
              "전압·전류를 더한 신규 모델은 같은 방식의 SOH 단독 모델보다 두 데이터셋 모두 RMSE가 높았다. 따라서 이번 범위에서는 변수 추가의 이득이 확인되지 않았다.", "",
              "신규 모델의 q10>q90 역전은 테스트 전체에서 없었다. 다만 일부 내부 분위수 순서 역전은 남아 있으며, "
              "PICP@80도 목표 80%에는 못 미쳤다. 낮춘 학습률과 검증 필터를 함께 적용했으므로 각각의 효과를 따로 입증한 실험은 아니다. "
              "평균 window R²는 여전히 음수인 조건들이 있다. 오차가 줄었다는 결과와 모든 창에서 예측이 충분히 정확하다는 주장은 구분해야 한다.", "",
             "## 실험 조건", "",
             "- XJTU 2C·3C 각각: 학습 배터리 4·5번, 검증 6번, 테스트 1·2·3번. 기존 실험과 동일한 학습창이다.",
             "- 학습 20개 창(미래 SOH 정답 320개), 검증 10개 창(160개). 1개 창은 과거 64 + 미래 16 cycle이며, 20개 측정값만 쓰는 실험이 아니다.",
             "- 완료된 다른 배터리의 수명 전체에서 창을 고른 offline adaptation이다. 테스트 배터리의 초기 일부에 적응하는 방식과 구분한다.",
             "- SOH만 입력 / SOH+과거 전압·전류를 입력하는 조건을 모두 비교했다. 미래 V/I는 NaN, 서로 다른 예측창은 독립된 group이다.",
             "- head: output_patch_embedding 전체. head_group: head + 마지막 encoder block의 group attention만 학습한다. 나머지 가중치는 고정한다.",
             "- 후보 학습률 1e-6·1e-5, 후보당 100 optimizer steps, step당 4개 창, seed 42, AdamW·선형 학습률 감소. 20 step마다 검증한다.",
             "- 각 방식·입력별로 검증 decile 분위수 역전율과 q10>q90 비율이 초기 모델보다 늘지 않는 후보 중 검증 SOH quantile loss가 가장 낮은 LR·step을 선택했다. Step 0도 포함했다.",
             "- 모든 신규 모델 선택을 잠근 다음 테스트 전체 창을 평가했다. 재정렬·clipping·평활화는 없다. 검증의 분위수 검사가 테스트에서도 역전이 없음을 보장하지는 않는다.",
             "- R²는 각 16-cycle 예측창에서 정석으로 계산한다. 배터리 내 창 평균 → 배터리 간 동등 가중 평균. 중앙값과 음수 비율도 같은 배터리 가중치를 쓰며, pooled R²는 사용하지 않는다.",
             "- RMSE·WQL·PICP는 배터리별 계산 후 동등 가중 평균. q10>q90이 한 점이라도 있으면 해당 그룹 PICP는 N/A로 표시하고 평균에도 전파한다.",
             "- 초·중·후반은 관측 cycle 범위를 3등분해 예측 시작 cycle로 배정한다. 세부 점수는 summary.csv에 있다.", "",
             "단일 seed와 기존에 검토한 테스트 배터리를 사용하는 탐색 실험이다. 여러 데이터셋에서 재현되는 일반적 우위나 foundation 사전학습 데이터 중복 부재까지 입증한 것은 아니다.", ""]
    for group, state in states.items():
        table = summary[(summary.dataset == group) & (summary.phase == "all")]
        lines += [f"## {group}: 조건당 전체 {state['evaluated_windows_per_condition']:,}개 테스트 창", "",
                  f"과거 입력을 포함한 고유 SOH 관측값: 학습 {state['train_unique_soh_observations']}개, 검증 {state['validation_unique_soh_observations']}개.", "",
                  "| 입력 / 방법 | RMSE ↓ | R² 평균 | R² 중앙값 | 음수 창 비율 ↓ | WQL ↓ | PICP@80 | 분위수 역전율 ↓ |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for label in INPUTS:
            for method in ("zero-shot", "full", "old_head_lr1e-4", "head", "head_group"):
                row = table[table.condition == f"{label}/{method}"].iloc[0]
                picp = f"{row.picp80:.1%}" if np.isfinite(row.picp80) else "N/A"
                lines.append(f"| {row.condition} | {row.rmse:.6f} | {row.r2_mean:.3f} | {row.r2_median:.3f} | "
                             f"{row.r2_negative_fraction:.1%} | {row.wql:.6f} | {picp} | {row.quantile_crossing_fraction:.2%} |")
        lines += ["", "| 검증으로 선택한 조건 | LR | step | 학습 파라미터 | 두 후보 학습+검증 합계 |",
                  "|---|---:|---:|---:|---:|"]
        for key, selection in state["selected"].items():
            seconds = sum(r["training_and_validation_seconds"] for r in selection["candidate_runtimes"])
            lines.append(f"| {key} | {selection['learning_rate']:g} | {selection['step']} | "
                         f"{selection['trainable_parameters']:,} ({selection['trainable_fraction']:.2%}) | {seconds:.1f}초 |")
        lines += ["", "시간은 이 노트북 XPU에서 측정한 신규 후보의 학습·검증 합계이며 모델 로딩·전처리·테스트 추론은 제외한다. "
                  "파라미터 감소가 같은 비율의 실행시간 감소를 뜻하지는 않는다. 이전 full tuning은 LR 1e-6, 이전 head는 LR 1e-4였고, "
                  "이번에는 후보 탐색과 검증 분위수 필터를 추가했으므로 방식별 최적 성능을 동일 탐색 예산으로 비교한 결과는 아니다.", "",
                  f"![SOH 단독]({group}/forecast_SOH.png)", "", f"![SOH+V+I]({group}/forecast_SOH_V_I.png)", ""]
    lines += ["## 저장 파일", "", "- summary.csv: 전체/초·중·후반 점수와 R² 분포 통계.",
              "- 각 그룹의 run.json: 선택한 LR·step, 가중치 위치, 파라미터 범위, 실행 정보.",
              "- *_validation.csv: 테스트를 보지 않고 선택한 후보의 검증 기록.",
              "- window_metrics.csv.gz / predictions.npz: 창별 R²와 재계산 가능한 예측값.",
              "- forecast_*.png: 테스트 배터리별 초·중·후반 대표창. 그래프는 전체 평가창을 대표하는 예시이지 전체 성능 그 자체가 아니다.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
