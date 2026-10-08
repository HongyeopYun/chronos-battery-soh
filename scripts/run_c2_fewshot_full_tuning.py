"""Small-data C2 full/head tuning: paired inputs, disjoint cells, best validation step."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import TrainerCallback, set_seed

from src.battery_fewshot_tuning import check_battery_split, select_label_windows, fixed_training_inputs, adaptation_scores
from src.c2_fewshot_trainer import TargetOnlyChronos2Trainer, HeadOnlyChronos2Trainer, HEAD_PREFIX
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, write_json, sha256
from src.battery_model_zeroshot import load_model, predict_quantiles, phase_labels, PHASES, SCORES, quantile_scores
from src.nasa_covariates import FEATURES
from src.xjtu_covariates import extract_xjtu_features
from src.soh_experiment import clear_device_cache, resolve_device
from src.soh_evaluation import window_r2_values, R2_DEFINITION
from scripts.run_c2_nasa_covariates import covariate_predictions

CONDITIONS = {"SOH/zero-shot": (), "SOH+V+I/zero-shot": FEATURES,
              "SOH/full-tuning": (), "SOH+V+I/full-tuning": FEATURES}


def conditions_for(tuning):
    return {"SOH/zero-shot": (), "SOH+V+I/zero-shot": FEATURES,
            f"SOH/{tuning}-tuning": (), f"SOH+V+I/{tuning}-tuning": FEATURES}


SOURCES = [
    ("Few-shot multidomain SOH estimation (IEEE TTE)", "https://ieeexplore.ieee.org/document/10699429/",
     "이완 전압으로 현재 SOH를 추정한다. Meta-learning 후 새 배터리의 라벨 있는 6개 cycle로 적응한다. 우리 미래 SOH 예측과는 목표가 다르다."),
    ("Cross-dataset battery life forecasting (Energy and AI)", "https://doi.org/10.1016/j.egyai.2026.100771",
     "TimesFM으로 capacity 궤적·RUL을 예측한다. Source 학습과 target 초반 이력 적응을 구분하고 30%·20%·10% prefix를 비교한다. 우리와 가까운 미래 열화 예측 설정이다."),
    ("Prognostics by generalists (Scientific Reports)", "https://www.nature.com/articles/s41598-026-43545-1",
     "LLaMA 가중치를 고정하고 운전 조건과 과거 SOH 3개를 프롬프트에 넣어 미래 SOH를 예측한다. 논문의 few-shot 단위는 우리 감독 학습창 단위와 다르다."),
    ("Fine-tuning enables SOH estimation (Energy)", "https://doi.org/10.1016/j.energy.2024.134177",
     "TimeGPT 튜닝은 143개 배터리와 방전 capacity 외생변수를 사용한다. 100 step이라는 작은 step 수만으로 few-shot 데이터 실험이라고 볼 수 없고, V/I만 사용하는 우리 미래 예측과 구분해야 한다."),
]


class TrainingLog(TrainerCallback):
    def __init__(self, path, condition, tuning="full"):
        self.path, self.condition, self.state, self.tuning = path, condition, {}, tuning

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        if self.tuning == "full" and trainable != total:
            raise ValueError("Full tuning requires all model parameters to be trainable")
        if self.tuning == "head" and (not names or any(not n.startswith(HEAD_PREFIX) for n in names)):
            raise ValueError("Head tuning must freeze all parameters outside the output head")
        optimizer = kwargs["optimizer"]
        actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        if actual != expected:
            raise ValueError("Optimizer parameters differ from the declared trainable subset")
        self.state.update(trainable_parameters=trainable, total_parameters=total,
                          tuning=self.tuning, trainable_parameter_names=names,
                          optimizer_parameter_subset_verified=True,
                          training_device=str(next(model.parameters()).device), max_steps=state.max_steps)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.state.update(global_step=state.global_step, log_history=state.log_history)
        write_json(self.path, self.state)
        print(f"TRAIN {self.condition} step {state.global_step}: {logs}", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        self.state.update(global_step=state.global_step, best_step=state.best_global_step,
                          best_validation_loss=state.best_metric, best_checkpoint=state.best_model_checkpoint,
                          log_history=state.log_history)
        write_json(self.path, self.state)


def plot_result(directory, prefix, group, records, data, predictions, tuning="full"):
    fig, axes = plt.subplots(len(records), 3, figsize=(15, 9.6), squeeze=False)
    palette = ["#2878b5", "#66a61e", "#8e44ad", "#d55e00"]
    for row, record in enumerate(records):
        b, w = record["battery_id"], record["windows"]
        cycles = data[(data.battery_id == b) & data.is_observed].cycle.to_numpy()
        labels = phase_labels(w["forecast_start"], cycles)
        centers = cycles.min() + (np.arange(3) + .5) * (cycles.max() - cycles.min() + 1) / 3
        for col, (phase, center) in enumerate(zip(PHASES, centers)):
            ax = axes[row, col]
            indices = np.flatnonzero(labels == phase)
            if not len(indices):
                ax.text(.5, .5, f"No {phase} window", ha="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            i = indices[np.abs(w["forecast_start"][indices] - center).argmin()]
            x = w["target_cycle"][i]
            ax.plot(w["context_cycle"][i], w["context"][i], color=".5", lw=1.2, label="Observed context")
            ax.plot(x, w["target"][i], color="black", lw=1.6, label="Actual SOH")
            for condition, color in zip(conditions_for(tuning), palette):
                q = predictions[(b, condition)][i]
                ax.plot(x, q[:, 4], color=color, lw=1.4, label=condition)
                if condition == f"SOH+V+I/{tuning}-tuning":
                    ax.fill_between(x, q[:, 0], q[:, -1], where=q[:, 0] <= q[:, -1], color=color, alpha=.13)
            ax.axvline(x[0], color=".6", ls=":", lw=1)
            ax.set(title=f"{b} | {phase.title()} | {x[0]}-{x[-1]}", xlabel="Cycle", ylabel="SOH")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"Chronos-2 {group} | small-data {tuning} tuning | 64 -> 16\n"
                 "Train: cells 4,5 (20 labeled windows); validation: cell 6; test: cells 1,2,3")
    fig.tight_layout(rect=(0, 0, 1, .94))
    name = f"{prefix}_forecast_windows.png"
    fig.savefig(directory / name, dpi=150)
    plt.close(fig)
    return name


def report(directory, prefix, group, summary, metrics, state, figure):
    if state.get("tuning") == "head":
        return head_report(directory, prefix, group, summary, metrics, state, figure)
    all_scores = summary[summary.phase == "all"].set_index("condition")
    lines = [f"# Chronos-2 {group}: 소량 데이터 full fine-tuning", "",
             "학습 4·5번, 검증 6번, 평가 1·2·3번으로 배터리를 분리했다. "
             "학습 배터리의 측정 완료 이력을 사전에 확보한 offline cross-battery adaptation 실험이다. "
             "평가 배터리 자신의 초반 라벨로 적응하는 target-cell few-shot과는 구분한다. "
             "Foundation 모델의 사전학습 데이터 중복까지 검증했다는 뜻은 아니다.", "",
             "1-shot 단위는 과거 64 cycle + SOH 정답 16 cycle인 하나의 감독 학습창이다. "
             "학습은 셀당 10개, 총 20개 창 / 미래 SOH loss 라벨 320개다. "
             "검증은 별도 셀의 10개 창 / 라벨 160개를 추가로 쓴다. 과거 SOH 입력도 필요하므로 "
             "'SOH 측정값이 총 20개만 필요'하다는 주장이 아니다. "
             "학습/검증 target 구간은 각각 서로 겹치지 않도록 선정하지만 과거 context는 겹칠 수 있다. "
             "창 시작 순서에서 균등 선택하며 정답 값이나 성능으로 고르지 않는다.", "",
             f"학습창에서 실제 필요한 고유 SOH 관측은 {state.get('train_unique_soh_observations', '미집계')}개, "
             f"검증창에서는 {state.get('validation_unique_soh_observations', '미집계')}개다. "
             "창을 전체 source 수명에서 고르기 때문에 source 데이터 취득을 초반에 끝낼 수 있다는 뜻은 아니다.", "",
             f"두 입력 조건 모두 같은 학습창, seed 42, 초기 learning rate 1e-6(선형 감소), 최대 {state['max_steps']} optimizer step, "
             "한 step에 4개 task를 사용한다(SOH batch_size=4, V/I batch_size=12). "
             "80길이 시퀀스, min_past=64로 학습 split을 64에 고정했다. "
             "센서는 과거 64개만 제공하고 미래 16개는 NaN이다. 센서 라벨 loss는 없고, "
             "공식 C2 loss의 masked covariate 행 평균 희석을 SOH 행 수 기준으로 보정했다.", "",
             "검증은 20 step마다 수행하며 가장 낮은 검증 quantile loss의 체크포인트를 사용한다. "
             "검증 라벨은 gradient update에 사용하지 않으며 full-data refit은 하지 않는다. "
             "각 조건은 원래 foundation 가중치에서 시작한다. 조건/step 선택에 테스트 점수를 사용하지 않는다. "
             "이번은 seed 하나의 pilot이며 일반적인 성능 개선을 단정하지 않는다.", "",
             "평가는 이전 cross-learning 비교와 동일한 각 셀 30개, 총 90개 창이다. "
             "zero-shot 캐시와 target/시작 cycle을 대조했다. 64→16, 과거 센서만 제공, "
             "cross_learning=False, 정답 clipping/평활화 없음. "
             "R²는 각 16-cycle 창에서 표준식으로 계산한 후 셀 내 창 산술평균, 셀 간 동일 가중 평균이다. "
             "초중후반은 원래 관측 cycle 범위의 1/3에 예측 시작 cycle을 배정한다. "
             "RMSE/WQL/PICP는 셀 안에서 계산하고 셀 간 동일 가중 평균이다.", "",
             "| 입력·학습 | RMSE ↓ | 평균 window R² ↑ | WQL ↓ | PICP@80 | 같은 입력 zero-shot 대비 RMSE |",
             "|---|---:|---:|---:|---:|---:|"]
    for condition, row in all_scores.iterrows():
        base = all_scores.loc[condition.split('/')[0] + '/zero-shot']
        lines.append(f"| {condition} | {row.rmse:.6f} | {row.r2:.4f} | {row.wql:.6f} | {row.picp80:.1%} | {(row.rmse/base.rmse-1)*100:+.2f}% |")
    soh_change = (all_scores.loc["SOH/full-tuning", "rmse"] / all_scores.loc["SOH/zero-shot", "rmse"] - 1) * 100
    vi_change = (all_scores.loc["SOH+V+I/full-tuning", "rmse"] / all_scores.loc["SOH+V+I/zero-shot", "rmse"] - 1) * 100
    sensor_change = (all_scores.loc["SOH+V+I/full-tuning", "rmse"] / all_scores.loc["SOH/full-tuning", "rmse"] - 1) * 100
    lines += ["", f"같은 입력 zero-shot 대비 RMSE 변화는 SOH {soh_change:+.2f}%, V/I {vi_change:+.2f}%다. "
              f"두 튜닝 모델끼리 비교한 센서 추가 효과는 RMSE {sensor_change:+.2f}%다. "
              "V/I 모델이 자신의 zero-shot보다 좋아졌다는 사실만으로 센서 관계 학습이 효과적이라고 결론내릴 수 없다. "
              "SOH-only 모델도 학습했을 때의 성능과 비교해야 한다. "
              "PICP@80의 목표는 80%이며 정확도 개선과 불확실성 보정을 함께 확인해야 한다."]
    lines += ["", "## 검증으로 선택한 step", "",
              "| 조건 | best step | 검증 loss | 학습 device |", "|---|---:|---:|---|"]
    for condition, log in state["training"].items():
        lines.append(f"| {condition} | {log['best_step']} | {log['best_validation_loss']:.6f} | {log['training_device']} |")
    lines += ["", "## 배터리별 결과", "",
              "| 배터리 | SOH 0-shot RMSE | VI 0-shot RMSE | SOH FT RMSE | VI FT RMSE |", "|---|---:|---:|---:|---:|"]
    for b, table in metrics[metrics.phase == "all"].groupby("battery_id", sort=False):
        rmse = table.set_index("condition").rmse
        lines.append(f"| {b} | " + " | ".join(f"{rmse[c]:.6f}" for c in CONDITIONS) + " |")
    lines += ["", f"![초중후반 대표 예측창]({figure})", "", "## 배터리 few-shot 문헌과 적용 범위", ""]
    for title, url, description in SOURCES:
        lines += [f"- [{title}]({url}): {description}"]
    lines += ["", "문헌의 현재 SOH 추정 정확도를 우리 미래 16-cycle 예측 정확도와 직접 비교하지 않는다. "
              "few-shot은 LoRA와 동의어가 아니며, shot 단위·라벨 예산·시간 분할·배터리 분할을 먼저 명시해야 한다.", ""]
    (directory / f"{prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def head_report(directory, prefix, group, summary, metrics, state, figure):
    previous = ROOT / f"results/c2-battery/fewshot-tuning/{group}/c2-battery_{group}_fewshot_full_summary.csv"
    full = pd.read_csv(previous)
    current = summary[summary.phase == "all"].set_index("condition")
    # This is a paired comparison, not a comparison against the full-window benchmark.
    for condition in conditions_for("head"):
        if condition.endswith("zero-shot"):
            old = full[(full.condition == condition) & (full.phase == "all")].iloc[0]
            np.testing.assert_allclose(current.loc[condition, SCORES].astype(float), old[SCORES].astype(float), rtol=1e-10)
    comparison = pd.concat([summary[summary.phase == "all"],
                            full[(full.phase == "all") & full.condition.str.endswith("full-tuning")]], ignore_index=True)
    comparison.to_csv(directory / f"{prefix}_comparison.csv", index=False)
    log = next(iter(state["training"].values()))
    fraction = log['trainable_parameters'] / log['total_parameters']
    lines = [f"# Chronos-2 {group}: output head few-shot tuning", "",
             f"출력 head만 {log['trainable_parameters']:,}개 파라미터({fraction:.2%}) 학습했다. "
             "C2는 별도 Transformer decoder가 없는 encoder 구조다. 입력 embedding, encoder, "
             "변수 간 group attention은 고정하고 output_patch_embedding의 residual MLP 전체만 갱신했다. "
             "따라서 기존 표현을 SOH 출력에 맞게 조정하는 실험이며, attention의 변수 관계를 직접 재학습하는 실험은 아니다.", "",
             "- 학습: 배터리 4·5번에서 각 10개, 총 20개 창. 검증: 6번의 10개 창. 평가: 1·2·3번의 각 30개, 총 90개 창.",
             "- 1개 창 = 과거 64 cycle → 미래 SOH 16 cycle. 학습 loss 라벨 320개 + 별도 검증 라벨 160개. "
             f"과거 입력까지 포함한 고유 SOH 관측은 학습 {state['train_unique_soh_observations']}개, 검증 {state['validation_unique_soh_observations']}개다.",
             "- source 배터리의 전체 수명에서 창을 골랐다. 다른 배터리의 완료된 이력을 이용하는 소량 데이터 적응이며, "
             "평가 배터리 초반만으로 적응하는 프로토콜은 아니다.",
             "- 학습·검증·평가 배터리를 분리하고, 미래 전압·전류는 NaN으로 제외했다. SOH와 V/I 조건은 같은 창을 사용한다.",
             f"- learning rate {state['learning_rate']:g}, {state['max_steps']} optimizer steps, step당 4개 창, seed 42. "
             "20 step마다 검증하고 가장 낮은 검증 quantile loss의 가중치를 선택했다. 테스트 점수로 step을 고르지 않는다.",
             "- frozen backbone은 eval mode, head는 train mode로 학습했다. SOH loss의 sensor-row 희석 보정은 기존 full tuning과 같다.",
             "- full tuning과 데이터·step 예산은 같지만 full tuning의 learning rate는 1e-6이다. "
             "이 비교만으로 튜닝 방식의 일반적 우열이나 각 방식의 최적 성능을 단정할 수 없다.",
             "- R²는 각각의 16-cycle 예측창에서 정석으로 계산한 뒤 배터리 내 산술평균 → 배터리 간 동일 가중 평균이다. "
             "초·중·후반도 해당 창의 R²를 평균한다. 실제값을 이어 붙인 pooled R²는 사용하지 않는다.", "",
             "분위수는 정렬·보정하지 않고 원본을 평가했다. q10 > q90인 지점이 하나라도 있으면 "
             "그 배터리/구간의 PICP·구간폭을 미정의(N/A) 처리하고, 배터리 평균에도 N/A를 전파한다. "
             "RMSE·R²·WQL은 원래 예측에서 계산할 수 있다. 그래프에서 뒤집힌 구간은 음영을 생략했다.", "",
             "| 입력·학습 | RMSE ↓ | 평균 window R² ↑ | WQL ↓ | PICP@80 |", "|---|---:|---:|---:|---:|"]
    for row in comparison.itertuples():
        picp = f"{row.picp80:.1%}" if np.isfinite(row.picp80) else "N/A (구간 역전)"
        lines.append(f"| {row.condition} | {row.rmse:.6f} | {row.r2:.4f} | {row.wql:.6f} | {picp} |")
    lines += ["", "| 조건 | q10/q90 구간 역전 비율 | 9개 분위수 중 순서 역전 비율 |", "|---|---:|---:|"]
    for row in summary[summary.phase == "all"].itertuples():
        lines.append(f"| {row.condition} | {row.invalid_interval_fraction:.2%} | {row.quantile_crossing_fraction:.2%} |")
    lines += ["", "| 입력 | head / zero-shot RMSE 변화 | head / full tuning RMSE 변화 |", "|---|---:|---:|"]
    full_scores = full[full.phase == "all"].set_index("condition")
    for mode in ["SOH", "SOH+V+I"]:
        head = current.loc[f"{mode}/head-tuning", "rmse"]
        zero = current.loc[f"{mode}/zero-shot", "rmse"]
        old = full_scores.loc[f"{mode}/full-tuning", "rmse"]
        lines.append(f"| {mode} | {(head / zero - 1):+.2%} | {(head / old - 1):+.2%} |")
    sensor = current.loc["SOH+V+I/head-tuning", "rmse"] / current.loc["SOH/head-tuning", "rmse"] - 1
    lines += ["", f"head 튜닝끼리 비교한 V/I 추가의 RMSE 변화: {sensor:+.2%}. "
              "V/I 모델이 자기 zero-shot보다 좋아졌다는 사실만으로 변수 추가의 효과를 주장하지 않는다.", "",
              "| 조건 | 선택 step | 검증 loss | 변경된 head tensor | 고정 tensor 변경 |", "|---|---:|---:|---:|---:|"]
    for condition, entry in state["training"].items():
        lines.append(f"| {condition} | {entry['best_step']} | {entry['best_validation_loss']:.6f} | "
                     f"{entry['changed_parameter_tensors']} | {entry['changed_frozen_parameter_tensors']} |")
    lines += ["", f"![초·중·후반 예측창]({figure})", "",
              "검증: 저장 예측에서 지표 재계산, full 실험과 학습창·평가창 일치, encoder 원본 가중치 불변, "
              "최종 가중치와 best validation checkpoint 일치. 단일 seed 파일럿 결과다.", ""]
    (directory / f"{prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def verify_group(directory, prefix):
    """Independent saved-array scoring and exact best-checkpoint/final-weight verification."""
    state_path = directory / f"{prefix}_run.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    tuning = state.get("tuning", "full")
    conditions = conditions_for(tuning)
    if state["status"] != "complete":
        raise ValueError("Cannot verify an unfinished run")
    metrics = pd.read_csv(directory / f"{prefix}_battery_metrics.csv")
    summary = pd.read_csv(directory / f"{prefix}_summary.csv")
    windows = pd.read_csv(directory / f"{prefix}_window_metrics.csv")
    budget = pd.read_csv(directory / f"{prefix}_label_budget.csv")
    check_battery_split(state["train_battery_ids"], state["validation_battery_ids"], state["test_battery_ids"])
    for split, battery_ids in [("train", state["train_battery_ids"]), ("validation", state["validation_battery_ids"])]:
        table = budget[budget.split == split]
        assert set(table.battery_id) == set(battery_ids)
        count = 0
        for b, rows in table.groupby("battery_id"):
            assert len(rows) == 10 and (rows.label_end - rows.label_start).eq(15).all()
            assert (rows.context_end + 1).eq(rows.label_start).all()
            assert (rows.context_end - rows.context_start).eq(63).all()
            ordered = rows.sort_values("label_start")
            assert (ordered.label_start.to_numpy()[1:] > ordered.label_end.to_numpy()[:-1]).all()
            count += len(set(c for r in rows.itertuples() for c in range(r.context_start, r.label_end + 1)))
        state[f"{split}_unique_soh_observations"] = count
    with np.load(directory / f"{prefix}_predictions.npz") as cache:
        if tuning == "head":
            previous = ROOT / f"results/c2-battery/fewshot-tuning/{directory.name}/c2-battery_{directory.name}_fewshot_full_predictions.npz"
            with np.load(previous) as full_cache:
                for b in state["test_battery_ids"]:
                    for suffix in ["target", "forecast_start", "condition0_quantiles", "condition1_quantiles"]:
                        np.testing.assert_array_equal(cache[f"{b}_{suffix}"], full_cache[f"{b}_{suffix}"])
        for row in metrics.itertuples():
            ci = list(conditions).index(row.condition)
            starts = cache[f"{row.battery_id}_forecast_start"]
            table = windows[(windows.battery_id == row.battery_id) & (windows.condition == row.condition)].sort_values("forecast_start")
            np.testing.assert_array_equal(table.forecast_start, starts)
            y = cache[f"{row.battery_id}_target"]
            q = cache[f"{row.battery_id}_condition{ci}_quantiles"].astype(np.float64)
            assert np.isfinite(q).all()
            if tuning != "head":
                assert (q[..., 0] <= q[..., -1]).all()
            sse = np.square(y - q[..., 4]).sum(axis=1)
            sst = np.square(y - y.mean(axis=1, keepdims=True)).sum(axis=1)
            values = np.full(len(y), np.nan)
            np.divide(sse, sst, out=values, where=sst > 0)
            values = 1 - values
            np.testing.assert_allclose(values, table.r2.to_numpy(), rtol=1e-10, atol=1e-10)
            mask = np.ones(len(y), dtype=bool) if row.phase == "all" else table.phase.to_numpy() == row.phase
            ys, qs = y[mask], q[mask]
            errors = ys[..., None] - qs
            levels = np.arange(1, 10) / 10
            wql = 2 * np.maximum(levels * errors, (levels - 1) * errors).sum(axis=(0, 1)).mean() / np.abs(ys).sum()
            crossed = qs[..., 0] > qs[..., -1]
            expected = [np.sqrt(np.square(ys - qs[..., 4]).mean()), np.nanmean(values[mask]), wql,
                        np.nan if crossed.any() else ((ys >= qs[..., 0]) & (ys <= qs[..., -1])).mean()]
            np.testing.assert_allclose([row.rmse, row.r2, row.wql, row.picp80], expected, rtol=1e-10, atol=1e-10)
            if tuning == "head":
                np.testing.assert_allclose(row.invalid_interval_fraction, crossed.mean())
                np.testing.assert_allclose(row.quantile_crossing_fraction, (np.diff(qs, axis=-1) < 0).any(axis=-1).mean())
            assert mask.sum() == row.n_windows
    grouped = metrics.groupby(["condition", "phase"], sort=False, as_index=False)[SCORES]
    expected = grouped.agg(lambda column: column.mean(skipna=False)) if tuning == "head" else grouped.mean()
    pd.testing.assert_frame_equal(summary[["condition", "phase", *SCORES]], expected, check_dtype=False, rtol=1e-10, atol=1e-10)
    for log in state["training"].values():
        assert log["global_step"] == state["max_steps"]
        if tuning == "head":
            assert 0 < log["trainable_parameters"] < log["total_parameters"]
            assert all(n.startswith(HEAD_PREFIX) for n in log["trainable_parameter_names"])
            assert log["changed_frozen_parameter_tensors"] == 0
        else:
            assert log["trainable_parameters"] == log["total_parameters"]
        checkpoints = [item for item in log["log_history"] if "eval_loss" in item and item["step"] > 0]
        chosen = min(checkpoints, key=lambda item: item["eval_loss"])
        assert chosen["step"] == log["best_step"]
        np.testing.assert_allclose(chosen["eval_loss"], log["best_validation_loss"])
        best = Path(log["best_checkpoint"]) / "model.safetensors"
        final = Path(log["final_model"]) / "model.safetensors"
        if sha256(best) != sha256(final):
            raise ValueError("Exported final weights differ from selected validation checkpoint")
        log["final_matches_best_checkpoint"] = True
    state.update(lr_scheduler="linear", warmup_steps=0, max_grad_norm=1., train_window_draws=state["max_steps"] * 4,
                 independent_saved_metrics_verified=True, final_best_weights_verified=True)
    write_json(state_path, state)
    group = directory.name
    report(directory, prefix, group, summary, metrics, state, f"{prefix}_forecast_windows.png")
    print(f"VERIFIED {group}: independent RMSE/R2/WQL/PICP; exact best weights; observed SOH budgets="
          f"{state['train_unique_soh_observations']}/{state['validation_unique_soh_observations']}", flush=True)
    return state


def run_group(group, base, steps, tuning="full", learning_rate=None):
    learning_rate = learning_rate if learning_rate is not None else (1e-4 if tuning == "head" else 1e-6)
    result_root = "fewshot-head-tuning" if tuning == "head" else "fewshot-tuning"
    conditions = conditions_for(tuning)
    directory = ROOT / f"results/c2-battery/{result_root}/{group}"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"c2-battery_{group}_fewshot_{tuning}"
    state_path = directory / f"{prefix}_run.json"
    existing = {}
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            if existing.get("max_steps") != steps or existing.get("learning_rate") != learning_rate:
                raise ValueError("Completed output uses different settings; use a separate output directory")
            print(f"SKIP complete {group}", flush=True)
            return pd.read_csv(directory / f"{prefix}_summary.csv").assign(dataset=group)
    original = json.loads((ROOT / f"results/c2-battery/full/{group}/c2-battery_run.json").read_text(encoding="utf-8"))
    batch, rate = ("batch1", "2C") if group == "xjtu_2c" else ("batch2", "3C")
    ids = lambda numbers: [f"{batch}_{rate}_battery-{n}" for n in numbers]
    train_ids, val_ids, test_ids = ids([4, 5]), ids([6]), ids([1, 2, 3])
    check_battery_split(train_ids, val_ids, test_ids)
    config = replace(ZeroShotConfig(**original["config"]), battery_ids=tuple(train_ids + val_ids + test_ids), max_windows_per_battery=None)
    records, _, data, manifest = prepare_battery_windows(original["bundle"], config)
    if sha256(Path(original["bundle"]) / "manifest.json") != original["bundle_fingerprint"]:
        raise ValueError("Cleaned dataset changed since zero-shot evaluation")
    record_map = {r["battery_id"]: r for r in records}
    old = ROOT / f"results/c2-battery/covariates/{group}"
    old_prefix = f"c2-battery_{group}_vi"
    support = ROOT / f"results/c2-battery/fewshot/{group}/c2-battery_{group}_fewshot_support_features.csv"
    features = pd.concat([pd.read_csv(old / f"{old_prefix}_cycle_features.csv"), pd.read_csv(support),
                          extract_xjtu_features(ROOT / "data/raw/XJTU", data[data.battery_id.isin(val_ids)])], ignore_index=True)
    selection, budget = {}, []
    for b in train_ids + val_ids:
        w = record_map[b]["windows"]
        selection[b] = select_label_windows(w, 10)
        for index in selection[b]:
            budget.append({"split": "train" if b in train_ids else "validation", "battery_id": b,
                           "context_start": int(w["context_cycle"][index, 0]), "context_end": int(w["context_cycle"][index, -1]),
                           "label_start": int(w["target_cycle"][index, 0]), "label_end": int(w["target_cycle"][index, -1])})
    pd.DataFrame(budget).to_csv(directory / f"{prefix}_label_budget.csv", index=False)
    if tuning == "head":
        previous_budget = pd.read_csv(ROOT / f"results/c2-battery/fewshot-tuning/{group}/c2-battery_{group}_fewshot_full_label_budget.csv")
        pd.testing.assert_frame_equal(pd.DataFrame(budget), previous_budget)
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
             "train_battery_ids": train_ids, "validation_battery_ids": val_ids, "test_battery_ids": test_ids,
             "train_windows": 20, "validation_windows": 10, "train_forecast_labels": 320, "validation_forecast_labels": 160,
             "max_steps": steps, "learning_rate": learning_rate, "seed": 42, "tasks_per_step": 4, "tuning": tuning,
             "context_length": 64, "prediction_length": 16, "min_past": 64,
             "model_id": config.model_id, "revision": config.revision, "device": resolve_device(config.device),
             "r2": R2_DEFINITION, "scope": f"Other-cell small-data {tuning} tuning; sampled paired test windows; not target-cell prefix adaptation",
             "offline_source_data_assumption": "Completed labeled histories of source/validation cells are available before deployment",
             "bundle_fingerprint": original["bundle_fingerprint"], "training": {}}
    if existing.get("training"):
        for key in ["max_steps", "learning_rate", "tuning", "bundle_fingerprint", "train_battery_ids",
                    "validation_battery_ids", "test_battery_ids", "model_id", "revision"]:
            if existing.get(key, "full" if key == "tuning" else None) != state[key]:
                raise ValueError(f"Cannot resume saved tuning with different {key}")
        state["training"] = existing["training"]
        state["resumed_from"] = existing.get("error", existing["status"])
    for split, battery_ids in [("train", train_ids), ("validation", val_ids)]:
        state[f"{split}_unique_soh_observations"] = sum(
            len(np.unique(np.concatenate([record_map[b]["windows"][key][selection[b]].ravel()
                                          for key in ["context_cycle", "target_cycle"]]))) for b in battery_ids)
    state.update(lr_scheduler="linear", warmup_steps=0, max_grad_norm=1.,
                 loss="C2 normalized quantile loss; corrected to SOH rows only",
                 train_window_draws=steps * 4, checkpoint_selection="Minimum held-out validation loss; never test metrics")
    if tuning == "head":
        state["interval_policy"] = "Raw quantiles, no rearrangement. Any q10>q90 makes cell/phase PICP and width undefined; macro propagates NaN."
    write_json(state_path, state)
    try:
        sampled, predictions, saved = [], {}, {}
        # The exact held-out windows from the earlier in-context comparison.
        with np.load(ROOT / f"results/c2-battery/fewshot/{group}/c2-battery_{group}_fewshot_predictions.npz") as cache:
            for b in test_ids:
                full = record_map[b]["windows"]
                starts = cache[f"{b}_forecast_start"]
                indices = np.searchsorted(full["forecast_start"], starts)
                w = {k: v[indices].copy() for k, v in full.items()}
                np.testing.assert_array_equal(w["forecast_start"], starts)
                np.testing.assert_allclose(w["target"], cache[f"{b}_target"], rtol=1e-12, atol=1e-12)
                sampled.append({**record_map[b], "windows": w})
                saved[f"{b}_target"], saved[f"{b}_forecast_start"] = w["target"], starts
                for condition, old_index in [("SOH/zero-shot", 0), ("SOH+V+I/zero-shot", 1)]:
                    predictions[(b, condition)] = cache[f"{b}_condition{old_index}_quantiles"].copy()
        original_versions = tuple(p._version for p in base.model.parameters())
        for mode, names in [("SOH", ()), ("SOH+V+I", FEATURES)]:
            condition = mode + f"/{tuning}-tuning"
            sets = []
            for battery_ids in [train_ids, val_ids]:
                inputs = []
                for b in battery_ids:
                    inputs.extend(fixed_training_inputs(record_map[b]["windows"], features[features.battery_id == b], selection[b], names))
                sets.append(inputs)
            set_seed(42)
            work = ROOT / f"models/c2-fewshot-{tuning}/{group}/{mode.replace('+', '_')}"
            logger = TrainingLog(directory / f"{prefix}_{mode.replace('+', '_')}_training.json", condition, tuning)
            if condition in state["training"]:
                logger.state = state["training"][condition]
                if logger.state["global_step"] != steps:
                    raise ValueError("Cannot reuse an incomplete training checkpoint")
                print(f"REUSE {group} {condition}: saved validation-selected model", flush=True)
                tuned = type(base).from_pretrained(logger.state["final_model"], device_map=str(base.model.device),
                                                  dtype=torch.float32, local_files_only=True)
            else:
                print(f"FIT {group} {condition}: 20 windows, {steps} steps, device={base.model.device}", flush=True)
                trainer_class = HeadOnlyChronos2Trainer if tuning == "head" else TargetOnlyChronos2Trainer
                with patch("chronos.chronos2.trainer.Chronos2Trainer", trainer_class):
                    # fit's 'full' path creates a plain model copy. The custom head trainer
                    # freezes that copy BEFORE optimizer creation; no LoRA is involved.
                    tuned = base.fit(inputs=sets[0], validation_inputs=sets[1], finetune_mode="full", context_length=64,
                                     prediction_length=16, min_past=64, learning_rate=learning_rate, num_steps=steps,
                                     batch_size=4 * (1 + len(names)), output_dir=work, callbacks=[logger],
                                     optim="adamw_torch", eval_steps=20, save_steps=20, logging_steps=10,
                                     seed=42, data_seed=42, bf16=False, fp16=False, tf32=False, disable_tqdm=True,
                                     dataloader_pin_memory=False, greater_is_better=False, eval_on_start=True,
                                     remove_printer_callback=True)
            if tuple(p._version for p in base.model.parameters()) != original_versions:
                raise ValueError("Original foundation weights were modified")
            with torch.no_grad():
                source = dict(base.model.named_parameters())
                changed_names = [n for n, p in tuned.model.named_parameters() if not torch.equal(source[n], p)]
                changed = len(changed_names)
                changed_frozen = sum(not n.startswith(HEAD_PREFIX) for n in changed_names) if tuning == "head" else 0
            if changed_frozen:
                raise ValueError("Head-only tuning changed frozen backbone weights")
            if not changed:
                raise ValueError("Fine-tuning did not update any weights")
            tuned.model.eval()
            state["training"][condition] = {**logger.state, "changed_parameter_tensors": changed,
                                           "changed_frozen_parameter_tensors": changed_frozen,
                                           "changed_parameter_names": changed_names,
                                           "final_model": str(work / "finetuned-ckpt")}
            write_json(state_path, state)
            for record in sampled:
                b, w = record["battery_id"], record["windows"]
                print(f"EVAL {group} {condition} {b}: {len(w['target'])} windows", flush=True)
                _, q = (covariate_predictions(tuned, w, features[features.battery_id == b], names, config) if names
                        else predict_quantiles(tuned, w["context"], config))
                predictions[(b, condition)] = q
                saved[f"{b}_condition{list(conditions).index(condition)}_quantiles"] = q
            np.savez_compressed(directory / f"{prefix}_predictions.npz", **saved)
            del tuned
            clear_device_cache()
        rows, window_rows = [], []
        for record in sampled:
            b, w = record["battery_id"], record["windows"]
            labels = phase_labels(w["forecast_start"], data[(data.battery_id == b) & data.is_observed].cycle.to_numpy())
            for ci, condition in enumerate(conditions):
                q = predictions[(b, condition)]
                saved[f"{b}_condition{ci}_quantiles"] = q
                r2 = window_r2_values(w["target"], q[..., 4])
                for phase in ["all", *PHASES]:
                    mask = np.ones(len(labels), dtype=bool) if phase == "all" else labels == phase
                    if mask.any():
                        score = (adaptation_scores(w["target"][mask], q[mask]) if tuning == "head"
                                 else quantile_scores(w["target"][mask], q[mask, :, 4], q[mask]))
                        np.testing.assert_allclose(score["r2"], np.nanmean(r2[mask]), rtol=1e-10, atol=1e-10)
                        rows.append({"battery_id": b, "condition": condition, "phase": phase, "n_windows": int(mask.sum()), **score})
                for start, phase, score in zip(w["forecast_start"], labels, r2):
                    window_rows.append({"battery_id": b, "condition": condition, "forecast_start": int(start), "phase": phase, "r2": score})
        metrics, window_metrics = pd.DataFrame(rows), pd.DataFrame(window_rows)
        score_columns = SCORES + (["invalid_interval_fraction", "quantile_crossing_fraction"] if tuning == "head" else [])
        grouped = metrics.groupby(["condition", "phase"], sort=False, as_index=False)[score_columns]
        summary = grouped.agg(lambda column: column.mean(skipna=False)) if tuning == "head" else grouped.mean()
        summary = summary.merge(metrics.groupby(["condition", "phase"], sort=False, as_index=False).agg(
            n_batteries=("battery_id", "size"), n_windows=("n_windows", "sum")), on=["condition", "phase"], validate="one_to_one")
        for name, table in [("summary", summary), ("battery_metrics", metrics), ("window_metrics", window_metrics)]:
            table.to_csv(directory / f"{prefix}_{name}.csv", index=False)
        np.savez_compressed(directory / f"{prefix}_predictions.npz", **saved)
        figure = plot_result(directory, prefix, group, sampled, data, predictions, tuning)
        state.update(evaluated_windows_per_condition=sum(len(r['windows']['target']) for r in sampled),
                     original_model_unchanged=True, status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        report(directory, prefix, group, summary, metrics, state, figure)
        write_json(state_path, state)
        verify_group(directory, prefix)
        print(summary[summary.phase == "all"].to_string(index=False), flush=True)
        return summary.assign(dataset=group)
    except Exception as error:
        state.update(status="failed", error=repr(error))
        write_json(state_path, state)
        raise


def main(default_mode="full"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", choices=["xjtu_2c", "xjtu_3c"], default=["xjtu_2c", "xjtu_3c"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tuning", choices=["full", "head"], default=default_mode)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()
    if args.steps < 20 or args.steps % 20:
        parser.error("Use a positive multiple of 20 steps so validation/checkpoint cadence matches")
    if args.learning_rate is not None and not (0 < args.learning_rate < 1):
        parser.error("learning-rate must be between 0 and 1")
    config = ZeroShotConfig(**json.loads((ROOT / "configs/c2-battery.json").read_text(encoding="utf-8")))
    base = load_model(config)
    print(f"MODEL {config.model_id} device={base.model.device}", flush=True)
    tables = [run_group(group, base, args.steps, args.tuning, args.learning_rate) for group in args.groups]
    result_root = "fewshot-head-tuning" if args.tuning == "head" else "fewshot-tuning"
    output = ROOT / f"results/c2-battery/{result_root}"
    pd.concat(tables, ignore_index=True).to_csv(output / f"c2-battery_fewshot_{args.tuning}_summary.csv", index=False)
    del base
    clear_device_cache()


if __name__ == "__main__": main()
