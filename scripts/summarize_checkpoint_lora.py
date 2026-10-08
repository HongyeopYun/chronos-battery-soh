"""Summarize the fixed three-LR checkpoint search from saved results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_checkpoint import load_checkpoint_config, verify_checkpoint_run
from src.soh_plots import forecast_grid
from src.soh_evaluation import export_run


def plot_learning_curves(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    batteries = manifest["config"]["battery_ids"]
    seed = manifest["config"]["train_seeds"][0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, battery in zip(axes.flat, batteries):
        selected = next(f for f in manifest["folds"] if f["test_battery"] == battery and f["train_seed"] == seed)
        for path in sorted((Path(directory) / "trials").glob(f"{battery}_train{seed}_*/history.json")):
            trial = json.loads(path.read_text(encoding="utf-8"))
            line, = ax.plot([h["step"] for h in trial["history"]],
                            [h["validation"]["macro"]["crps"] for h in trial["history"]],
                            marker=".", label=f"LR {trial['learning_rate']:g}")
            if trial["learning_rate"] == selected["learning_rate"]:
                ax.scatter(selected["selected_step"], selected["selected_validation"]["macro"]["crps"],
                           marker="*", s=150, color=line.get_color(), edgecolor="black", linewidth=.7,
                           zorder=5, label="Selected checkpoint")
        ax.set(title=battery, xlabel="Optimizer step", ylabel="Macro validation CRPS")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def md_table(frame):
    # Avoid an optional tabulate dependency.
    rows = [[str(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                      "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
                     + ["| " + " | ".join(row) + " |" for row in rows])


def summarize(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    cfg, policy = load_checkpoint_config(ROOT)
    audit = verify_checkpoint_run(directory, ROOT, cfg, policy)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    evaluation = export_run(ROOT, directory)
    data = pd.read_csv(evaluation / "distribution_metrics.csv")
    columns = ["median_rmse", "median_mae", "median_r2", "mean_rmse", "mean_r2", "crps", "coverage80", "width80", "above2_rate"]
    battery = data.groupby(["model", "battery"])[columns].mean()
    macro = battery.groupby(level="model").mean().loc[["ZeroShot", "SelectedLoRA"]]
    macro.to_csv(output / "macro_summary.csv")
    battery.to_csv(output / "battery_summary.csv")
    choices = pd.DataFrame([{
        "battery": f["test_battery"], "train_seed": f["train_seed"], "learning_rate": f["learning_rate"],
        "selected_step": f["selected_step"], "executed_steps_per_candidate": f["executed_steps"],
        "selected_examples_seen": f["selected_examples_seen"],
        "validation_crps": f["selected_validation"]["macro"]["crps"],
    } for f in manifest["folds"]])
    candidates = pd.read_csv(evaluation / "validation_candidates.csv")
    choices = choices.merge(
        candidates.rename(columns={"test_battery": "battery"})[
            ["battery", "train_seed", "learning_rate", "selected_step", "median_r2", "mean_r2"]],
        on=["battery", "train_seed", "learning_rate", "selected_step"], validate="one_to_one")
    choices.to_csv(output / "selected_checkpoints.csv", index=False)
    candidates.to_csv(output / "validation_candidates.csv", index=False)
    plot_learning_curves(directory).savefig(output / "validation_learning_rates.png", dpi=180)
    plt.close("all")
    figure = forecast_grid(directory)
    figure.savefig(output / "best_checkpoint_windows.png", dpi=180)
    figure.savefig(output / "best_checkpoint_windows.pdf")
    plt.close(figure)
    gains = 100 * (1 - macro.loc["SelectedLoRA"] / macro.loc["ZeroShot"])
    main_table = macro.reset_index().round(6)
    comparison = battery["median_rmse"].unstack("model")[["ZeroShot", "SelectedLoRA"]]
    comparison["error_change_pct"] = 100 * (comparison.SelectedLoRA / comparison.ZeroShot - 1)
    r2_comparison = battery["median_r2"].unstack("model")
    comparison["ZeroShot_R2"] = r2_comparison["ZeroShot"]
    comparison["SelectedLoRA_R2"] = r2_comparison["SelectedLoRA"]
    comparison = comparison.reset_index().round(6)
    zero_count = int((choices.selected_step == 0).sum())
    replay = json.loads((directory / "checkpoint_replay.json").read_text(encoding="utf-8"))
    replay_note = ""
    interruption = output / "replay_interruption.json"
    if interruption.exists():
        event = json.loads(interruption.read_text(encoding="utf-8"))
        if Path(event["run"]).resolve() == directory:
            replay_note = ("학습 직후와 resume 경로에서 한 sample 배열의 9,728개 값 중 1개가 달라 "
                           "test 예측 전에 비트 단위 비교가 여러 번 실패했다. 새 프로세스의 별도 검사에서는 "
                           "네 adapter의 모든 저장 tensor가 정확히 로드되는지 확인했으며, 각 모델의 validation을 "
                           "두 번씩 반복해 원본과 정확히 일치함을 확인했다. 실행 경로별 차이가 남아 "
                           "`scripts/resume_checkpoint_verified.py`에서 가중치는 완전 일치를 요구하되, "
                           "sample은 차이를 저장하고 macro CRPS 차이가 기존 선택 기준 `min_delta=1e-5` 미만인지 검사하도록 "
                           "**평가 재현성 검사만 변경했다**. 이 변경은 held-out 예측 생성 전에 결정했다. "
                           "학습 코드·가중치·학습률·선택 step은 바꾸지 않았다. 별도 평가 코드 해시와 재생성한 validation 배열도 보관했다. "
                           "불일치의 원인은 확정하지 않았으며, 모든 실행에서 bitwise 재현성을 보장한다는 뜻은 아니다. "
                           "[중단 기록](replay_interruption.json) · [별도 재검증](replay_diagnostic.json)")
    report = f"""# EOS loss 제외 + best checkpoint + 학습률 3개 비교

실행: `{directory.name}`. 기준 zero-shot: `{Path(manifest['zeroshot_run']).name}`.

## 고정한 프로토콜

- 학습률 `1e-5`, `3e-5`, `1e-4`만 비교했다. Optuna는 사용하지 않고 세 후보를 모두 실행했다.
- 각 outer fold는 평가 배터리를 제외한 세 배터리의 앞쪽 구간으로만 학습했다. 뒤쪽 20% target은 validation으로 유지했다.
- 입력 EOS와 decoder 입력은 유지했다. 마지막 출력 EOS label만 마스킹하고 loss에 `16/17`을 곱해 기존 미래값 토큰 가중치를 유지했다.
- LoRA q/v, rank 8, alpha 16, dropout 0.05, batch 16, gradient accumulation 1. 후보마다 같은 초기화·shuffle·학습 seed를 사용했다.
- 후보당 최대 {policy.max_steps} optimizer steps, {policy.eval_every_steps} steps마다 검증했다. epoch는 기록용이며 학습량을 결정하지 않는다. 조기 종료나 pruning은 하지 않았다.
- Validation은 seed {list(policy.validation_seeds)}, 각각 {policy.validation_samples} samples. 각 seed·배터리의 CRPS를 계산한 뒤 같은 가중치로 평균했다. sample을 합쳐 한 분포로 만들지 않았다.
- Step 0도 후보이며 CRPS가 {policy.min_delta:g}보다 크게 감소할 때만 교체했다. 유사한 점수는 먼저 평가한 step/학습률을 유지했다. RMSE·coverage도 기록했지만 추가 선택 조건으로 사용하지 않았다.
- 네 fold의 학습률과 checkpoint를 모두 고정한 뒤 test 예측을 시작했다. 선택한 adapter를 다시 불러왔고 **full-data refit은 하지 않았다**.
- 최종 학습 seed {list(cfg.train_seeds)}, 추론 seed {list(cfg.inference_seeds)}, window당 {cfg.num_samples} samples. 모델 revision과 실행 환경은 zero-shot과 동일하다.

## 선택 결과

{md_table(choices.round(7))}

Step 0을 선택한 모델은 {zero_count}/{len(choices)}개다. Step 0 결과는 LoRA 학습의 이득으로 해석하지 않는다. 선택 step이 100이면 탐색 상한에서 선택됐다는 뜻이며, 더 긴 학습의 효과는 이번에 확인하지 않았다.

## 평가 결과

배터리와 추론 반복에 같은 가중치를 준 평균이다. `mean_rmse`는 sample mean으로 만든 점예측의 RMSE다. 비율 열은 0~1 단위다.

`median_r2`와 `mean_r2`는 Mean window R²다. 각 창의 실제 미래 평균으로 `1 - SSE/SST`를 계산한 뒤 유효 창별 점수를 단순평균하고 seed·배터리를 동일 가중 평균한다. 음수는 유지하고 상수 target 창은 NaN으로 제외한다. Validation 후보 점수는 선택된 checkpoint의 저장 예측에서 계산하며 다른 step의 원시 예측이 없는 이력은 NaN이다. R²는 모델 선택에 사용하지 않았다.

{md_table(main_table)}

Zero-shot 대비 오차 감소율: 중앙값 RMSE **{gains['median_rmse']:+.2f}%**, 평균 예측 RMSE **{gains['mean_rmse']:+.2f}%**, CRPS **{gains['crps']:+.2f}%**. 양수는 개선, 음수는 악화다.

R² 절대 변화: 중앙값 **{macro.loc['SelectedLoRA', 'median_r2'] - macro.loc['ZeroShot', 'median_r2']:+.6f}**, 평균 예측 **{macro.loc['SelectedLoRA', 'mean_r2'] - macro.loc['ZeroShot', 'mean_r2']:+.6f}**. R²에는 백분율 개선율을 적용하지 않는다.

{md_table(comparison)}

배터리 표의 `error_change_pct`는 오차 증가율이므로 음수가 개선이다.

## 결과를 해석할 때

- 이전 EOS 제외 진단의 중앙값 RMSE 0.014894는 세 학습 배터리 전체로 기존 epoch만큼 학습한 결과다. 이번에는 validation을 남기고 checkpoint와 학습률을 CRPS로 선택하므로 학습 데이터·학습량·선택 방식이 다르다. 이번 점수 차이를 특정 변경 한 가지의 인과 효과로 해석하지 않는다.
- 이 네 배터리는 이미 관찰한 데이터다. 새로운 독립 벤치마크나 통계적 유의성 검증은 아니다. 현재 학습 seed 하나에 대해 세 학습률을 탐색했으며, 추론 seed 반복은 학습 안정성 검증을 대체하지 않는다.
- 중첩 window는 독립적인 배터리 표본이 아니다. Validation은 기존 학습 배터리의 미래 구간으로, 새 배터리의 분포 차이를 완전히 대변하지 않는다.
- Test 결과로 후보·선택 규칙을 다시 고르지 않았다. 원시 samples에 clipping이나 평활화를 적용하지 않았다.

## 검증

- 원시 예측에서 point metric {audit['metric_rows_verified']}행, distribution metric {audit['distribution_rows_verified']}행을 재계산했다. 후보 {audit['trial_count']}개와 adapter·산출물 해시를 확인했다.
- 선택된 {len(replay)}개 adapter를 재로딩했을 때 validation 원시 sample 배열 {sum(r['validation_arrays_identical'] for r in replay)}/{sum(r.get('validation_arrays_checked', 9) for r in replay)}개가 checkpoint 저장 시점과 정확히 같았다. Macro CRPS 절대 차이의 최댓값은 {max(abs(r.get('macro_crps_delta', 0)) for r in replay):.10g}다.
- EOS gradient 제외, validation 난수 복구, 정확한 step 예산, 마지막 가중치 대신 best checkpoint 저장, validation 기반 후보 선택을 단위 테스트로 검사했다.

{replay_note}

![학습률별 validation CRPS](validation_learning_rates.png)

![선택 모델의 초기·중간·후기 SOH 예측](best_checkpoint_windows.png)

설정: `configs/lora_checkpoint.json`. 실행: `python scripts/run_checkpoint_lora.py`.
기존 epoch/refit 실험은 별도 소스·결과로 보존한다. 이 보고서와 03 노트북의 현재 프로토콜은 `src/soh_checkpoint.py`에 구현돼 있다.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "verification.json").write_text(json.dumps({"run": str(directory), "audit": audit, "checkpoint_replay": replay}, indent=2) + "\n", encoding="utf-8")
    print(macro.to_string())
    print(choices.to_string(index=False))
    print("REPORT=" + str(output / "report.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/checkpoint_lora")
    args = parser.parse_args()
    summarize(args.run, args.output)
