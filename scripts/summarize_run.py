"""Export a concise report and a standalone comparison figure from saved runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/improvement_summary.md"))
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Report requires a complete run")
    data = pd.read_csv(run / "paired_comparison.csv")
    median = data[data.estimator == "median"]
    metrics = ["rmse_zero", "rmse_lora", "mae_zero", "mae_lora"]
    grouped = median.groupby("battery")[metrics].mean()
    macro = grouped.mean()
    improvement = 100 * (1 - macro.rmse_lora / macro.rmse_zero)
    mae_improvement = 100 * (1 - macro.mae_lora / macro.mae_zero)
    config = manifest["config"]
    lines = ["# 개선 후 실험 결과", "", f"실행: `{run.name}`", "",
             f"Zero-shot 참조: `{Path(manifest['zeroshot_run']).name}`", "",
             f"설정: context={config['context_length']}, horizon={config['prediction_length']}, samples={config['num_samples']}, "
             f"학습 seed={config['train_seeds']}, 추론 seed={config['inference_seeds']}.", "",
             "주 지표는 보간 중앙값의 RMSE/MAE입니다. 아래 표는 추론 및 학습 seed별 점수를 평균한 값입니다.", "",
             "| 배터리 | Zero-shot RMSE | LoRA RMSE | RMSE 개선율 | Zero-shot MAE | LoRA MAE |",
             "|---|---:|---:|---:|---:|---:|"]
    for battery, row in grouped.iterrows():
        gain = 100 * (1 - row.rmse_lora / row.rmse_zero)
        lines.append(f"| {battery} | {row.rmse_zero:.6f} | {row.rmse_lora:.6f} | {gain:+.2f}% | {row.mae_zero:.6f} | {row.mae_lora:.6f} |")
    lines.extend([f"| Macro | {macro.rmse_zero:.6f} | {macro.rmse_lora:.6f} | {improvement:+.2f}% | {macro.mae_zero:.6f} | {macro.mae_lora:.6f} |", "",
                  f"Macro RMSE 변화: {improvement:+.2f}% 개선. Macro MAE 변화: {mae_improvement:+.2f}% 개선.", "",
                  "## 검증에서 선택한 epoch", "",
                  "| 평가 배터리 | 학습 seed | 선택 epoch |",
                  "|---|---:|---:|"])
    for fold in manifest["folds"]:
        lines.append(f"| {fold['test_battery']} | {fold['train_seed']} | {fold['selected_epoch']} |")
    run_scores = pd.read_csv(run / "paired_macro.csv")
    lines.extend(["", "## 추론 반복별 결과", "", "| 학습 seed | 추론 seed | Zero-shot macro RMSE | LoRA macro RMSE | 개선율 |",
                  "|---|---|---:|---:|---:|"])
    for row in run_scores[run_scores.estimator == "median"].itertuples(index=False):
        lines.append(f"| {row.train_seed} | {row.inference_seed} | {row.rmse_zero:.6f} | {row.rmse_lora:.6f} | {row.rmse_improvement_pct:+.2f}% |")
    lines.extend(["", "## 원시 예측 분포 진단", "",
                  "| 모델 | 전체 샘플 최솟값 | 전체 샘플 최댓값 | SOH < 0 샘플 비율 | SOH > 2 샘플 비율 |",
                  "|---|---:|---:|---:|---:|"])
    for label, path in [("Zero-shot", Path(manifest["zeroshot_run"])), ("LoRA", run)]:
        low, high, count, large, negative = float("inf"), float("-inf"), 0, 0, 0
        for file in (path / "predictions").glob("*.npz"):
            with np.load(file, allow_pickle=False) as saved:
                values = saved["samples"]
                low, high = min(low, float(values.min())), max(high, float(values.max()))
                count += values.size
                large += int((values > 2).sum())
                negative += int((values < 0).sum())
        lines.append(f"| {label} | {low:.6f} | {high:.6f} | {negative / count:.6%} | {large / count:.6%} |")
    lines.extend(["", "SOH는 Qk/Q1입니다. 드문 극단 샘플은 sample mean에 큰 영향을 줄 수 있습니다. "
                  "원시 samples를 보존했으며 임의의 clipping이나 평가 결과 기반 집계 변경을 적용하지 않았습니다."])
    lines.extend(["", "**중앙값 점수 개선만으로 예측 분포도 개선되었다고 결론내릴 수 없습니다.**", "",
                  "| 모델 | Sample mean의 macro RMSE | Sample mean의 macro MAE | 평균 80% coverage | 평균 구간 폭 |",
                  "|---|---:|---:|---:|---:|"])
    for label, path in [("Zero-shot", Path(manifest["zeroshot_run"])), ("LoRA", run)]:
        summary = pd.read_csv(path / "summary.csv")
        score = summary[(summary.estimator == "mean") & summary.model.isin(["ZeroShot", "LoRA"])].iloc[0]
        metrics_frame = pd.read_csv(path / "metrics.csv")
        intervals = metrics_frame[(metrics_frame.estimator == "median") & metrics_frame.model.isin(["ZeroShot", "LoRA"])]
        intervals = intervals.groupby("battery")[["coverage80", "width80"]].mean().mean()
        lines.append(f"| {label} | {score.rmse_mean:.6f} | {score.mae_mean:.6f} | {intervals.coverage80:.2%} | {intervals.width80:.6f} |")
    lines.extend(["", "Validation은 중앙값 RMSE로 epoch를 선택했습니다. 극단 샘플의 빈도나 분포 전체의 품질을 직접 최적화하지 않았으므로, "
                  "향후 확률 예측을 개선하려면 validation에서 CRPS 등 분포 지표와 비현실적 샘플 비율을 함께 평가해야 합니다. "
                  "이번 test 결과로 사후에 최적 epoch나 집계 규칙을 다시 고르지는 않았습니다."])
    lines.extend(["", "## 해석 범위", "",
                  "- 평가 배터리를 제외한 3개 배터리의 시간순 validation으로 epoch를 선택하고 전체 학습 데이터로 재학습했습니다.",
                  "- 저장된 원시 samples에서 자동으로 계산한 지표이며 과거 하드코딩 비교표를 사용하지 않았습니다.",
                  "- 추론 seed 반복은 학습 초기화 반복을 대신하지 않습니다. 학습 seed가 1개이면 학습 변동성은 평가되지 않았습니다.",
                  "- 중첩 윈도우는 독립 표본이 아니며, 위 변화율은 통계적 유의성의 증거가 아닙니다.",
                  "- 기존 20-sample/lower-median 결과와 새 점수를 직접 비교하지 않고 동일 조건으로 재실행한 zero-shot을 기준으로 합니다.",
                  "- 같은 실험의 mean estimator, horizon별 오차, coverage 및 window별 오차는 실행 디렉터리의 CSV에 저장되어 있습니다."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    positions = list(range(len(grouped)))
    for ax, metric, title in zip(axes, ["rmse", "mae"], ["RMSE", "MAE"]):
        ax.bar([p - 0.18 for p in positions], grouped[f"{metric}_zero"], width=0.36, label="Zero-shot", color="#527a9f")
        ax.bar([p + 0.18 for p in positions], grouped[f"{metric}_lora"], width=0.36, label="LoRA", color="#cf7f45")
        ax.set_xticks(positions, grouped.index)
        ax.set_ylabel(title + " (SOH fraction)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend()
    figure.suptitle(f"NASA SOH: matched forecasting protocol\n{config['num_samples']} samples; "
                    f"{len(config['inference_seeds'])} inference seeds; {len(config['train_seeds'])} training seed(s)")
    figure.tight_layout()
    figure.savefig(args.output.with_suffix(".png"), dpi=180)
    plt.close(figure)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
