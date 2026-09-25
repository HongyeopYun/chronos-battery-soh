"""Compare matched saved CUDA runs, keeping training-seed stability separate."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_experiment import (audit_saved_run, distribution_metrics, point_metrics,
                                resolve_reference, summarize_samples, write_json)


def score_run(directory, label, train_seeds=None):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for path in sorted((directory / "predictions").glob("*.npz")):
        parts = path.stem.split("_")
        train_seed = int(parts[1][5:]) if len(parts) == 3 else -1
        if train_seeds is not None and train_seed not in train_seeds:
            continue
        infer_seed = int(parts[-1][5:])
        with np.load(path, allow_pickle=False) as saved:
            samples, target = saved["samples"], saved["target"]
            summary = summarize_samples(samples)
            median = point_metrics(target, summary["median"])
            mean = point_metrics(target, summary["mean"])
            rows.append({"model": label, "battery": parts[0], "train_seed": train_seed,
                         "inference_seed": infer_seed, "median_rmse": median["rmse"],
                         "median_mae": median["mae"], "mean_rmse": mean["rmse"],
                         **distribution_metrics(target, samples)})
    if not rows:
        raise ValueError(f"No matching predictions in {directory}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/cuda_improvement_report.md"))
    args = parser.parse_args()
    before, after = args.before.resolve(), args.after.resolve()
    bm = json.loads((before / "manifest.json").read_text(encoding="utf-8"))
    am = json.loads((after / "manifest.json").read_text(encoding="utf-8"))
    fields = ("model_id", "revision", "battery_ids", "context_length", "prediction_length",
              "inference_batch_size", "num_samples", "inference_seeds", "temperature", "top_k", "top_p")
    for field in fields:
        if bm["config"][field] != am["config"][field]:
            raise ValueError(f"Unmatched evaluation setting: {field}")
    for field in ("runtime", "data_sha256", "versions"):
        if bm["evaluation_signature"][field] != am["evaluation_signature"][field]:
            raise ValueError(f"Unmatched evaluation environment: {field}")
    common_seeds = sorted(set(bm["config"]["train_seeds"]) & set(am["config"]["train_seeds"]))
    if not common_seeds:
        raise ValueError("No common training seed")
    zero = resolve_reference(ROOT, am["zeroshot_run"])
    audits = [audit_saved_run(path, ROOT) for path in (before, after, zero)]
    previous = score_run(before, "Before", common_seeds)
    improved = score_run(after, "After", common_seeds)
    keys = ["battery", "train_seed", "inference_seed"]
    if set(map(tuple, previous[keys].to_numpy())) != set(map(tuple, improved[keys].to_numpy())):
        raise ValueError("Predictions are not paired by battery and seeds")
    frames = pd.concat([score_run(zero, "Zero-shot"), previous, improved], ignore_index=True)
    values = ["median_rmse", "median_mae", "mean_rmse", "crps", "coverage80", "width80",
              "negative_sample_rate", "above_two_sample_rate"]
    per_battery = frames.groupby(["model", "battery"])[values].mean()
    macro = per_battery.groupby("model").mean().reindex(["Zero-shot", "Before", "After"])
    stability = score_run(after, "After").groupby(["train_seed", "inference_seed"])[values].mean()
    train_scores = stability.groupby("train_seed").mean()
    gains = 100 * (macro.loc["Before"] - macro.loc["After"]) / macro.loc["Before"]
    lines = ["# CUDA 개선 전후 실험", "",
             f"수정 전: `{before.name}`", f"수정 후: `{after.name}`", f"Zero-shot: `{zero.name}`", "",
             f"동일한 CUDA 환경·데이터·모델 revision·100 samples·추론 seed 42/43/44를 사용했습니다. "
             f"주 비교의 공통 학습 seed: `{common_seeds}`.", "",
             "수정 후 추가 학습 seed의 결과는 아래 안정성 표에 별도로 표시합니다. "
             "평가 배터리의 결과로 설정이나 checkpoint를 선택하지 않았습니다.", "",
             "| 모델 | Median RMSE | Median MAE | Mean RMSE | CRPS | 80% 구간 폭 | SOH<0 | SOH>2 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, row in macro.iterrows():
        lines.append(f"| {label} | {row.median_rmse:.6f} | {row.median_mae:.6f} | {row.mean_rmse:.6f} | "
                     f"{row.crps:.6f} | {row.width80:.6f} | {row.negative_sample_rate:.4%} | {row.above_two_sample_rate:.4%} |")
    lines += ["", f"수정 전 대비 감소율: median RMSE {gains.median_rmse:+.2f}%, "
              f"median MAE {gains.median_mae:+.2f}%, mean RMSE {gains.mean_rmse:+.2f}%, CRPS {gains.crps:+.2f}%.", "",
              "## 배터리별 동일 seed 비교", "", "| 배터리 | 이전 RMSE | 이후 RMSE | RMSE 감소율 | 이전 CRPS | 이후 CRPS |",
              "|---|---:|---:|---:|---:|---:|"]
    for battery in am["config"]["battery_ids"]:
        b, a = per_battery.loc[("Before", battery)], per_battery.loc[("After", battery)]
        lines.append(f"| {battery} | {b.median_rmse:.6f} | {a.median_rmse:.6f} | "
                     f"{100*(1-a.median_rmse/b.median_rmse):+.2f}% | {b.crps:.6f} | {a.crps:.6f} |")
    lines += ["", "## 개선 모델의 학습 seed 반복", "",
              "각 행은 추론 seed 3개와 배터리 4개를 동일 가중치로 평균한 값입니다.", "",
              "| 학습 seed | Median RMSE | Median MAE | Mean RMSE | CRPS |", "|---|---:|---:|---:|---:|"]
    for seed, row in train_scores.iterrows():
        lines.append(f"| {seed} | {row.median_rmse:.6f} | {row.median_mae:.6f} | {row.mean_rmse:.6f} | {row.crps:.6f} |")
    fallback = sum(fold["selected_epoch"] == 0 for fold in am["folds"])
    paired_folds = [fold for fold in am["folds"] if fold["train_seed"] in common_seeds]
    paired_fallback = sum(fold["selected_epoch"] == 0 for fold in paired_folds)
    lines += ["", f"전체 {len(am['folds'])}개 모델 중 {fallback}개가 epoch 0을 선택했습니다. "
              f"공통 학습 seed의 {len(paired_folds)}개 모델 중 epoch 0 선택은 {paired_fallback}개입니다."]
    if paired_fallback == len(paired_folds):
        lines += ["", "**주 비교의 개선 모델은 모든 fold에서 미학습 adapter를 선택해 zero-shot과 동일한 예측을 냈습니다. "
                  "이 결과는 나빠진 미세조정을 거부한 효과이며, 새 LoRA 학습이 zero-shot보다 우수하다는 증거가 아닙니다.**"]
    lines += ["", "## 해석 범위", "",
              "- 검증 기준, learning rate, KL 가중치는 개선 실험 시작 전에 고정했습니다.",
              "- 원시 예측에 clipping을 적용하지 않았습니다. SOH>2 비율은 분포 진단값입니다.",
              "- 개선 모델은 시간순 validation에서 선택된 checkpoint를 그대로 사용합니다. 이전 모델의 전체 데이터 refit과 학습 프로토콜이 다릅니다.",
              "- 네 배터리와 중첩 window 결과이며, 추가 데이터에서의 성능이나 통계적 유의성을 보장하지 않습니다.",
              "- 모든 예측·데이터 해시와 지표를 재검산했습니다. 실행별 manifest/NPZ/checkpoint는 로컬 results/runs/ 및 models/에 보존합니다."]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    frames.to_csv(args.output.with_suffix(".csv"), index=False)
    write_json(args.output.with_suffix(".json"), {"before": before.name, "after": after.name,
        "zero": zero.name, "common_train_seeds": common_seeds, "audits": audits,
        "epoch_zero_models": fallback, "total_models": len(am["folds"]),
        "macro": macro.to_dict(orient="index"), "training_seed_scores": train_scores.to_dict(orient="index")})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, ["median_rmse", "mean_rmse", "crps"], ["Median RMSE", "Mean RMSE", "CRPS"]):
        ax.bar(macro.index, macro[metric], color=["#64748b", "#cc7755", "#2c8b78"])
        ax.set_title(title)
        ax.set_ylabel("SOH fraction (lower is better)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"Matched CUDA evaluation: training seed(s) {common_seeds}")
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    plt.close(fig)
    print(macro.to_string(), flush=True)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
