"""Recompute a compact report from an audited LoRA ablation run."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_experiment import audit_saved_run, file_sha256, resolve_reference, write_json
from src.soh_ablation import choose_candidate
from compare_runs import score_run


def summarize(directory, output):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["kind"] != "lora_ablation":
        raise ValueError("Expected a LoRA ablation run")
    choices_path = directory / "choices.json"
    if file_sha256(choices_path) != manifest["choices_sha256"]:
        raise ValueError("Frozen validation choices changed")
    choices = json.loads(choices_path.read_text(encoding="utf-8"))
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    search_seed = protocol["protocol"]["search_train_seed"]
    candidate_audits = []
    for heldout, chosen in choices["choices"].items():
        records = {}
        for name in protocol["candidate_configs"]:
            label = f"{heldout}_{name}_train{search_seed}"
            selection_path = directory / "selection" / f"{label}.json"
            record = json.loads(selection_path.read_text(encoding="utf-8"))
            if set(record["train_batteries"]) != set(manifest["config"]["battery_ids"]) - {heldout}:
                raise ValueError("Incorrect training/validation battery partition")
            if any(heldout in row["by_battery"] for row in record["history"]):
                raise ValueError("Held-out battery entered validation")
            records[name] = record
        if choose_candidate(records, manifest["config"]["min_delta"]) != chosen:
            raise ValueError("Stored candidate choice differs from validation-only rule")
        baselines = [r["baseline_validation"] for r in records.values()]
        if any(b != baselines[0] for b in baselines[1:]):
            raise ValueError("Candidate epoch-zero validation scores differ")
    for fold in manifest["folds"]:
        heldout, name, seed = fold["test_battery"], fold["candidate"], fold["train_seed"]
        if choices["choices"][heldout] != name or fold["config"] != protocol["candidate_configs"][name]:
            raise ValueError("Final fold differs from frozen candidate configuration")
        record = json.loads((directory / "selection" / f"{heldout}_{name}_train{seed}.json").read_text(encoding="utf-8"))
        if fold["selected_epoch"] != record["selected_epoch"]:
            raise ValueError("Final epoch differs from validation-selected checkpoint")
    for path in sorted((directory / "receipts").glob("*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if file_sha256(directory / "selection" / path.name) != receipt["selection_sha256"]:
            raise ValueError(f"Changed selection receipt: {path.name}")
        for name, digest in receipt["adapter_sha256"].items():
            if file_sha256(ROOT / "models" / directory.name / path.stem / name) != digest:
                raise ValueError(f"Changed search adapter: {path.stem}/{name}")
        candidate_audits.append(path.stem)
    baseline = resolve_reference(ROOT, manifest["zeroshot_run"])
    audits = [audit_saved_run(directory, ROOT, verify_adapters=True), audit_saved_run(baseline, ROOT)]
    frames = pd.concat([score_run(baseline, "Zero-shot"), score_run(directory, "Selected LoRA")], ignore_index=True)
    columns = ["median_rmse", "median_mae", "mean_rmse", "crps", "coverage80", "width80"]
    per_battery = frames.groupby(["model", "battery"])[columns].mean()
    macro = per_battery.groupby("model").mean().reindex(["Zero-shot", "Selected LoRA"])
    seeds = frames[frames.model == "Selected LoRA"].groupby(["train_seed", "battery"])[columns].mean().groupby("train_seed").mean()
    gains = 100 * (1 - macro.loc["Selected LoRA"] / macro.loc["Zero-shot"])
    fallback = sum(f["selected_epoch"] == 0 for f in manifest["folds"])
    model_count = len(manifest["folds"])
    lines = ["# LoRA 원인 분리 실험", "", f"실행: `{directory.name}`", "",
             "학습 seed 42에서 학습률·KL·조기 종료를 비교하고, 각 outer fold의 다른 세 배터리 validation만으로 설정을 선택했습니다. "
             "네 fold의 설정을 모두 고정한 뒤 학습 seed 42/43/44, 추론 seed 42/43/44로 평가했습니다. "
             "모든 값은 배터리에 동일 가중치를 준 평균입니다.", "",
             "| 모델 | Median RMSE | Median MAE | Mean RMSE | CRPS | 80% coverage | 80% width |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in macro.iterrows():
        lines.append(f"| {name} | {row.median_rmse:.6f} | {row.median_mae:.6f} | {row.mean_rmse:.6f} | {row.crps:.6f} | {row.coverage80:.2%} | {row.width80:.6f} |")
    lines += ["", f"Zero-shot 대비 감소율: median RMSE **{gains.median_rmse:+.3f}%**, median MAE {gains.median_mae:+.3f}%, "
              f"mean RMSE {gains.mean_rmse:+.3f}%, CRPS {gains.crps:+.3f}%.", "",
              f"{model_count}개 최종 모델 중 epoch 0 선택은 **{fallback}개**, 학습된 checkpoint 선택은 **{model_count-fallback}개**입니다. "
              "Epoch 0의 성능은 LoRA 학습 이득으로 계산하지 않습니다.", "",
              "## 배터리별 결과", "", "| 배터리 | 선택 설정 | Zero-shot RMSE | LoRA RMSE | RMSE 감소율 | 학습 seed별 epoch (42/43/44) |",
              "|---|---|---:|---:|---:|---|"]
    for battery, name in choices["choices"].items():
        z, l = per_battery.loc[("Zero-shot", battery)], per_battery.loc[("Selected LoRA", battery)]
        epochs = "/".join(str(f["selected_epoch"]) for f in manifest["folds"] if f["test_battery"] == battery)
        lines.append(f"| {battery} | {name} | {z.median_rmse:.6f} | {l.median_rmse:.6f} | {100*(1-l.median_rmse/z.median_rmse):+.3f}% | {epochs} |")
    lines += ["", "## 학습 seed 반복", "", "| 학습 seed | Median RMSE | Median MAE | Mean RMSE | CRPS |", "|---|---:|---:|---:|---:|"]
    for seed, row in seeds.iterrows():
        lines.append(f"| {seed} | {row.median_rmse:.6f} | {row.median_mae:.6f} | {row.mean_rmse:.6f} | {row.crps:.6f} |")
    lines += ["", "## Validation에서 분리한 영향", "",
              "`longer_training`은 LR 3e-5, KL 0.5에서 patience만 3→10으로 변경했습니다. "
              "`higher_lr`은 이 설정에서 LR만 1e-4로, `without_kl`은 KL만 0으로 변경했습니다. "
              "모두 최대 10 epoch, validation 64 samples, 같은 초기화와 학습 seed입니다.", "",
              "짧은 학습 결과는 같은 학습 경로에서 patience=3을 재현한 것입니다. "
              "중앙값 최소 epoch는 guard의 영향을 진단하는 값이며 최종 모델 선택에는 쓰지 않았습니다.", "",
              "| 평가 제외 배터리 | 설정 | patience 3 epoch | patience 10 epoch | 선택 CRPS | 중앙값 최소 epoch | 해당 mean RMSE / epoch 0 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    diagnostic_rows = []
    for battery, records in choices["validation_search"].items():
        for name, record in records.items():
            median = record["median_only_best"]
            ratio = median["validation"]["mean_rmse"] / record["baseline_validation"]["mean_rmse"]
            lines.append(f"| {battery} | {name} | {record['short_policy']['selected_epoch']} | {record['selected_epoch']} | "
                         f"{record['selected_validation']['crps']:.6f} | {median['epoch']} | {ratio:.3f}x |")
            diagnostic_rows.append({"heldout": battery, "candidate": name,
                "short_epoch": record["short_policy"]["selected_epoch"], "full_epoch": record["selected_epoch"],
                "validation_crps": record["selected_validation"]["crps"], "median_best_epoch": median["epoch"],
                "median_best_mean_rmse_ratio": ratio})
    lines += ["", "## 해석 범위", "",
              "- 학습률과 KL의 상호작용을 탐색하는 전체 grid는 아닙니다. 이번 결과는 조사한 세 설정에 한정됩니다.",
              "- 후보 설정 탐색 seed는 42 하나입니다. 43/44는 선택된 설정의 반복이며 독립적인 후보 탐색 반복은 아닙니다.",
              "- validation은 학습 배터리의 뒤쪽 구간입니다. 새 배터리의 분포 차이를 완전히 대변하지 않습니다.",
              "- 이 네 배터리는 이전 실험에서도 확인했습니다. 새 독립 benchmark나 통계적 유의성 검증으로 해석하지 않습니다.",
              "- 추론 seed와 겹치는 window를 독립 배터리 표본으로 취급하지 않습니다.",
              f"- 원본 NPZ의 지표·해시를 재검산했고, {model_count}개 adapter를 재로딩해 첫 추론 batch가 정확히 재현되는지 확인했습니다.",
              "", "![검증 학습 곡선](lora_ablation_validation.png)", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    frames.to_csv(output.with_suffix(".csv"), index=False)
    pd.DataFrame(diagnostic_rows).to_csv(output.with_name(output.stem + "_validation.csv"), index=False)
    write_json(output.with_suffix(".json"), {"run": directory.relative_to(ROOT).as_posix(),
        "zero": baseline.relative_to(ROOT).as_posix(), "audits": audits,
        "macro": macro.to_dict(orient="index"), "training_seed_scores": seeds.to_dict(orient="index"),
        "search_and_repeat_receipts_verified": candidate_audits, "validation_choices_recomputed": True,
        "improvement_pct": gains.to_dict(), "epoch_zero_models": fallback,
        "choices": choices["choices"], "folds": [{k: f[k] for k in ("test_battery", "train_seed", "candidate", "selected_epoch")} for f in manifest["folds"]],
        "reload_checks": json.loads((directory / "reload_checks.json").read_text(encoding="utf-8"))})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (battery, records) in zip(axes.flat, choices["validation_search"].items()):
        for name, record in records.items():
            history = json.loads((directory / "selection" / f"{battery}_{name}_train{search_seed}.json").read_text(encoding="utf-8"))["history"]
            ax.plot([r["epoch"] for r in history], [r["validation"]["crps"] for r in history], marker=".", label=name)
        ax.axhline(next(iter(records.values()))["baseline_validation"]["crps"], color="gray", linestyle="--", label="Epoch 0")
        ax.set(title=f"Held out: {battery} (validation uses other 3)", xlabel="Epoch", ylabel="Validation macro CRPS")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output.with_name("lora_ablation_validation.png"), dpi=160)
    plt.close(fig)
    print(macro.to_string(), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/lora_ablation_report.md"))
    args = parser.parse_args()
    summarize(args.run.resolve(), args.output.resolve())
