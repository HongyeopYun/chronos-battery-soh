"""Audit raw forecasts and report the predeclared robust LoRA follow-up."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import soh_experiment as e


POLICIES = ("single_seed10", "multi_seed10", "multi_seed20", "balanced20")
FINAL_POLICIES = POLICIES[1:]
VALUES = ("median_rmse", "median_mae", "mean_rmse", "crps", "coverage80", "width80",
          "negative_sample_rate", "above_two_sample_rate")
KEYS = ("battery", "train_seed", "inference_seed")
PREDICTION_NAME = re.compile(r"^(?P<battery>B\d+)(?:_train(?P<train>-?\d+))?_infer(?P<infer>\d+)\.npz$")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_table(directory, filename, rows, row_keys, model):
    actual = pd.read_csv(directory / filename)
    if model == "ZeroShot":
        actual = actual.loc[actual["model"] == model]
    expected = pd.DataFrame(rows)
    if set(actual.columns) != set(expected.columns):
        raise ValueError(f"Unexpected columns in {directory.name}/{filename}")
    if actual.duplicated(row_keys).any():
        raise ValueError(f"Duplicate metric rows in {directory.name}/{filename}")
    actual = actual[expected.columns].sort_values(row_keys).reset_index(drop=True)
    expected = expected.sort_values(row_keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False,
                                  rtol=0, atol=1e-12)


def score_predictions(directory, label, config, series, training_seeds):
    """Recompute every saved sample summary and table; require the complete grid."""
    model = "ZeroShot" if training_seeds == [-1] else "LoRA"
    expected_keys = set(product(config.battery_ids, training_seeds, config.inference_seeds))
    observed, scores, rows, horizons, windows = set(), [], [], [], []
    for path in sorted((directory / "predictions").glob("*.npz")):
        match = PREDICTION_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected prediction filename: {path.name}")
        battery = match["battery"]
        seed = int(match["train"]) if match["train"] is not None else -1
        infer = int(match["infer"])
        key = (battery, seed, infer)
        if key not in expected_keys or key in observed:
            raise ValueError(f"Unexpected or repeated forecast: {path}")
        observed.add(key)
        reference = e.make_windows(series[battery], config.context_length, config.prediction_length)
        with np.load(path, allow_pickle=False) as saved:
            for name in ("context", "target", "forecast_start"):
                np.testing.assert_array_equal(saved[name], reference[name])
            samples = saved["samples"]
            expected_shape = (len(reference["target"]), config.num_samples, config.prediction_length)
            if samples.shape != expected_shape or not np.isfinite(samples).all():
                raise ValueError(f"Invalid samples: {path}")
            summary = e.summarize_samples(samples)
            for name, value in summary.items():
                np.testing.assert_array_equal(saved[name], value)
            r, h, w = e.metric_rows(reference, summary, model, battery, seed, infer, samples=samples)
            rows.extend(r)
            horizons.extend(h)
            windows.extend(w)
            median = e.point_metrics(reference["target"], summary["median"])
            mean = e.point_metrics(reference["target"], summary["mean"])
            scores.append({"model": label, "battery": battery, "train_seed": seed,
                           "inference_seed": infer, "median_rmse": median["rmse"],
                           "median_mae": median["mae"], "mean_rmse": mean["rmse"],
                           **e.distribution_metrics(reference["target"], samples)})
    if observed != expected_keys:
        raise ValueError(f"Missing predictions in {directory}: {sorted(expected_keys - observed)}")
    keys = ["model", "battery", "train_seed", "inference_seed", "estimator"]
    for filename, values, extra in (("metrics.csv", rows, []),
                                     ("horizon_metrics.csv", horizons, ["horizon"]),
                                     ("window_metrics.csv", windows, ["forecast_start"])):
        _assert_table(directory, filename, values, keys + extra, model)
    return pd.DataFrame(scores), {"model": label, "forecast_files_verified": len(observed),
                                  "metric_rows_verified": len(rows), "all_three_tables_verified": True}


def verify_decisions(directory, manifest, decisions):
    if decisions.get("test_results_used") is not False:
        raise ValueError("Decisions must explicitly exclude held-out results")
    for key in ("decisions_sha256", "search_decisions_sha256"):
        if key in manifest and manifest[key] != e.file_sha256(directory / "search_decisions.json"):
            raise ValueError("Frozen decisions changed")
    batteries = set(manifest["config"]["battery_ids"])
    if set(decisions["choices"]) != batteries or set(decisions["policies"]) != batteries:
        raise ValueError("Missing validation decisions")
    delta = manifest["config"]["min_delta"]
    for battery, records in decisions["policies"].items():
        if set(records) != set(POLICIES):
            raise ValueError(f"Unexpected policy set for {battery}")
        for record in records.values():
            config = record["config"]
            if config["learning_rate"] != 3e-5 or config["kl_weight"] != 0.5:
                raise ValueError("Policy differs from the fixed LR/KL anchor described in this report")
            if config["selection_metric"] != "guarded_crps" or config["refit_full_data"]:
                raise ValueError("Expected guarded selection without full-data refitting")
        winner = FINAL_POLICIES[0]
        baseline = records[winner]["baseline_validation"]
        for policy in FINAL_POLICIES:
            row = records[policy]
            if row["baseline_validation"] != baseline:
                raise ValueError("Multi-seed policies must use the same epoch-zero validation")
            score = row["selected_validation"]
            for metric in ("median_rmse", "mean_rmse"):
                if not np.isfinite(score[metric]) or score[metric] > baseline[metric] + delta:
                    raise ValueError(f"Selected policy violates validation guard: {battery}/{policy}")
            if not np.isfinite(score["crps"]):
                raise ValueError("Nonfinite validation CRPS")
            if score["crps"] < records[winner]["selected_validation"]["crps"] - delta:
                winner = policy
        if decisions["choices"][battery] != winner:
            raise ValueError(f"Final choice differs from the predeclared CRPS rule: {battery}")
    expected = set(product(batteries, manifest["config"]["train_seeds"]))
    observed = set()
    for fold in manifest["folds"]:
        battery, seed = fold["test_battery"], fold["train_seed"]
        if (battery, seed) not in expected or (battery, seed) in observed:
            raise ValueError("Unexpected or repeated final fold")
        observed.add((battery, seed))
        if set(fold["train_batteries"]) != batteries - {battery}:
            raise ValueError("Held-out battery entered training")
        chosen = decisions["choices"][battery]
        if fold["selected_policy"] != chosen:
            raise ValueError("Final fold differs from frozen policy")
        if fold["config"] != decisions["policies"][battery][chosen]["config"]:
            raise ValueError("Final training configuration differs from frozen policy")
        if seed == 42 and fold["selected_epoch"] != decisions["policies"][battery][chosen]["selected_epoch"]:
            raise ValueError("Search-seed checkpoint differs from frozen policy")
    if observed != expected:
        raise ValueError("Missing final folds")


def _macro(frame):
    return frame.groupby(["model", "battery"])[list(VALUES)].mean().groupby("model").mean()


def _table(frame):
    lines = ["| 모델/정책 | Median RMSE | Median MAE | Mean RMSE | CRPS | 80% coverage | 80% width |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, row in frame.iterrows():
        lines.append(f"| {label} | {row.median_rmse:.6f} | {row.median_mae:.6f} | "
                     f"{row.mean_rmse:.6f} | {row.crps:.6f} | {row.coverage80:.2%} | {row.width80:.6f} |")
    return lines


def summarize(directory, output, root=ROOT):
    directory, output, root = directory.resolve(), output.resolve(), root.resolve()
    manifest = read_json(directory / "manifest.json")
    if manifest["kind"] != "lora_robust":
        raise ValueError("Expected a completed lora_robust run")
    decisions = read_json(directory / "search_decisions.json")
    verify_decisions(directory, manifest, decisions)
    config = e.ExperimentConfig(**manifest["config"])
    zero = e.resolve_reference(root, manifest["zeroshot_run"])
    zero_manifest = read_json(zero / "manifest.json")
    if zero_manifest["kind"] != "zeroshot" or manifest["evaluation_signature"] != zero_manifest["evaluation_signature"]:
        raise ValueError("Baseline evaluation settings differ")
    audits = [e.audit_saved_run(directory, root, verify_adapters=True), e.audit_saved_run(zero, root)]
    series = e.load_soh(root, config)
    frames, prediction_audits = [], []
    for path, label, seeds in [(zero, "Zero-shot", [-1]), (directory, "Selected LoRA", list(config.train_seeds))] + [
            (directory / "policy_evaluation" / policy, policy, [42]) for policy in POLICIES]:
        frame, audit = score_predictions(path, label, config, series, seeds)
        frames.append(frame)
        prediction_audits.append(audit)
    scored = pd.concat(frames, ignore_index=True)
    # Search-seed final forecasts must be the already frozen chosen policy,
    # not merely a similarly scoring model evaluated under a different label.
    for battery, policy in decisions["choices"].items():
        for infer_seed in config.inference_seeds:
            filename = f"{battery}_train42_infer{infer_seed}.npz"
            with np.load(directory / "predictions" / filename, allow_pickle=False) as selected, np.load(
                    directory / "policy_evaluation" / policy / "predictions" / filename, allow_pickle=False) as candidate:
                np.testing.assert_array_equal(selected["samples"], candidate["samples"])
    macro = _macro(scored)
    final_macro = macro.loc[["Zero-shot", "Selected LoRA"]]
    policy_macro = macro.loc[["Zero-shot", *POLICIES]]
    final = scored[scored["model"] == "Selected LoRA"]
    seeds = final.groupby(["train_seed", "battery"])[list(VALUES)].mean().groupby("train_seed").mean()
    gains = 100 * (1 - final_macro.loc["Selected LoRA"] / final_macro.loc["Zero-shot"])
    contrasts = []
    for previous, current, label in (("single_seed10", "multi_seed10", "검증 seed 확대"),
                                     ("multi_seed10", "multi_seed20", "학습 상한 10→20"),
                                     ("multi_seed20", "balanced20", "배터리 균형 loss")):
        changes = 100 * (1 - macro.loc[current] / macro.loc[previous])
        contrasts.append({"factor": label, "before": previous, "after": current,
                          **{key + "_reduction_pct": float(changes[key]) for key in ("median_rmse", "mean_rmse", "crps")}})
    epoch_zero = sum(fold["selected_epoch"] == 0 for fold in manifest["folds"])
    per_battery = scored.groupby(["model", "battery"])[list(VALUES)].mean()
    policy_zero = {policy: sum(records[policy]["selected_epoch"] == 0 for records in decisions["policies"].values())
                   for policy in POLICIES}
    lines = ["# LoRA 추가 개선 실험", "", f"실행: `{directory.name}`", "",
             "평가 배터리를 제외한 세 배터리의 시간순 validation으로 정책을 선택했습니다. "
             "모든 fold의 선택을 고정한 다음 정책별 평가와 선택 설정의 학습 seed 반복을 수행했습니다. "
             "지표는 추론 seed와 배터리에 동일 가중치를 준 평균입니다.", "",
             "## 최종 선택 결과", "", *_table(final_macro), "",
             f"Zero-shot 대비 감소율: median RMSE **{gains.median_rmse:+.3f}%**, MAE {gains.median_mae:+.3f}%, "
             f"mean RMSE {gains.mean_rmse:+.3f}%, CRPS {gains.crps:+.3f}%. 양수는 오차 감소입니다.", "",
             f"최종 {len(manifest['folds'])}개 모델 중 epoch 0 선택은 **{epoch_zero}개**입니다. "
             "Epoch 0으로 돌아간 결과를 LoRA 학습의 개선으로 해석하지 않습니다.", "",
             "## 요인별 비교: 학습 seed 42", "",
             "학습률 3e-5, KL 0.5를 고정했습니다. `single_seed10`은 검증 seed 31415 하나, "
             "나머지는 31415–31419의 seed별·배터리별 지표 평균을 사용합니다. "
             "각 seed는 64 samples이며, 예측을 합쳐 320 samples로 다시 계산하지 않습니다. "
             "모든 정책은 학습 배터리의 validation context를 결합해 batch 32로 추론한 뒤 배터리별 지표를 분리합니다. "
             "따라서 single-seed 진단도 같은 계산 순서를 사용하는 대조군이며, 이전의 배터리별 호출을 그대로 재현한 결과는 아닙니다. "
             "10/20 epoch 정책은 같은 학습 경로에서 선택하고 조기 종료를 적용하지 않습니다. "
             "`single_seed10`은 진단용이며 최종 정책 선택 대상에서 제외했습니다.", "", *_table(policy_macro), "",
             "| 변경 요인 | Median RMSE 감소율 | Mean RMSE 감소율 | CRPS 감소율 |", "|---|---:|---:|---:|"]
    for row in contrasts:
        lines.append(f"| {row['factor']} | {row['median_rmse_reduction_pct']:+.3f}% | "
                     f"{row['mean_rmse_reduction_pct']:+.3f}% | {row['crps_reduction_pct']:+.3f}% |")
    lines += ["", "위 감소율은 고정한 정책의 held-out 결과입니다. 이 결과로 최종 정책을 다시 고르지 않았습니다.", "",
              "## 배터리별 최종 결과", "", "| 배터리 | 선택 정책 | Zero-shot RMSE | LoRA RMSE | 감소율 | 학습 seed별 epoch |",
              "|---|---|---:|---:|---:|---|"]
    for battery in config.battery_ids:
        base, selected = per_battery.loc[("Zero-shot", battery)], per_battery.loc[("Selected LoRA", battery)]
        folds = sorted((f for f in manifest["folds"] if f["test_battery"] == battery), key=lambda f: f["train_seed"])
        epochs = ", ".join(f"{f['train_seed']}:{f['selected_epoch']}" for f in folds)
        lines.append(f"| {battery} | {decisions['choices'][battery]} | {base.median_rmse:.6f} | "
                     f"{selected.median_rmse:.6f} | {100*(1-selected.median_rmse/base.median_rmse):+.3f}% | {epochs} |")
    lines += ["", "## 선택 설정의 학습 seed 반복", "", *_table(seeds), "",
              "## Validation에서 선택한 epoch", "", "| 평가 제외 배터리 | single_seed10 | multi_seed10 | multi_seed20 | balanced20 |",
              "|---|---:|---:|---:|---:|"]
    for battery in config.battery_ids:
        lines.append("| " + battery + " | " + " | ".join(str(decisions["policies"][battery][p]["selected_epoch"]) for p in POLICIES) + " |")
    lines += ["", "## 해석 범위", "",
              "- 이 네 배터리는 이전 실험에서도 관찰했습니다. 새 독립 benchmark나 통계적 유의성 검증이 아닙니다.",
              "- 이전 실험은 fold별 KL 선택이 달랐습니다. 이번 요인 비교는 LR 3e-5·KL 0.5 고정 기준이며, 이전 설정에서 한 요인만 바꾼 비교가 아닙니다.",
              "- 후보 검색은 학습 seed 42에서 수행했습니다. 43/44는 선택 설정의 반복이지 독립적인 후보 검색 반복은 아닙니다.",
              "- 20 epoch 후보에는 10 epoch의 checkpoint가 포함됩니다. Validation 최솟값 감소 자체를 일반화 개선으로 보지 않습니다.",
              "- 5개 validation seed는 Monte Carlo 변동을 줄이지만 새 배터리 데이터를 추가하지 않습니다. 겹치는 window도 독립 표본이 아닙니다.",
              "- 평균 예측 RMSE guard와 SOH<0·SOH>2 진단을 유지했습니다. 원시 예측을 clipping하지 않았습니다.",
              "- 원시 NPZ에서 summary와 지표를 다시 계산하고 metrics·horizon·window 테이블, 전체 run 해시 및 최종 adapter 해시를 확인했습니다.", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    scored.to_csv(output.with_suffix(".csv"), index=False)
    e.write_json(output.with_suffix(".json"), {
        "run": directory.relative_to(root).as_posix(), "zero": zero.relative_to(root).as_posix(),
        "audits": audits, "prediction_audits": prediction_audits, "validation_choices_recomputed": True,
        "selected_search_seed_matches_policy_forecasts": True,
        "macro": final_macro.to_dict(orient="index"), "policy_macro_train42": policy_macro.to_dict(orient="index"),
        "policy_contrasts": contrasts, "training_seed_scores": seeds.to_dict(orient="index"),
        "improvement_pct": {key: float(gains[key]) for key in ("median_rmse", "median_mae", "mean_rmse", "crps")},
        "epoch_zero_models": epoch_zero, "total_models": len(manifest["folds"]), "policy_epoch_zero_counts": policy_zero,
        "choices": decisions["choices"], "folds": [{k: f[k] for k in ("test_battery", "train_seed", "selected_policy", "selected_epoch")} for f in manifest["folds"]],
        "per_battery": per_battery.reset_index().to_dict(orient="records"),
        "limitations": ["Previously observed four-battery benchmark", "Fixed LR3e-5/KL0.5 anchor; prior choices used mixed KL",
                        "No statistical significance claim", "Search seed 42 only; repeated training seeds are not independent searches"]})
    print(final_macro.to_string(), flush=True)
    print(output, flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/lora_robust_report.md"))
    args = parser.parse_args()
    summarize(e.resolve_reference(ROOT, args.run), args.output)
