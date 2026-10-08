"""Causal cross-learning few-shot: isolated query + one/two other XJTU cells."""
import argparse
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

from src.battery_fewshot import history_bank, support_index, isolated_inputs, predict_isolated_group
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, write_json, sha256, WQL_QUANTILES
from src.battery_model_zeroshot import load_model, phase_labels, PHASES, SCORES, quantile_scores
from src.nasa_covariates import FEATURES
from src.xjtu_covariates import extract_xjtu_features
from src.soh_evaluation import window_r2_values, R2_DEFINITION
from src.soh_experiment import clear_device_cache, resolve_device

CONDITIONS = [(f"{mode}/{shots}-shot", mode, shots)
              for shots in [0, 1, 2] for mode in ["SOH", "SOH+V+I"]]


def plot_comparison(directory, prefix, group, records, data, predictions):
    fig, axes = plt.subplots(3, 3, figsize=(15, 9.6), squeeze=False)
    for row, record in enumerate(records):
        b, w = record["battery_id"], record["windows"]
        cycles = data[(data.battery_id == b) & data.is_observed].cycle.to_numpy()
        labels = phase_labels(w["forecast_start"], cycles)
        centers = cycles.min() + (np.arange(3) + .5) * (cycles.max() - cycles.min() + 1) / 3
        for col, (phase, center) in enumerate(zip(PHASES, centers)):
            ax = axes[row, col]
            candidates = np.flatnonzero(labels == phase)
            if not len(candidates):
                ax.text(.5, .5, f"No paired {phase} window", ha="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            i = candidates[np.abs(w["forecast_start"][candidates] - center).argmin()]
            future = w["target_cycle"][i]
            ax.plot(w["context_cycle"][i], w["context"][i], color=".5", lw=1.2, label="Observed context")
            ax.plot(future, w["target"][i], color="black", lw=1.5, label="Actual SOH")
            text = []
            for condition, color in [("SOH+V+I/0-shot", "#2878b5"), ("SOH+V+I/2-shot", "#d55e00")]:
                q = predictions[(b, condition)]
                ax.plot(future, q[i, :, 4], color=color, lw=1.5, label=condition)
                ax.fill_between(future, q[i, :, 0], q[i, :, -1], color=color, alpha=.12)
                score = quantile_scores(w["target"][i:i+1], q[i:i+1, :, 4])
                text.append(f"{condition}: RMSE {score['rmse']:.4f}, R2 {score['r2']:.2f}")
            ax.text(.02, .03, "\n".join(text), transform=ax.transAxes, fontsize=7)
            ax.axvline(future[0], color=".6", ls=":", lw=1)
            ax.set(title=f"{b} | {phase.title()} | {future[0]}-{future[-1]}", xlabel="Cycle", ylabel="SOH")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Chronos-2 {group} | causal cross-learning | 64 -> 16\n"
                 "2-shot: two separate support cells, past cycles only; one query per inference group")
    fig.tight_layout(rect=(0, 0, 1, .94))
    name = f"{prefix}_forecast_windows.png"
    fig.savefig(directory / name, dpi=150)
    plt.close(fig)
    return name


def write_report(directory, prefix, group, summary, metrics, state, figure):
    overall = summary[summary.phase == "all"].set_index("condition")
    lines = [f"# Chronos-2 {group}: 누수를 차단한 cross-learning few-shot", "",
             "가중치 고정 in-context few-shot이며 fine-tuning이나 LoRA가 아니다. "
             "1-shot은 참고 배터리 하나의 과거 창, 2-shot은 참고 배터리 두 개의 과거 창을 뜻한다.", "",
             "평가 배터리는 파일 1~3번, 참고 배터리는 같은 충전율의 파일 4·5번이다. 참고 배터리 순서는 고정한다. "
             "참고 창은 cycle 번호 기준으로 평가 창의 예측 시작 이전에 끝나는 최신 유효 64-cycle 구간이다. "
             "이 cutoff는 보수적인 cycle 기준 규칙이며 서로 다른 셀의 실제 취득 달력 시각 동기화를 뜻하지 않는다.", "",
             f"배터리당 최대 {state['windows_per_battery']}개 창을 유효 창 순서에서 균등 간격으로 선택했다. "
             f"공통 평가 창은 {state['evaluated_windows_per_condition']}개다. "
             "zero-shot과 1/2-shot은 모두 이 동일한 창에서 평가하며, 이전 전체 창 점수와 직접 비교하지 않는다. "
             "참고 배터리 두 개의 과거 문맥이 모두 확보되지 않는 창은 모든 조건에서 제외했다.", "",
             "SOH-only와 SOH+전압·전류를 각각 비교한다. 센서 특징은 이전 실험과 같은 방전 첫 600초 시간 가중 평균이다. "
             "과거 SOH/센서만 입력하며 future_covariates는 사용하지 않는다.", "",
             "## 데이터 누수 차단", "",
             "- 각 cross-learning 호출에는 평가 창 하나와 그 창의 참고 입력만 넣는다. 다른 시점의 평가 창을 섞지 않는다.",
             "- 평가 배터리는 참고 배터리 집합에 들어가지 않는다. 모든 입력 cycle은 평가 시작 cycle 미만이다.",
             "- batch_size는 그룹의 target·covariate 개수와 정확히 같아 한 그룹이 분할되지 않는다.",
             "- 참고 창 선택은 예측 시작 cycle과 관측 유효성만 사용한다. 미래 SOH나 예측 오차로 선택하지 않는다.",
             "- 테스트에서 참고 미래 SOH/센서를 바꿔도 실제 입력이 동일함을 확인했다. 실행 중 같은 입력 변형 검증을 한 번 더 수행했다.",
             "- 선택한 zero-shot 캐시와 target/시작 cycle을 대조하고, cross_learning=False의 실제 추론도 대조했다.", "",
             "## 동일한 창의 결과", "",
             "| 입력 | RMSE ↓ | 평균 window R² ↑ | WQL ↓ | PICP@80 | 동일 입력 0-shot 대비 RMSE |",
             "|---|---:|---:|---:|---:|---:|"]
    for condition, row in overall.iterrows():
        mode = condition.split("/")[0]
        baseline = overall.loc[f"{mode}/0-shot"]
        lines.append(f"| {condition} | {row.rmse:.6f} | {row.r2:.4f} | {row.wql:.6f} | {row.picp80:.1%} | "
                     f"{(row.rmse/baseline.rmse-1)*100:+.2f}% |")
    lines += ["", "## 전압·전류 입력의 배터리별 결과", "",
              "| 배터리 | 창 수 | 0-shot RMSE | 1-shot RMSE | 2-shot RMSE | 2-shot 변화 |",
              "|---|---:|---:|---:|---:|---:|"]
    for b, table in metrics[metrics.phase == "all"].groupby("battery_id", sort=False):
        table = table.set_index("condition")
        a, one, two = [table.loc[f"SOH+V+I/{n}-shot"] for n in [0, 1, 2]]
        lines.append(f"| {b} | {int(a.n_windows)} | {a.rmse:.6f} | {one.rmse:.6f} | {two.rmse:.6f} | {(two.rmse/a.rmse-1)*100:+.2f}% |")
    lines += ["", "R²는 각 16-cycle 예측창에서 계산 후 배터리 내 유효 창 단순 평균, 평가 배터리 동일 가중 평균이다. "
              "구간은 관측 cycle 범위의 1/3, 예측 시작 cycle로 구분한다. "
              "모든 점수는 샘플 창 기준이며 전체 배터리·전체 창 결과가 아니다. "
              "참고 입력의 예측값은 평가하지 않는다. 미래 target은 점수 계산에만 사용한다. "
              "shot 수에 따른 개선과 센서의 효과는 구분하며, 최선의 결과를 고른 뒤 전체 성능으로 보고하지 않는다.", "",
              f"![동일 창 비교]({figure})", "",
              f"[구간별 점수]({prefix}_summary.csv) · [참고 입력 cycle 기록]({prefix}_support_selection.csv)"]
    (directory / f"{prefix}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_future_invariance(record, index, all_data, all_features, support_ids, names):
    """Perturb unavailable future measurements and rebuild the actual model inputs."""
    b, w = record["battery_id"], record["windows"]
    start = int(w["forecast_start"][index])
    args = (b, w["context"][index], w["context_cycle"][index], start)
    qf = all_features[all_features.battery_id == b]
    banks = [history_bank(all_data[all_data.battery_id == sid], all_features[all_features.battery_id == sid]) for sid in support_ids]
    before, _ = isolated_inputs(*args, qf, banks, names)
    changed_data, changed_features = all_data.copy(), all_features.copy()
    changed_data.loc[changed_data.cycle >= start, "soh"] = 999.
    changed_features.loc[changed_features.cycle >= start, list(FEATURES)] = -999.
    changed_banks = [history_bank(changed_data[changed_data.battery_id == sid], changed_features[changed_features.battery_id == sid]) for sid in support_ids]
    after, _ = isolated_inputs(*args, changed_features[changed_features.battery_id == b], changed_banks, names)
    for a, c in zip(before, after):
        np.testing.assert_array_equal(a["target"], c["target"])
        for name in names:
            np.testing.assert_array_equal(a["past_covariates"][name], c["past_covariates"][name])


def run_group(group, pipeline, limit):
    old_dir = ROOT / f"results/c2-battery/covariates/{group}"
    old_prefix = f"c2-battery_{group}_vi"
    old = json.loads((old_dir / f"{old_prefix}_run.json").read_text(encoding="utf-8"))
    original = json.loads((ROOT / f"results/c2-battery/full/{group}/c2-battery_run.json").read_text(encoding="utf-8"))
    config = replace(ZeroShotConfig(**old["config"]), max_windows_per_battery=None)
    records, _, data, _ = prepare_battery_windows(original["bundle"], config)
    if original["bundle_fingerprint"] != old["bundle_fingerprint"]:
        raise ValueError("Dataset changed since saved zero-shot evaluation")
    batch, rate = ("batch1", "2C") if group == "xjtu_2c" else ("batch2", "3C")
    support_ids = [f"{batch}_{rate}_battery-{i}" for i in [4, 5]]
    if set(support_ids) & set(config.battery_ids):
        raise ValueError("Evaluation and support sets overlap")
    directory = ROOT / f"results/c2-battery/fewshot/{group}"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"c2-battery_{group}_fewshot"
    state = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
             "group": group, "query_battery_ids": list(config.battery_ids), "support_battery_ids": support_ids,
             "windows_per_battery": limit, "selection": "Uniform eligible-window indices; both support contexts must end before query origin",
             "scope": "Sampled paired windows of three selected query batteries", "r2_scope": "sampled_windows",
             "r2": R2_DEFINITION, "conditions": [c[0] for c in CONDITIONS], "device": resolve_device(config.device),
             "model_id": config.model_id, "revision": config.revision, "context_length": 64, "prediction_length": 16,
             "cross_learning": "One query per call; one/two disjoint support cells; batch_size equals total variates",
             "cutoff": "Every input cycle < query forecast_start; no future covariates; no inference-time query mixing",
             "support_time_semantics": "Conservative cycle-index cutoff, not a claim of calendar-time synchronization",
             "bundle_fingerprint": original["bundle_fingerprint"], "zero_shot_source": str(old_dir),
             "support_raw_sha256": {sid: sha256(ROOT / "data/raw/XJTU" / (sid.split("_", 1)[1] + ".mat")) for sid in support_ids},
             "completed_queries": 0}
    state_path = directory / f"{prefix}_run.json"
    write_json(state_path, state)
    try:
        # Cached query features are the exact features used in the zero-shot comparison.
        query_features = pd.read_csv(old_dir / f"{old_prefix}_cycle_features.csv")
        support_features = extract_xjtu_features(ROOT / "data/raw/XJTU", data[data.battery_id.isin(support_ids)])
        support_features.to_csv(directory / f"{prefix}_support_features.csv", index=False)
        features = pd.concat([query_features, support_features], ignore_index=True)
        banks = [history_bank(data[data.battery_id == sid], support_features[support_features.battery_id == sid]) for sid in support_ids]
        predictions, saved, selections, coverage, score_rows, window_rows = {}, {}, [], [], [], []
        sampled_records = []
        with np.load(old_dir / f"{old_prefix}_predictions.npz", allow_pickle=False) as cache:
            for record in records:
                b, full = record["battery_id"], record["windows"]
                np.testing.assert_array_equal(cache[f"{b}_forecast_start"], full["forecast_start"])
                np.testing.assert_allclose(cache[f"{b}_target"], full["target"], rtol=1e-12, atol=1e-12)
                ready = []
                for i, start in enumerate(full["forecast_start"]):
                    try:
                        for bank in banks: support_index(bank, int(start))
                        ready.append(i)
                    except ValueError:
                        pass
                ready = np.asarray(ready, dtype=int)
                if not len(ready):
                    raise ValueError(f"No paired windows with both references: {b}")
                indices = ready[np.linspace(0, len(ready)-1, min(limit, len(ready)), dtype=int)]
                w = {key: value[indices].copy() for key, value in full.items()}
                selected = {**record, "windows": w}
                sampled_records.append(selected)
                coverage.append({"battery_id": b, "original_windows": len(full["target"]), "paired_eligible_windows": len(ready),
                                 "unavailable_support_windows": len(full["target"])-len(ready), "sampled_windows": len(indices)})
                qf = query_features[query_features.battery_id == b]
                labels = phase_labels(w["forecast_start"], data[(data.battery_id == b) & data.is_observed].cycle.to_numpy())
                saved[f"{b}_target"], saved[f"{b}_forecast_start"] = w["target"], w["forecast_start"]
                result = {"SOH/0-shot": cache[f"{b}_condition0_quantiles"][indices].copy(),
                          "SOH+V+I/0-shot": cache[f"{b}_condition3_quantiles"][indices].copy()}
                for mode in ["SOH", "SOH+V+I"]:
                    names = () if mode == "SOH" else FEATURES
                    for shots in [1, 2]:
                        condition = f"{mode}/{shots}-shot"
                        values = []
                        for i, start in enumerate(w["forecast_start"]):
                            inputs, selection = isolated_inputs(b, w["context"][i], w["context_cycle"][i], int(start), qf, banks[:shots], names)
                            with torch.inference_mode():
                                _, q = predict_isolated_group(pipeline, inputs, config)
                            values.append(q)
                            if mode == "SOH+V+I" and shots == 2:
                                selections.extend(selection)
                                state["completed_queries"] += 1
                            if i % 10 == 0:
                                print(f"RUN {group} {b} {condition}: {i+1}/{len(indices)}", flush=True)
                                write_json(state_path, state)
                        result[condition] = np.stack(values)
                if not state.get("future_invariance_verified"):
                    verify_future_invariance(selected, 0, data, features, support_ids, FEATURES)
                    inputs, _ = isolated_inputs(b, w["context"][0], w["context_cycle"][0], int(w["forecast_start"][0]), qf, banks, FEATURES)
                    with torch.inference_mode():
                        unchanged, _ = pipeline.predict_quantiles(inputs=inputs, prediction_length=16, context_length=64,
                            quantile_levels=list(WQL_QUANTILES), batch_size=9, cross_learning=False)
                    np.testing.assert_allclose(unchanged[0].cpu().numpy()[0], result["SOH+V+I/0-shot"][0], rtol=2e-5, atol=1e-7)
                    state.update(future_invariance_verified=True, isolated_zero_shot_verified=True)
                for condition_index, (condition, mode, shots) in enumerate(CONDITIONS):
                    q = result[condition]
                    predictions[(b, condition)] = q
                    saved[f"{b}_condition{condition_index}_quantiles"] = q
                    point = q[..., 4]
                    for phase in ["all", *PHASES]:
                        mask = np.ones(len(labels), dtype=bool) if phase == "all" else labels == phase
                        if mask.any():
                            score_rows.append({"battery_id": b, "condition": condition, "input_mode": mode, "shots": shots,
                                               "phase": phase, "n_windows": int(mask.sum()), "r2_scope": "sampled_windows",
                                               **quantile_scores(w["target"][mask], point[mask], q[mask])})
                    for start, phase, r2, rmse in zip(w["forecast_start"], labels, window_r2_values(w["target"], point),
                                                    np.sqrt(np.square(w["target"]-point).mean(axis=1))):
                        window_rows.append({"battery_id": b, "condition": condition, "forecast_start": int(start),
                                            "phase": phase, "rmse": float(rmse), "r2": float(r2)})
                write_json(state_path, state)
        metrics, windows = pd.DataFrame(score_rows), pd.DataFrame(window_rows)
        summary = metrics.groupby(["condition", "phase"], sort=False, as_index=False)[SCORES].mean()
        counts = metrics.groupby(["condition", "phase"], sort=False, as_index=False).agg(
            n_batteries=("battery_id", "size"), n_windows=("n_windows", "sum"))
        summary = summary.merge(counts, on=["condition", "phase"], validate="one_to_one")
        summary["r2_scope"] = "sampled_windows"
        for row in metrics.to_dict("records"):
            mask = windows.battery_id.eq(row["battery_id"]) & windows.condition.eq(row["condition"])
            if row["phase"] != "all": mask &= windows.phase.eq(row["phase"])
            np.testing.assert_allclose(row["r2"], windows.loc[mask, "r2"].mean(), rtol=1e-10, atol=1e-10)
        selection = pd.DataFrame(selections)
        if not ((selection.support_context_end < selection.forecast_start).all()
                and (selection.query_context_end < selection.forecast_start).all()
                and set(selection.support_battery_id).isdisjoint(set(selection.query_battery_id))):
            raise ValueError("Leakage in support selection log")
        for name, table in [("summary", summary), ("battery_metrics", metrics), ("window_metrics", windows),
                            ("support_selection", selection), ("coverage", pd.DataFrame(coverage))]:
            table.to_csv(directory / f"{prefix}_{name}.csv", index=False)
        np.savez_compressed(directory / f"{prefix}_predictions.npz", **saved)
        figure = plot_comparison(directory, prefix, group, sampled_records, data, predictions)
        state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(), figures=[figure],
                     evaluated_windows_per_condition=sum(len(r["windows"]["target"]) for r in sampled_records),
                     no_cross_origin_query_batching=True)
        write_report(directory, prefix, group, summary, metrics, state, figure)
        write_json(state_path, state)
        print(summary[summary.phase == "all"].to_string(index=False), flush=True)
        return summary.assign(group=group)
    except Exception as error:
        state.update(status="failed", error=str(error))
        write_json(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-per-battery", type=int, default=30)
    args = parser.parse_args()
    if args.windows_per_battery < 3:
        parser.error("Use at least three windows per battery")
    config = ZeroShotConfig(**json.loads((ROOT / "results/c2-battery/full/xjtu_2c/c2-battery_run.json").read_text(encoding="utf-8"))["config"])
    print("Loading local Chronos-2; frozen weights, causal 1/2-shot groups", flush=True)
    pipeline = load_model(config)
    versions = tuple(parameter._version for parameter in pipeline.model.parameters())
    try:
        summaries = [run_group(group, pipeline, args.windows_per_battery) for group in ["xjtu_2c", "xjtu_3c"]]
        if versions != tuple(parameter._version for parameter in pipeline.model.parameters()):
            raise ValueError("Model parameters changed during frozen inference")
        for group in ["xjtu_2c", "xjtu_3c"]:
            state_path = ROOT / f"results/c2-battery/fewshot/{group}/c2-battery_{group}_fewshot_run.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["model_parameters_unchanged"] = True
            write_json(state_path, state)
        output = ROOT / "results/c2-battery/fewshot"
        pd.concat(summaries, ignore_index=True).to_csv(output / "c2-battery_fewshot_summary.csv", index=False)
        print("COMPLETE causal cross-learning few-shot; model parameters unchanged", flush=True)
    finally:
        del pipeline
        clear_device_cache()


if __name__ == "__main__":
    main()
