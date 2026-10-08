"""Summarize retained experiments without training or changing original results.

R2 distribution weights match the existing equal-battery / equal-seed mean.
Missing window-level predictions remain missing; aggregate RMSE cannot restore them.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.soh_evaluation import window_r2_values
from src.battery_model_zeroshot import quantile_scores

OUT = ROOT / "results/r2-distribution"
GROUPS = ["nasa_aging", "calce_cs2_1c", "xjtu_2c", "xjtu_3c", "mit_batch1", "mit_batch2", "mit_batch3"]
ROWS = []


def distribution(frame):
    """Weighted median, not mean of battery medians; preserve existing macro mean."""
    valid = frame[np.isfinite(frame.r2)].copy()
    if valid.empty:
        return {"r2_mean": np.nan, "r2_median": np.nan, "r2_negative_fraction": np.nan,
                "r2_min": np.nan, "r2_n_windows": 0, "r2_undefined_windows": len(frame)}
    if "seed" not in valid:
        valid["seed"] = 42
    batteries = valid.battery_id.nunique()
    seed_counts = valid.groupby("battery_id").seed.transform("nunique")
    window_counts = valid.groupby(["battery_id", "seed"]).r2.transform("size")
    weights = (1 / (batteries * seed_counts * window_counts)).to_numpy()
    values = valid.r2.to_numpy()
    np.testing.assert_allclose(weights.sum(), 1)
    order = np.argsort(values, kind="stable")
    ordered, cumulative = values[order], np.cumsum(weights[order])
    i = int(np.searchsorted(cumulative, .5 - 1e-12))
    median = (ordered[i] + ordered[i + 1]) / 2 if i + 1 < len(ordered) and abs(cumulative[i] - .5) < 1e-12 else ordered[i]
    return dict(r2_mean=float(values @ weights), r2_median=float(median),
                r2_negative_fraction=float(weights[values < 0].sum() / weights.sum()), r2_min=float(values.min()),
                r2_n_windows=len(valid), r2_undefined_windows=len(frame) - len(valid))


def add(category, dataset, condition, metrics, frame=None, phase="all", scope="all_windows", source="", **extra):
    row = dict(category=category, dataset=dataset, condition=condition, phase=phase, scope=scope,
               rmse=metrics.get("rmse", np.nan), wql=metrics.get("wql", np.nan), picp80=metrics.get("picp80", np.nan),
               r2_mean=metrics.get("r2", np.nan), r2_median=np.nan, r2_negative_fraction=np.nan,
               r2_min=np.nan, r2_n_windows=np.nan, r2_undefined_windows=np.nan,
               source=str(source), **extra)
    if frame is not None:
        scores = distribution(frame)
        if np.isfinite(row["r2_mean"]):
            np.testing.assert_allclose(scores["r2_mean"], row["r2_mean"], rtol=1e-8, atol=1e-9,
                                       err_msg=f"Stored mean mismatch: {dataset}/{condition}/{phase}")
        row.update(scores)
        row["n_batteries"] = frame.battery_id.nunique()
    ROWS.append(row)


def c2_experiments():
    for family, category in [("covariates", "C2_covariates"), ("fewshot", "C2_adaptation"),
                             ("fewshot-tuning", "C2_adaptation"), ("fewshot-head-tuning", "C2_adaptation")]:
        for path in sorted((ROOT / "results/c2-battery" / family).glob("*/*_window_metrics.csv")):
            windows = pd.read_csv(path)
            summary = pd.read_csv(path.with_name(path.name.replace("window_metrics", "summary")))
            for score in summary.to_dict("records"):
                condition, phase = score["condition"], score["phase"]
                if family in ["fewshot-tuning", "fewshot-head-tuning"] and condition.endswith("zero-shot"):
                    continue  # Same paired baseline is already in the cross-learning experiment.
                selected = windows[windows.condition == condition]
                if phase != "all":
                    selected = selected[selected.phase == phase]
                add(category, path.parent.name, condition, score, selected, phase,
                    "selected_cells_all_windows" if family == "covariates" else "paired_90_windows",
                    path.relative_to(ROOT), context=64, horizon=16)


def sample_forecasts(files):
    windows, scores = [], []
    for battery, seed, path in files:
        with np.load(path, allow_pickle=False) as cache:
            samples = cache["samples"].astype(np.float64)
            y = cache["target"].astype(np.float64)
            q = np.quantile(samples, np.arange(1, 10) / 10, axis=1).transpose(1, 2, 0)
            values = window_r2_values(y, q[..., 4])
            windows.extend(dict(battery_id=battery, seed=seed, forecast_start=int(start), r2=value)
                           for start, value in zip(cache["forecast_start"], values))
            scores.append(dict(battery_id=battery, seed=seed, **quantile_scores(y, q[..., 4], q)))
    score_frame = pd.DataFrame(scores)
    macro = score_frame.groupby("battery_id")[["rmse", "r2", "wql", "picp80"]].mean().mean()
    return pd.DataFrame(windows), macro.to_dict()


def t5_adaptation():
    runs = [("20260923T063348Z_zeroshot_b57856", "Zero-shot (3 seeds)"),
            ("20260923T063628Z_lora_2df6cb", "Original LoRA"),
            ("20260929T175636Z_checkpointlora_39b551", "LoRA: EOS masked + best checkpoint + LR selection")]
    for run, label in runs:
        directory = ROOT / "results/runs" / run / "predictions"
        files = [(p.stem.split("_")[0], int(p.stem.rsplit("infer", 1)[1]), p) for p in sorted(directory.glob("*.npz"))]
        frame, scores = sample_forecasts(files)
        add("T5_NASA_adaptation", "nasa_aging", label, scores, frame, source=directory.relative_to(ROOT), context=64, horizon=16)
    base = ROOT / "results/diagnostics/20260929"
    files = []
    for b in ["B0005", "B0006", "B0007", "B0018"]:
        sub = "training" if b == "B0005" else "confirm_eos"
        files.append((b, 42, base / sub / f"{b}_without_eos/final.npz"))
        files += [(b, seed, base / "seed_confirmation" / f"{b}_without_eos_infer{seed}.npz") for seed in [43, 44]]
    frame, scores = sample_forecasts(files)
    add("T5_NASA_adaptation", "nasa_aging", "LoRA: EOS masked (old epoch/refit)", scores, frame,
        source=base.relative_to(ROOT), context=64, horizon=16)
    for family in ["training", "inference", "float32", "confirm_eos"]:
        for path in sorted((base / family).rglob("*.npz")):
            battery = path.parent.name.split("_")[0] if family in ["training", "confirm_eos"] else path.stem.split("_")[0]
            frame, scores = sample_forecasts([(battery, 42, path)])
            label = str(path.relative_to(base)).replace("\\", "/").removesuffix(".npz")
            add("T5_diagnostic_single_cell", battery, label, scores, frame,
                scope="single_cell_seed42", source=path.relative_to(ROOT), context=64, horizon=16)


def zeroshot():
    for tag in ["t5", "c2"]:
        for group in GROUPS + ["calce_cs2_1c_legacy32"]:
            dataset = group.replace("_legacy32", "")
            if dataset == "calce_cs2_1c" and not group.endswith("legacy32"):
                directory = ROOT / f"results/{tag}-battery/calce64-16/{dataset}"
                prefix = f"{tag}-battery_"
            else:
                directory = ROOT / f"results/{tag}-battery" / (f"full/{dataset}" if tag == "c2" else dataset)
                prefix = "c2-battery_" if tag == "c2" else ""
            manifest = json.loads((directory / f"{prefix}run.json").read_text(encoding="utf-8"))
            summary = pd.read_csv(directory / f"{prefix}summary.csv")
            summary = summary[summary.model.isin(["Chronos", "t5-battery", "c2-battery"])]
            representative = directory / "representative_r2_windows.csv"
            # A later, explicitly chosen re-inference scope can supply per-window data here.
            recovered = OUT / "windows" / f"{tag}_{group}.csv.gz"
            frame = pd.read_csv(recovered) if recovered.exists() else None
            scope = "all_windows; window_values_not_retained"
            if frame is not None:
                info = json.loads(recovered.with_suffix(".json").read_text(encoding="utf-8"))
                if info["status"] != "complete":
                    raise ValueError("Recovered window distribution is incomplete")
                scope = info["scope"]
            elif representative.exists():
                frame = pd.read_csv(representative)
                frame = frame[frame.model == "Chronos"]
                scope = "representative_windows_R2; other_metrics_all_windows"
            original_summary = summary.copy()
            if recovered.exists() and frame is not None:
                summary = pd.DataFrame(info["summary"])
            for score in summary.to_dict("records"):
                phase = score.get("phase", "all")
                selected = frame
                if frame is not None and phase != "all":
                    selected = frame[frame.phase == phase]
                original_phase = original_summary if "phase" not in original_summary else original_summary[original_summary.phase == phase]
                old = original_phase.iloc[0] if len(original_phase) and (phase == "all" or "phase" in original_summary) else {}
                original_r2 = old.get("r2", np.nan)
                add("Zero-shot_legacy32" if group.endswith("legacy32") else "Zero-shot", dataset, tag.upper(), score,
                    selected, phase=phase, scope=scope, source=directory.relative_to(ROOT),
                    context=manifest["config"]["context_length"], horizon=manifest["config"]["prediction_length"],
                    n_batteries=manifest.get("batteries_evaluated"), original_r2_mean=original_r2,
                    original_r2_scope=manifest.get("r2_scope", "all_windows"),
                    original_rmse=old.get("rmse", np.nan), original_wql=old.get("wql", np.nan), original_picp80=old.get("picp80", np.nan),
                    original_evaluated_windows=manifest.get("evaluated_windows"))


def write_report(frame):
    lines = ["# T5·C2 zero-shot 및 튜닝: 평균·중앙값 R² 비교", "",
             "C2 zero-shot은 전체 창, T5 zero-shot은 관측 기간의 초·중·후반 중심에 가장 가까운 대표 창으로 계산했다. "
             "해당 구간에 유효 창이 없으면 그 구간은 제외한다. T5 표의 RMSE·WQL·PICP도 동일한 대표 창 기준이다. "
             "기존 전체 평가값은 CSV의 original_* 열에 보존했다. 두 모델의 zero-shot 평가 창 수가 다르므로 동일 창의 직접 비교는 아니다.", "",
             "- C2 MIT batch 2는 평균 R²가 음수여도 중앙값이 0.828, 음수 비율이 12.9%다. 일부 큰 실패의 영향이 크다.",
             "- C2 NASA와 CALCE는 중앙값도 음수다. 이 경우는 평균의 극단값 민감성만으로 설명할 수 없다.",
             "- 3C 소량 데이터 튜닝에서 SOH+V+I full tuning은 중앙값 0.751, 음수 창 18.9%; head는 0.699, 25.6%다.", "",
             "R²는 개별 예측창에서 1−SSE/SST로 계산한다. 점예측은 forecast median(q0.5)으로 통일했다.", "",
             "- 평균: 기존처럼 배터리 내 창·seed 평균 후 배터리 동일 가중 평균.",
             "- 중앙값/음수 비율: 평균과 같은 가중치(배터리 동일, 배터리 내 seed 동일, seed 내 유효 창 동일)를 "
             "사용한 창별 R² 분포에서 계산. 배터리별 중앙값을 평균한 값이 아니다.",
             "- 각 배터리의 창 수가 같은 90창 비교에서는 일반적인 전체 창 중앙값·음수 창 비율과 같다.",
             "- 음수는 자르지 않고, 상수 target 창의 미정의 R²만 제외한다. 모델·checkpoint 선택은 바꾸지 않는다.",
             "- 기존 median_r2 열은 중앙값 예측기의 평균 R²다. 이번 r2_median은 창별 R²의 분포 중앙값이다.",
             "- 저장된 평균만 있는 경우 중앙값·음수 비율은 N/A. 대표 창 R²는 전체 창 평가와 구분한다.",
             "- Head tuning의 PICP N/A는 예측구간 역전 때문이며, R² 누락과 다른 사유다.", "",
             "전체 지표와 초·중·후반 결과: [CSV](all_experiments.csv). 아래는 phase=all 결과다.", ""]
    captions = {
        "Zero-shot": ("다중 데이터셋 zero-shot", "모두 64→16. C2 전체 창, T5 대표 창. CALCE는 파일 경계를 연결하는 현재 64→16 설정이다."),
        "Zero-shot_legacy32": ("이전 CALCE 설정", "32→8 보관 결과이며 현재 64→16과 분리해서 읽는다."),
        "T5_NASA_adaptation": ("NASA T5 LoRA", "동일한 4개 배터리의 전체 320창 × 추론 seed 3개 = 960개 창별 예측. "
                               "old epoch/refit과 best checkpoint는 학습 데이터·선택 방식이 달라 변경 하나의 인과 효과로 해석하지 않는다."),
        "T5_diagnostic_single_cell": ("T5 단일 배터리 분리 진단", "과거 EOS·학습량·학습률·추론 경로 진단. 아래 창 수는 각 진단의 실제 평가 범위다."),
        "C2_covariates": ("C2 과거 전압·전류 추가 zero-shot", "NASA 4개 배터리의 320창; XJTU는 선택한 1·2·3번 배터리 전체 창(2C 950, 3C 637). "
                          "가중치는 학습하지 않았다. 아래 소량 튜닝의 90창과 평가 범위가 다르다."),
        "C2_adaptation": ("C2 cross-learning 및 소량 데이터 튜닝", "각 그룹에서 동일한 평가 배터리 1·2·3번의 각 30창, 총 90창. "
                          "1-shot/2-shot은 추론 시 참조 배터리를 제공하는 cross-learning으로 가중치를 갱신하지 않는다. "
                          "full/head tuning은 배터리 4·5번의 20개 학습창과 6번의 검증창을 사용한다. "
                          "full learning rate 1e-6, head 1e-4, 각각 100 step이므로 방식 자체의 최적 성능 비교는 아니다.")}
    for category, group in frame[frame.phase == "all"].groupby("category", sort=False):
        title, description = captions[category]
        lines += [f"## {title}", "", description, "", "| 데이터 | 조건 | 창×seed 수 | RMSE | R² 평균 | R² 중앙값 | R²<0 | WQL | PICP80 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in group.itertuples():
            def f(value, digits=3): return f"{value:.{digits}f}" if pd.notna(value) else "N/A"
            def pct(value): return f"{value:.1%}" if pd.notna(value) else "N/A"
            lines.append(f"| {row.dataset} | {row.condition} | {f(row.r2_n_windows,0)} | {f(row.rmse,6)} | {f(row.r2_mean)} | {f(row.r2_median)} | "
                         f"{pct(row.r2_negative_fraction)} | {f(row.wql,6)} | {pct(row.picp80)} |")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ROWS.clear()
    zeroshot()
    t5_adaptation()
    c2_experiments()
    frame = pd.DataFrame(ROWS)
    frame.to_csv(OUT / "all_experiments.csv", index=False)
    write_report(frame)
    print(frame[frame.phase == "all"][["category", "dataset", "condition", "r2_mean", "r2_median", "r2_negative_fraction"]].to_string(index=False))
    print(f"Saved {len(frame)} rows to {OUT}")


if __name__ == "__main__":
    main()
