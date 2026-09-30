"""Recalculate reporting metrics without retraining or altering signed run files."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_evaluation import export_run, export_diagnosis, r2_score, _r2_from_rmse, R2_DEFINITION
from src.soh_experiment import load_config, load_soh, make_windows, _baselines, point_metrics


def export_legacy():
    source = ROOT / "results/metrics/chronos_lobo_results.csv"
    frame = pd.read_csv(source)
    cfg = load_config(ROOT)
    series = load_soh(ROOT, cfg)
    batteries = frame[frame.Battery != "Macro"].copy()
    with np.load(source.parent / "chronos_zeroshot_predictions.npz", allow_pickle=False) as zero:
        for name in ("Last", "Linear", "ZeroShot", "LoRA"):
            values = []
            for row in batteries.itertuples(index=False):
                windows = make_windows(series[row.Battery], cfg.context_length, cfg.prediction_length)
                if name == "LoRA":
                    # The old CSV rounds RMSE to six decimals; no LoRA predictions survive.
                    values.append(_r2_from_rmse(windows["target"], row.LoRA_RMSE))
                else:
                    pred = zero[row.Battery] if name == "ZeroShot" else _baselines(windows)["LastValue" if name == "Last" else "Linear"]
                    np.testing.assert_allclose(point_metrics(windows["target"], pred)["rmse"],
                                               getattr(row, f"{name}_RMSE"), rtol=0, atol=5.1e-7)
                    values.append(r2_score(windows["target"], pred))
            batteries[f"{name}_R2"] = values
    macro = frame[frame.Battery == "Macro"].copy()
    for col in [c for c in batteries if c.endswith("_R2")]:
        macro[col] = batteries[col].mean()
    result = pd.concat([batteries, macro], ignore_index=True)
    result["LoRA_R2_is_approximate"] = True
    out = ROOT / "results/evaluation_r2/metrics"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / source.name, index=False)
    (out / "README.md").write_text(
        "# 이전 20-sample 실험의 R²\n\nLast/Linear/ZeroShot은 원시 예측 또는 원래 baseline 수식에서 계산했다. "
        "LoRA 원시 예측은 보관돼 있지 않아 소수점 6자리 RMSE와 해당 target 분산으로 복원한 **근삿값**이다. "
        "현재 100-sample 실험과 구분한다. Macro는 배터리별 R²의 동일 가중 평균이다.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.run:
        print(export_run(ROOT, args.run))
        return
    summaries = []
    for manifest_path in sorted((ROOT / "results/runs").glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            continue
        out = export_run(ROOT, manifest_path.parent)
        summary = pd.read_csv(out / "summary.csv")
        summary["run_id"] = manifest_path.parent.name
        summaries.append(summary)
        print("R2:", out, flush=True)
    out = export_diagnosis(ROOT)
    print("DIAGNOSIS_R2:", out, flush=True)
    export_legacy()
    destination = ROOT / "results/evaluation_r2"
    pd.concat(summaries).to_csv(destination / "run_summary.csv", index=False)
    (destination / "README.md").write_text(
        "# R²를 포함한 성능평가\n\n`python scripts/add_r2_metrics.py`로 재생성한다. 재학습은 하지 않는다.\n\n"
        + R2_DEFINITION + "\n\n"
        "원본 `results/runs`의 예측·학습 기록·선택·서명된 CSV는 보존하고, 여기에는 R²를 추가한 평가표를 저장한다. "
        "RMSE/MAE도 원시 예측에서 검산한다. 02·03 노트북과 현재 보고서는 이 평가표를 사용한다. "
        "Validation 전체 이력은 seed별 원래 RMSE와 target 분산으로 같은 정의의 R²를 계산한다. "
        "이미 평균한 RMSE에서 macro R²를 역산하지 않는다.\n\n"
        "R²는 1이면 완전 예측, 0이면 평가 target의 전체 평균을 항상 예측한 기준과 같다. "
        "이 평균은 평가 구간의 사후 평균이므로 Last-value baseline과 다르다. 음수도 유효하며 자르지 않는다. "
        "정답이 상수이거나 값이 2개 미만이면 분모가 정의되지 않아 NaN으로 표시한다. "
        "짧고 변화가 작은 window의 R²는 매우 낮아질 수 있다. R² 차이는 퍼센트 개선율로 계산하지 않는다.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
