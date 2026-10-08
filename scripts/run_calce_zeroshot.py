"""Run CALCE 64 -> 16 on all valid observation-order windows; keep old runs."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows
from src.battery_model_zeroshot import run_model_battery
from src.soh_data import write_json


def write_comparison(destination):
    frames, manifests, baseline = [], [], []
    for tag in ["t5-battery", "c2-battery"]:
        directory = ROOT / "results" / tag / "calce64-16/calce_cs2_1c"
        path = directory / f"{tag}_run.json"
        if not path.is_file():
            return
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            return
        manifests.append(manifest)
        scores = pd.read_csv(directory / f"{tag}_battery_metrics.csv")
        baseline.append(scores[scores.model.isin(["LastValue", "Linear"])].sort_values(
            ["battery_id", "phase", "model"]).reset_index(drop=True))
        summary = pd.read_csv(directory / f"{tag}_summary.csv")
        summary.insert(0, "run_model", tag)
        frames.append(summary)
    keys = ["context_length", "prediction_length", "stride", "calce_cross_file_windows",
            "battery_ids", "max_windows_per_battery"]
    left, right = manifests
    if (left["bundle_fingerprint"] != right["bundle_fingerprint"]
            or any(left["config"][k] != right["config"][k] for k in keys)):
        raise ValueError("CALCE T5/C2 comparison has mismatched inputs")
    pd.testing.assert_frame_equal(*baseline, check_dtype=False, rtol=1e-10, atol=1e-12)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(destination / "t5-c2-battery_summary.csv", index=False)
    lines = ["\n## 전체 점예측·예측구간\n",
             "| 모델 | RMSE (%p) | WQL | PICP@80 |",
             "|---|---:|---:|---:|"]
    for tag in ["t5-battery", "c2-battery"]:
        row = combined[(combined.model == tag) & (combined.phase == "all")].iloc[0]
        lines.append(f"| {tag} | {100 * row.rmse:.3f} | {row.wql:.6f} | {row.picp80:.1%} |")
    for name in ["LastValue", "Linear"]:
        row = combined[(combined.model == name) & (combined.phase == "all")].iloc[0]
        lines.append(f"| {name} | {100 * row.rmse:.3f} | — | — |")
    lines += ["\n## 관측 기간 초·중·후반 Mean window R²\n",
              "| 모델 | 초반 | 중반 | 후반 |", "|---|---:|---:|---:|"]
    for tag in ["t5-battery", "c2-battery"]:
        values = [combined[(combined.model == tag) & (combined.phase == phase)].r2.iloc[0]
                  for phase in ["early", "middle", "late"]]
        lines.append(f"| {tag} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    (destination / "README.md").write_text(
        "# CALCE 64→16 zero-shot\n\n"
        "4개 배터리의 시간순 관측, stride 1. 두 모델의 입력·창 설정과 baseline 점수 일치를 확인했다. "
        "각 기록의 관측 기간을 3등분한다. R²는 한 예측창의 미래 실제 평균을 기준으로 계산한 뒤 "
        "해당 구간/전체의 유효 창을 단순평균하고 배터리 점수를 동일 가중 평균한다. 상수 창은 제외하며 음수는 유지한다.\n\n"
        "파일 경계는 연결하지만 누락·불완전 관측은 연결하지 않는다. 달력 시간 휴지는 cycle 입력에 추가하지 않는다. "
        "기존 32→8보다 horizon과 평가 창이 달라 점수 증감을 모델 개선으로 해석하지 않는다.\n\n"
        "- [T5 점수·그래프](../../t5-battery/calce64-16/calce_cs2_1c/t5-battery_report.md)\n"
        "- [C2 점수·그래프](../../c2-battery/calce64-16/calce_cs2_1c/c2-battery_report.md)\n"
        + "\n".join(lines) + "\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["t5", "c2", "both"], default="both")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    bundle = ROOT / "data/processed/clean/calce_cs2_1c"
    families = ["t5", "c2"] if args.model == "both" else [args.model]
    state_root = ROOT / "results/t5-c2-battery-comparison/calce64-16"
    state_root.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "models_completed": [], "models": families}
    if not args.prepare_only:
        write_json(state_root / "status.json", state)
    try:
        for family in families:
            tag = family + "-battery"
            config = ZeroShotConfig(**json.loads((ROOT / "configs" / f"{tag}.json").read_text(encoding="utf-8")))
            config = replace(config, context_length=64, prediction_length=16, stride=1,
                             battery_ids=None, max_windows_per_battery=None, calce_cross_file_windows=True)
            records, coverage, _, _ = prepare_battery_windows(bundle, config)
            print(tag, coverage[["battery_id", "evaluated_windows"]].to_string(index=False), flush=True)
            if len(records) != 4 or (coverage.evaluated_windows == 0).any():
                raise ValueError("All four CALCE batteries must have eligible windows")
            if args.prepare_only:
                continue
            directory = run_model_battery(bundle, config,
                output_root=ROOT / "results" / tag / "calce64-16", scope="full")
            print("REPORT=" + str(directory / f"{tag}_report.md"), flush=True)
            state["models_completed"].append(family)
            write_json(state_root / "status.json", state)
        if not args.prepare_only:
            write_comparison(state_root)
            state["status"] = "complete"
            write_json(state_root / "status.json", state)
            print("CALCE64_16_COMPLETE", flush=True)
    except Exception as error:
        if not args.prepare_only:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(state_root / "status.json", state)
        raise


if __name__ == "__main__":
    main()
