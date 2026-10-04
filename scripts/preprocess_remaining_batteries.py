"""Process all remaining raw datasets; keep reports to essential checks and SOH curves."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.battery_preprocessing import read_calce, read_mit, read_nasa, save_bundle
from src.soh_data import load_bundle


NOTES = {
    "mit": "Batch 1의 cycle 1은 0인 미측정값으로 보존하고 SOH 기준은 cycle 2 용량으로 명시했다. "
           "batch를 이어 측정한 5개 기록은 원래 배터리의 기준 용량과 physical_battery_id를 공유한다. "
           "train/test 분리 시 이 ID를 함께 묶어야 한다. 제작자가 noisy channel로 표시한 Batch 3의 6개 기록은 "
           "곡선과 데이터에 보존하고 예측 window에서는 제외한다. 기록 140개는 물리 배터리 135개에 해당한다.",
    "calce": "누적 Discharge_Capacity(Ah)의 방전 종료값에서 방전 직전 값을 빼 cycle 용량을 계산했다. "
             "날짜별 파일의 Cycle_Index가 다시 1부터 시작하므로 시간순 관측 번호와 원본 cycle 번호를 함께 보존한다. "
             "완료되지 않은 방전은 표시하고 예측에서 제외하며, window는 파일 경계를 넘지 않는다. "
             "기준은 첫 완료 방전 용량이다. 같은 측정 구간과 용량을 중복 내보낸 관측은 한 번만 사용하고 "
             "원본 파일 이름을 모두 기록한다. 원래 기록에 없는 cycle은 보간하지 않는다.",
    "nasa": "discharge 기록의 data.Capacity를 사용하고 첫 방전 용량을 기준으로 SOH를 계산했다. "
            "원본 MAT의 전체 기록 위치도 보존한다. 기존 NASA 02·03 실험과 같은 Q/Q1 기준이다.",
}


def report(group, data, batteries, audit, manifest, output, bundle):
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / f"{group}_audit.csv", index=False, float_format="%.17g")
    compact = [c for c in ["battery_id", "batch", "physical_battery_id", "continuation_of", "reference_method",
                          "reference_cycle", "reference_capacity_ah", "source_charge_policy", "known_collection_issue"] if c in batteries]
    batteries[compact].to_csv(output / f"{group}_batteries.csv", index=False, float_format="%.17g")
    groups = list(data.groupby("battery_id", sort=False))
    images = []
    for page, start in enumerate(range(0, len(groups), 24), 1):
        subset = groups[start:start + 24]
        columns = min(4, len(subset))
        rows = math.ceil(len(subset) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(15, 3 * rows), squeeze=False)
        for ax, (battery_id, values) in zip(axes.flat, subset):
            numeric = np.isfinite(values.soh) & values.is_observed
            usable = numeric & values.capacity_kind.eq("full_discharge") & (values.capacity_ah > 0)
            meta = batteries[batteries.battery_id == battery_id].iloc[0]
            noisy = bool(meta.get("known_collection_issue", False))
            title = battery_id + (" *" if noisy else "")
            if meta.get("continuation_of", ""):
                title += f" ({meta.continuation_of})"
            curve = values.soh.where(usable)
            # CALCE acquisition restarts remain visible as gaps in the line.
            for _, segment in values.assign(plot_soh=curve).groupby("segment_id", sort=False):
                ax.plot(segment.cycle, segment.plot_soh, color=".5" if noisy else "#2878b5", linewidth=1.1)
            excluded = numeric & ~usable
            ax.scatter(values.loc[excluded, "cycle"], values.loc[excluded, "soh"], marker="x",
                       color="#c44e52", s=14, zorder=3)
            ax.set(title=title, xlabel="Cycle", ylabel="SOH (Q / Q_ref)")
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(subset):]:
            ax.set_visible(False)
        fig.suptitle(f"{group}: observed capacity curves (page {page})\nNo smoothing or interpolation; * known noisy channel; red x = incomplete/invalid observation")
        fig.tight_layout(rect=(0, 0, 1, .95))
        name = f"{group}_soh_curves_{page:02d}.png"
        fig.savefig(output / name, dpi=160)
        plt.close(fig)
        images.append(name)
    dataset = data.dataset.iloc[0]
    lines = [f"# {group} SOH 추출", "",
             f"기록 **{len(batteries)}개**, cycle **{len(data):,}개**. 필수 확인: **{manifest['status']}**.", "",
             "용량·cycle 배열, 중복·순서, 정규화 계산을 확인했다. 유효하지 않거나 불완전한 관측은 표시해서 "
             "예측 window에서 제외한다. 고유 관측과 원래 capacity는 보존하며 보간·평활화·이상치 자동 제거를 하지 않았다.", "",
             NOTES[dataset], ""]
    for name in images:
        lines += [f"![{group} SOH curves]({name})", ""]
    lines += ["| Battery | 저장 cycle | 예측 제외 | 64→16 windows | 마지막 SOH | 상태 |",
              "|---|---:|---:|---:|---:|---|"]
    for r in audit.to_dict("records"):
        lines.append(f"| {r['battery_id']} | {r['n_cycles']} | {r['n_excluded_measurements']} | "
                     f"{r['n_prediction_windows']} | {r['last_soh']:.4f} | {r['status']} |")
    source = batteries.source_documentation.iloc[0]
    if dataset == "calce":
        lines += ["", f"중복 내보내기 **{manifest['duplicate_cycle_exports_removed']} cycle**을 한 번씩만 기록했다. "
                  "64→16 window 수가 0이면 현재 파일 내 연속 관측 길이가 부족하거나 제외 관측이 포함된 것이다. "
                  "예측 단계에서는 context를 줄이거나 파일 경계의 연속성을 별도로 확인해야 한다."]
    lines += ["", f"전체 공통 데이터: `{bundle.relative_to(ROOT).as_posix()}`.",
              "아직 모델 예측은 실행하지 않았다.", "", f"[데이터 제작자의 설명]({source})", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe scripts\\preprocess_remaining_batteries.py", "```", ""]
    (output / f"{group}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["nasa", "mit", "calce"], default=["nasa", "mit", "calce"])
    args = parser.parse_args()
    readers = {"nasa": (read_nasa, "Nasa"), "mit": (read_mit, "MIT"), "calce": (read_calce, "Calce")}
    failures = []
    for dataset in args.datasets:
        read, folder = readers[dataset]
        try:
            groups = read(ROOT / "data/raw" / folder)
            for group, data, batteries, hashes in groups:
                bundle = ROOT / "data/processed" / folder / group
                audit, manifest = save_bundle(group, data, batteries, hashes, bundle)
                output = ROOT / "results/data_audit" / group
                report(group, data, batteries, audit, manifest, output, bundle)
                print(f"COMPLETE {group}: {len(batteries)} records, {len(data)} cycles", flush=True)
                print("REPORT=" + str(output / f"{group}_report.md"), flush=True)
        except (ValueError, KeyError, OSError, AssertionError) as error:
            failures.append(dataset)
            print(f"FAILED {dataset}: {type(error).__name__}: {error}", flush=True)
    if failures:
        raise SystemExit("Unfinished datasets: " + ", ".join(failures))
    write_overview()


def write_overview():
    locations = [("xjtu_2c", "XJTU/2C"), ("xjtu_3c", "XJTU/3C"),
                 ("nasa_aging", "Nasa/nasa_aging"), ("mit_batch1", "MIT/mit_batch1"),
                 ("mit_batch2", "MIT/mit_batch2"), ("mit_batch3", "MIT/mit_batch3"),
                 ("calce_cs2_1c", "Calce/calce_cs2_1c")]
    lines = ["# 배터리 데이터 가공 현황", "",
             "raw 폴더의 데이터에서 cycle별 용량과 SOH를 공통 포맷으로 추출했다. 검토 결과는 필수 오류 확인과 관측 곡선이다.", "",
             "| 그룹 | 배터리 기록 | 저장 cycle | 64→16 windows | 곡선·요약 |",
             "|---|---:|---:|---:|---|"]
    total = 0
    for group, directory in locations:
        bundle = ROOT / "data/processed" / directory
        if not (bundle / "manifest.json").exists():
            continue
        data, batteries, _ = load_bundle(bundle)
        audit = pd.read_csv(bundle / "audit.csv")
        total += len(data)
        lines.append(f"| {group} | {len(batteries)} | {len(data):,} | {int(audit.n_prediction_windows.sum()):,} | "
                     f"[보기]({group}/{group}_report.md) |")
    lines += ["", f"총 **{total:,}행**을 저장했다. 아직 새로운 데이터셋의 모델 예측은 실행하지 않았다.", "",
              "- MIT 140개 기록은 연속 측정 구간 5개를 포함한 물리 배터리 135개다. 분할 시 `physical_battery_id`를 함께 묶는다.",
              "- CALCE는 중복 내보내기를 제거하고 파일별 측정 구간을 구분했다. 파일당 최대 50 cycle이므로 현재 64→16 window는 없다. 예측 시 context를 줄이거나 파일 간 연속성을 확인해야 한다.",
              "- NASA 공통 포맷은 기존 02·03 실험과 같은 Q/Q1 기준이다.",
              "- 전체 CSV는 `data/processed/<dataset>/<group>/`에 저장하며 Git에서 제외한다. 검토 파일명에는 dataset과 group을 넣었다.", ""]
    if (ROOT / "results/data_audit/outlier_cleaning/soh_outlier_cleaning_report.md").exists():
        lines += ["예측용 급등 이상치 제거본: `data/processed/clean/<group>/`. "
                  "[제거 목록·전후 그림·window 수](outlier_cleaning/soh_outlier_cleaning_report.md). "
                  "위 표는 기존 가공본 기준이며 제거본 수치는 해당 보고서에 기록했다.", ""]
    (ROOT / "results/data_audit/battery_datasets_overview.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
