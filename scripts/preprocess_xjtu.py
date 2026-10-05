"""Extract XJTU 2C/3C bundles and write essential summaries and SOH curves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_data import load_bundle
from src.xjtu_preprocessing import preprocess, source_profile


def report(data, audit, manifest, output, bundle):
    rate, _ = source_profile(manifest["selection"])
    group = f"xjtu_{rate}c"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("batteries.csv", "audit.csv"):
        shutil.copy2(bundle / name, output / f"{group}_{name}")
    if len(data):
        groups = list(data.groupby("battery_id", sort=False))
        height = (len(groups) + 3) // 4
        fig, axes = plt.subplots(height, 4, figsize=(15, 3 * height),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, (battery, rows) in zip(axes.flat, groups):
            normal = rows[rows.cycle > 1]
            ax.plot(normal.cycle, normal.soh, color="#2878b5", linewidth=1.2,
                    label=f"{rate}C charge / 1C discharge")
            ax.plot(rows.cycle.iloc[:2], rows.soh.iloc[:2], color=".65", linestyle=":", linewidth=1)
            ax.scatter(rows.cycle.iloc[0], rows.soh.iloc[0], facecolors="none", edgecolors="#c44e52",
                       s=30, zorder=3, label="Initial capacity test (Q1)")
            ax.set(title=battery.removeprefix(manifest["selection"]["batch"] + "_"),
                   xlabel="Cycle", ylabel="SOH (Q / Q1)")
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(groups):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"XJTU {manifest['selection']['batch']}: {rate}C charge / 1C discharge\n"
                     "Observed SOH; no smoothing or interpolation")
        fig.tight_layout()
        fig.savefig(output / f"{group}_soh_curves.png", dpi=180)
        plt.close(fig)
    lines = [f"# XJTU {rate}C SOH 추출", "",
             f"배터리 **{manifest['n_batteries']}개**, 관측 **{manifest['n_rows']:,}개**. 추출 상태: **{manifest['status']}**.", "",
             "`summary.discharge_capacity_Ah`를 사용하고 `SOH = Q / Q1`로 계산했다. "
             "용량 배열과 cycle 수 일치, 유효한 용량, 시험 조건, 중복 및 정규화 일관성을 확인했다.", "",
             "첫 cycle은 0.5C 충전·0.2C 방전 초기 시험이다. 빨간 원으로 표시하고 "
             f"이후 {rate}C 충전·1C 방전 곡선과 점선으로 연결했다. 첫 구간의 급락에는 시험 조건 차이가 포함된다. "
             "전 cycle을 보존하며 보간·평활화·단조 감소 보정을 하지 않았다. 용량 회복이나 급변만으로 오류로 판단하지 않는다.", "",
             f"![배터리 {manifest['n_batteries']}개의 SOH 곡선]({group}_soh_curves.png)", "",
             "| Battery | Cycles | Q1 (Ah) | 마지막 SOH | 64→16 windows | 확인 결과 |",
             "|---|---:|---:|---:|---:|---|"]
    for row in audit.to_dict("records"):
        if row["status"] == "failed" and row.get("error"):
            lines.append(f"| {row['battery_id']} | — | — | — | — | {row['error']} |")
        else:
            lines.append(f"| {row['battery_id']} | {int(row['n_extracted_cycles'])} | "
                         f"{row['reference_capacity_ah']:.3f} | {row['last_soh']:.4f} | "
                         f"{int(row['n_prediction_windows'])} | {row['status']} |")
    if manifest["selection"].get("audit_level", "detailed") == "detailed":
        lines += ["", "`passed_with_notes`는 초기 시험 조건 차이·용량 회복 등을 기록한 상태이며 오류가 아니다. "
                  "기존 2C의 상세 표시는 공통 데이터의 `quality_flags`와 `audit.csv`에서 확인할 수 있다."]
    location = bundle.relative_to(ROOT) if bundle.is_relative_to(ROOT) else bundle
    lines += ["", f"전체 데이터: `{location.as_posix()}`.",
              "cycle 번호는 공개 MAT 배열의 1부터 시작하는 위치다. 초기 시험은 예측 window에서 제외한다. "
              "64→16 설정의 첫 예측 구간은 cycle 66~81이다. 아직 모델 예측은 실행하지 않았다.", "",
              "검토 폴더에는 audit·배터리 메타데이터·보고서·곡선만 보관한다. 스키마·해시는 전체 데이터 묶음에 있다.", "",
              "[제작자의 시험 조건 설명](https://wang-fujin.github.io/)", "",
              "```powershell", f".\\.venv\\Scripts\\python.exe scripts\\preprocess_xjtu.py --group {rate}c", "```", ""]
    (output / f"{group}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["2c", "3c", "all"], default="all")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/XJTU")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reports-only", action="store_true", help="Read existing bundles without re-extracting raw MAT files")
    args = parser.parse_args()
    if args.group == "all" and any([args.output, args.report_dir, args.config]):
        parser.error("Custom output/report/config paths require a single --group")
    for group in ["2c", "3c"] if args.group == "all" else [args.group]:
        bundle = (args.output or ROOT / f"data/processed/XJTU/{group.upper()}").resolve()
        output = (args.report_dir or ROOT / f"results/data_audit/xjtu_{group}").resolve()
        config = args.config or ROOT / f"configs/xjtu_{group}_preprocess.json"
        policy = json.loads(config.read_text(encoding="utf-8"))
        if source_profile(policy)[0] != int(group[0]):
            parser.error("Configuration does not match the selected group")
        if args.reports_only:
            data, _, manifest = load_bundle(bundle)
            if source_profile(manifest["selection"])[0] != int(group[0]):
                parser.error("Existing bundle does not match the selected group")
            audit = pd.read_csv(bundle / "audit.csv")
        else:
            data, _, audit, manifest = preprocess(args.raw_dir, bundle, policy)
        report(data, audit, manifest, output, bundle)
        print(f"{manifest['status']}: {manifest['n_batteries']} batteries, {manifest['n_rows']} cycles", flush=True)
        print("REPORT=" + str(output / f"xjtu_{group}_report.md"), flush=True)
        if manifest["status"] != "complete":
            raise SystemExit("Essential checks failed; inspect audit.csv")


if __name__ == "__main__":
    main()
