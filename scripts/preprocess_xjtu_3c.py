"""Extract XJTU 3C with essential checks and a curve-focused report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.xjtu_preprocessing import preprocess


def report(data, audit, manifest, output, bundle):
    output.mkdir(parents=True, exist_ok=True)
    for name in ("batteries.csv", "audit.csv"):
        shutil.copy2(bundle / name, output / name)
    if len(data):
        groups = list(data.groupby("battery_id", sort=False))
        fig, axes = plt.subplots((len(groups) + 3) // 4, 4, figsize=(15, 12),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, (battery, rows) in zip(axes.flat, groups):
            normal = rows[rows.cycle > 1]
            ax.plot(normal.cycle, normal.soh, color="#2878b5", linewidth=1.2, label="3C charge / 1C discharge")
            ax.plot(rows.cycle.iloc[:2], rows.soh.iloc[:2], color=".65", linestyle=":", linewidth=1)
            ax.scatter(rows.cycle.iloc[0], rows.soh.iloc[0], facecolors="none", edgecolors="#c44e52",
                       s=30, zorder=3, label="Initial capacity test (Q1)")
            ax.set(title=battery.removeprefix("batch2_"), xlabel="Cycle", ylabel="SOH (Q / Q1)")
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(groups):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=7, loc="lower left")
        fig.suptitle("XJTU Batch 2: 3C charge / 1C discharge\nObserved SOH; no smoothing or interpolation")
        fig.tight_layout()
        fig.savefig(output / "soh_curves.png", dpi=180)
        plt.close(fig)
    lines = ["# XJTU 3C SOH 추출", "",
             f"배터리 **{manifest['n_batteries']}개**, 관측 **{manifest['n_rows']:,}개**. 추출 상태: **{manifest['status']}**.", "",
             "`summary.discharge_capacity_Ah`를 사용하고 `SOH = Q / Q1`로 계산했다. "
             "용량 배열과 cycle 수 일치, 유효한 용량, 시험 조건, 중복 및 정규화 일관성만 확인했다.", "",
             "첫 cycle은 0.5C 충전·0.2C 방전 초기 시험이다. 빨간 원으로 표시하고 "
             "이후 3C 충전·1C 방전 곡선과 점선으로 연결했다. 첫 구간의 급락에는 시험 조건 차이가 포함된다. "
             "전 cycle을 보존하며 보간·평활화·단조 감소 보정을 하지 않았다. 용량 회복이나 급변만으로 오류로 판단하지 않는다.", "",
             "![배터리 15개의 SOH 곡선](soh_curves.png)", "",
             "| Battery | Cycles | Q1 (Ah) | 마지막 SOH | 필수 확인 |",
             "|---|---:|---:|---:|---|"]
    for row in audit.to_dict("records"):
        if row["status"] == "failed" and row.get("error"):
            lines.append(f"| {row['battery_id']} | — | — | — | {row['error']} |")
        else:
            lines.append(f"| {row['battery_id']} | {int(row['n_extracted_cycles'])} | "
                         f"{row['reference_capacity_ah']:.3f} | {row['last_soh']:.4f} | {row['status']} |")
    lines += ["", f"전체 데이터: `{bundle.relative_to(ROOT) if bundle.is_relative_to(ROOT) else bundle}`.",
              "cycle 번호는 공개 MAT 배열의 1부터 시작하는 위치다. 초기 시험은 예측 window에서 제외한다. "
              "64→16 설정의 첫 예측 구간은 cycle 66~81이다. 아직 모델 예측은 실행하지 않았다.", "",
              "[제작자의 시험 조건 설명](https://wang-fujin.github.io/)", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe scripts\\preprocess_xjtu_3c.py", "```", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/XJTU")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/XJTU/3C")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "results/data_audit/xjtu_3c")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/xjtu_3c_preprocess.json")
    args = parser.parse_args()
    policy = json.loads(args.config.read_text(encoding="utf-8"))
    if policy["batch"] != "batch2":
        parser.error("The 3C report requires the Batch-2 configuration")
    data, _, audit, manifest = preprocess(args.raw_dir, args.output, policy)
    report(data, audit, manifest, args.report_dir.resolve(), args.output.resolve())
    print(f"{manifest['status']}: {manifest['n_batteries']} batteries, {manifest['n_rows']} cycles", flush=True)
    print("REPORT=" + str((args.report_dir / "report.md").resolve()), flush=True)
    if manifest["status"] != "complete":
        raise SystemExit("Essential checks failed; inspect audit.csv")


if __name__ == "__main__":
    main()
