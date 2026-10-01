"""Extract XJTU 2C batteries, audit the common SOH bundle and render a QC report."""
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
from src.soh_data import sha256, write_json
from src.xjtu_preprocessing import preprocess


def report(data, audit, manifest, output, bundle):
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle / "audit.csv", output / "audit.csv")
    shutil.copy2(bundle / "batteries.csv", output / "batteries.csv")
    shutil.copy2(bundle / "manifest.json", output / "manifest.json")
    shutil.copy2(bundle / "schema.json", output / "schema.json")
    preview = pd.concat([rows.iloc[sorted(set(range(min(3, len(rows)))) |
                                              set(range(max(0, len(rows)-2), len(rows))))]
                         for _, rows in data.groupby(["dataset", "battery_id"], sort=False)],
                        ignore_index=True) if len(data) else data
    preview.to_csv(output / "soh_preview.csv", index=False, float_format="%.17g")
    events = data[data.quality_flags.str.contains("initial_capacity_test|protocol_transition|large_soh_change", regex=True)].copy()
    changes = data.groupby(["dataset", "battery_id"], sort=False).soh.diff()
    events["soh_change"] = changes.loc[events.index]
    events.to_csv(output / "quality_events.csv", index=False, float_format="%.17g")
    if len(data):
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
        for ax, (battery, rows) in zip(axes.flat, data.groupby("battery_id", sort=False)):
            ax.plot(rows.cycle, rows.soh, color="#2878b5", linewidth=1.3, label="Observed SOH")
            ax.scatter(rows.cycle.iloc[0], rows.soh.iloc[0], marker="o", facecolors="none",
                       edgecolors="#c44e52", s=40, zorder=3, label="Initial capacity test")
            ax.set(title=battery.replace("batch1_", ""), xlabel="Cycle (source array index)", ylabel="SOH (Q / Q1)")
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[data.battery_id.nunique():]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle("XJTU Batch 1: 2C charge / 1C discharge\nAll measured cycles retained; cycle 1 is excluded only when constructing forecast windows")
        fig.tight_layout()
        fig.savefig(output / "soh_curves.png", dpi=180)
        plt.close(fig)
    lines = ["# XJTU 2C 데이터 Audit 및 공통 SOH 추출", "",
             f"상태: **{manifest['status']}**. 배터리 {manifest['n_batteries']}개, cycle 관측 {manifest['n_rows']:,}개.", "",
             "## 추출 규칙", "",
             "- `summary.discharge_capacity_Ah`를 cycle별 용량으로 추출하고 `SOH = Q / Q1`로 계산했다.",
             "- 모든 cycle을 보존했다. 첫 cycle은 0.5C 충전·0.2C 방전 초기 용량 시험으로 표시했다.",
             "- 이후 cycle은 2C 충전·1C 방전이다. 이 규칙은 Batch 1만 지원한다.",
             "- 용량 증가와 SOH > 1은 표시하고 보존한다. 보간·평활화·clipping·강제 단조 감소는 적용하지 않는다.",
             f"- 같은 시험 조건에서 SOH 변화가 {100 * manifest['selection']['large_soh_change_threshold']:g}%p를 초과하면 추가 검토용으로 표시한다. 제외 기준으로 사용하지 않는다. 초기 시험 전환은 별도 표시한다.",
             "- cycle은 공개 MAT `data` 배열의 1-based 위치다. 원래 장비 기록의 누락 cycle 여부는 이 인덱스만으로 판단할 수 없다.",
             "- 예측 준비에서는 초기 시험 cycle과 비정상 관측을 제외하고, 연속된 실제 관측의 64 → 16 window만 사용한다.",
             "- 원시 데이터와 기존 NASA 실험은 변경하지 않았다. 이번 작업에서 모델 예측은 실행하지 않았다.", "",
             "[데이터 제작자의 시험 조건](https://wang-fujin.github.io/)", "", "## 배터리별 확인", "",
             "| Battery | 보존 cycle | Q1 (Ah) | 마지막 SOH | 용량 증가 횟수 | 큰 SOH 변화 | 64→16 windows | 상태 |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in audit.to_dict("records"):
        if row["status"] == "failed" and row.get("error"):
            lines.append(f"| {row['battery_id']} | — | — | — | — | — | — | {row['error']} |")
        else:
            lines.append(f"| {row['battery_id']} | {row['n_extracted_cycles']} | {row['reference_capacity_ah']:.3f} | "
                         f"{row['last_soh']:.6f} | {row['n_capacity_increases']} | {row['n_large_soh_changes']} | {row['n_prediction_windows']} | {row['status']} |")
    lines += ["", "`passed_with_notes`는 오류가 아니라 초기 시험 조건 차이·용량 증가 관측 등을 기록했다는 뜻이다.",
              "정규화 기준 Q1을 유지하므로 cycle 2의 SOH가 1보다 작은 것은 정상이다. 용량 증가를 열화 제거 대상으로 취급하지 않는다.", "",
              "큰 SOH 변화는 측정 이상인지 실제 거동인지 확정하지 않았다. `quality_events.csv`에 초기 시험·조건 전환·큰 변화의 cycle을 기록했다.", "",
              "![XJTU 2C SOH curves](soh_curves.png)", "", "## 산출물", "",
              f"전체 공통 데이터: `{bundle.relative_to(ROOT) if bundle.is_relative_to(ROOT) else bundle}`.",
              "`soh.csv`, `batteries.csv`, `audit.csv`, `schema.json`, `manifest.json`을 저장했다.",
              "전체 가공 데이터는 gitignore 대상이다. 이 보고서 폴더에는 audit·메타데이터·해시·40행 미리보기·그림을 보관한다.", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe scripts\\preprocess_xjtu_2c.py", "```", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "report_provenance.json", {"bundle_manifest_sha256": sha256(bundle / "manifest.json"),
        "report_script_sha256": sha256(Path(__file__)),
        "report_artifact_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "report_provenance.json"}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/XJTU")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/XJTU/2C")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "results/data_audit/xjtu_2c")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/xjtu_2c_preprocess.json")
    args = parser.parse_args()
    policy = json.loads(args.config.read_text(encoding="utf-8"))
    data, batteries, audit, manifest = preprocess(args.raw_dir, args.output, policy)
    report(data, audit, manifest, args.report_dir.resolve(), args.output.resolve())
    print(audit.to_string(index=False), flush=True)
    print("BUNDLE=" + str(args.output.resolve()), flush=True)
    print("REPORT=" + str((args.report_dir / "report.md").resolve()), flush=True)
    if manifest["status"] != "complete":
        raise SystemExit("Extraction audit failed; inspect audit.csv before forecasting")


if __name__ == "__main__":
    main()
