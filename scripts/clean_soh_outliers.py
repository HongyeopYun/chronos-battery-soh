"""Create conservative cleaned bundles and a small before/after report."""
from __future__ import annotations

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
from src.soh_cleaning import clean_bundle
from src.soh_data import load_bundle

GROUPS = [("xjtu_2c", "XJTU/2C"), ("xjtu_3c", "XJTU/3C"), ("nasa_aging", "Nasa/nasa_aging"),
          ("mit_batch1", "MIT/mit_batch1"), ("mit_batch2", "MIT/mit_batch2"),
          ("mit_batch3", "MIT/mit_batch3"), ("calce_cs2_1c", "Calce/calce_cs2_1c")]


def report(summary, removed, comparisons, policy, output):
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "soh_outlier_cleaning_summary.csv", index=False)
    removed.to_csv(output / "soh_outliers_removed.csv", index=False, float_format="%.17g")
    images = []
    for page, start in enumerate(range(0, len(comparisons), 4), 1):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        for ax, (event, original, cleaned) in zip(axes.flat, comparisons[start:start + 4]):
            zoom = original[(original.cycle >= event.cycle - 8) & (original.cycle <= event.cycle + 8)].copy()
            after = zoom.soh.where(zoom.cycle.isin(cleaned.cycle))
            ax.plot(zoom.cycle, zoom.soh, color=".7", linewidth=1, label="Before")
            ax.plot(zoom.cycle, after, color="#2878b5", marker=".", linewidth=1.5, label="Retained observations")
            ax.scatter([event.cycle], [event.soh], color="#c44e52", marker="x", s=65, zorder=3, label="Excluded spike")
            ax.set(title=f"{event.battery_id}, cycle {event.cycle}", xlabel="Original cycle", ylabel="SOH (Q / Q_ref)")
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(comparisons[start:start + 4]):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle("Conservative SOH outlier exclusion\nOriginal values and cycle numbers retained; no interpolation")
        fig.tight_layout(rect=(0, 0, 1, .94))
        name = f"soh_outlier_cleaning_comparison_{page:02d}.png"
        fig.savefig(output / name, dpi=170)
        plt.close(fig)
        images.append(name)
    lines = ["# SOH 이상치 제거 결과", "",
             f"전체 **{int(summary.rows_before.sum()):,}행**에서 급등 이상치 **{len(removed)}개**를 제외했다. "
             f"제거본은 **{int(summary.rows_after.sum()):,}행**이다.", "",
             f"고정한 보수적 규칙: SOH가 **{100 * policy['minimum_soh']:g}% 초과**이고, "
             f"직전 최대 {policy['history_length']}개 유효 관측의 중앙값보다 **{100 * policy['minimum_excess_soh']:g}%p 초과** 높을 때 제외한다. "
             f"이전 유효 관측이 {policy['minimum_history']}개 이상 있어야 한다. 배터리·시험 조건·측정 구간이 바뀌면 과거 기준을 다시 시작한다.", "",
             "현재 관측 이전 값만 사용하며 전체 수명 평균·미래 이웃값·예측 성능으로 기준을 조정하지 않았다. "
             "이 규칙은 큰 급등에 한정한다. 정상적인 작은 용량 회복, 초기 시험 조건 전환, 급격한 열화, 순간 하락값은 자동 제거하지 않았다.", "",
             "| 그룹 | 원래 행 | 제거 | 남은 행 | 64→16 windows 전 | 후 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.group} | {r.rows_before:,} | {r.removed} | {r.rows_after:,} | {r.windows_before:,} | {r.windows_after:,} |")
    lines += ["", "| Battery | Cycle | 제거한 SOH | 직전 유효값 중앙값 |",
              "|---|---:|---:|---:|"]
    for r in removed.itertuples():
        lines.append(f"| {r.battery_id} | {r.cycle} | {100 * r.soh:.2f}% | {100 * r.past_median_soh:.2f}% |")
    for name in images:
        lines += ["", f"![제거 전후 비교]({name})"]
    lines += ["", "원본과 기존 `data/processed/<dataset>/<group>` 묶음은 보존했다. "
              "새 예측용 묶음은 `data/processed/clean/<group>/`에 저장했다.", "",
              "- `soh.csv`: 급등 이상치를 제외한 데이터. 원래 cycle 번호와 나머지 capacity·SOH는 유지한다.",
              "- `soh_flagged.csv`: 모든 원래 행을 유지하고 해당 관측에 `capacity_outlier`를 표시한 비교용 데이터.",
              "- `audit.csv`, `batteries.csv`, `schema.json`, `manifest.json`: 제거 수·window 수·기준 용량·원본 묶음 해시를 기록한다.", "",
              "제거된 cycle을 보간하거나 앞뒤 구간을 이어 붙이지 않는다. 이상치가 포함됐던 예측 window도 함께 제외되므로 "
              "제거 전후 성능을 비교할 때 같은 평가 구간을 사용해야 한다. target 품질 제외 여부는 해당 측정값을 관측한 후 결정된다. "
              "이상치 처리가 실제 예측 성능을 개선했는지는 아직 평가하지 않았다.", "",
              "[전처리와 데이터 누출에 관한 참고](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe scripts\\clean_soh_outliers.py", "```", ""]
    (output / "soh_outlier_cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/soh_outlier_cleaning.json")
    args = parser.parse_args()
    policy = json.loads(args.config.read_text(encoding="utf-8"))
    summaries, events, comparisons = [], [], []
    for group, directory in GROUPS:
        source = ROOT / "data/processed" / directory
        destination = ROOT / "data/processed/clean" / group
        cleaned, removed, audit, manifest = clean_bundle(source, destination, policy)
        removed.insert(0, "group", group)
        events.append(removed)
        summaries.append({"group": group, "rows_before": manifest["n_rows_before"], "removed": manifest["n_removed_outliers"],
                          "rows_after": len(cleaned), "windows_before": int(audit.windows_before.sum()),
                          "windows_after": int(audit.windows_after.sum())})
        if len(removed):
            original, _, _ = load_bundle(source)
            for event in removed.itertuples():
                comparisons.append((event, original[original.battery_id == event.battery_id],
                                    cleaned[cleaned.battery_id == event.battery_id]))
        print(f"CLEAN {group}: removed={manifest['n_removed_outliers']}, rows={len(cleaned)}", flush=True)
    output = ROOT / "results/data_audit/outlier_cleaning"
    report(pd.DataFrame(summaries), pd.concat(events, ignore_index=True), comparisons, policy, output)
    print("REPORT=" + str(output / "soh_outlier_cleaning_report.md"), flush=True)


if __name__ == "__main__":
    main()
