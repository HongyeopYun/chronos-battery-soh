"""Audit every raw cycle and build separate strict SOH bundles. No model runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import strict_preprocessing as qc
from src.calce_windows import observation_segments
from src.soh_data import load_bundle, observed_windows, sha256, validate_tables, write_json

EXTRACTOR_SHA256 = sha256(ROOT / "src/strict_preprocessing.py")

DEST = ROOT / "data/processed/strict_v2"
REPORT = ROOT / "results/preprocessing_strict_v2"


def features_for(group, force=False):
    source = ROOT / "data/processed" / qc.BUNDLES[group]
    data, batteries, parent = load_bundle(source)
    folder = REPORT / "evidence"
    folder.mkdir(parents=True, exist_ok=True)
    path, meta_path = folder / f"{group}_raw_features.csv.gz", folder / f"{group}_raw_manifest.json"
    if path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if (meta["feature_version"] != qc.FEATURE_VERSION or meta["feature_sha256"] != sha256(path)
                or meta["parent_manifest_sha256"] != sha256(source / "manifest.json")):
            raise ValueError(f"Stale raw evidence: {group}; use --force")
        return pd.read_csv(path), meta
    dataset = data.dataset.iloc[0]
    raw = ROOT / "data/raw" / qc.RAW_FOLDERS[dataset]
    # Every file, including duplicate CALCE exports, is checked against extraction provenance.
    actual = {name: sha256(raw / name) for name in parent["source_files"]}
    if actual != parent["source_files"]:
        raise ValueError(f"Raw files differ from source bundle: {group}")
    started = time.monotonic()
    implementation_hash = EXTRACTOR_SHA256
    features = getattr(qc, f"extract_{dataset}")(raw, data, batteries)
    if len(features) != len(data):
        raise ValueError("Incomplete raw scan")
    features.to_csv(path, index=False, float_format="%.17g", compression="gzip")
    meta = {"status": "complete", "feature_version": qc.FEATURE_VERSION,
            "group": group, "raw_files_checked": len(actual), "n_cycles": len(features),
            "source_sha256": actual, "parent_manifest_sha256": sha256(source / "manifest.json"),
            "implementation_sha256": implementation_hash,
            "elapsed_seconds": time.monotonic() - started, "feature_sha256": sha256(path)}
    write_json(meta_path, meta)
    print(f"EVIDENCE COMPLETE {group}: {len(features)} cycles, {meta['elapsed_seconds']:.1f}s", flush=True)
    return features, meta


def valid_mask(data):
    return (data.is_observed & data.capacity_ah.gt(0) & np.isfinite(data.soh)
            & data.capacity_kind.isin(["full_discharge", "reference_test"])
            & ~data.quality_flags.map(lambda f: bool(set(str(f).split(";")) & qc.EXCLUDED_FLAGS)))


def plots(group, data):
    names = []
    cells = list(data.groupby("battery_id", sort=False))
    for start in range(0, len(cells), 12):
        subset = cells[start:start + 12]
        columns = min(3, len(subset))
        fig, axes = plt.subplots(math.ceil(len(subset) / columns), columns,
                                 figsize=(15, 3.4 * math.ceil(len(subset) / columns)), squeeze=False)
        for ax, (battery, d) in zip(axes.flat, subset):
            numeric = np.isfinite(d.soh) & d.capacity_ah.gt(0)
            ax.plot(d.cycle, d.soh.where(numeric), color=".7", lw=.8, label="Original")
            ax.plot(d.cycle, d.soh.where(d.quality_eligible), color="#1167a8", lw=1.2, label="Eligible (gaps kept)")
            rejected = numeric & d.quality_status.eq("excluded")
            pending = numeric & d.quality_status.eq("quarantined")
            ax.scatter(d.loc[rejected, "cycle"], d.loc[rejected, "soh"], s=12, marker="x", color="#d53e4f", label="Excluded")
            ax.scatter(d.loc[pending, "cycle"], d.loc[pending, "soh"], s=5, color="#df9d27", label="Quarantined / unverified")
            ax.set(title=f"{battery}: {int(d.quality_eligible.sum())}/{len(d)} eligible", xlabel="Original cycle", ylabel="SOH (Q/Qref)")
            ax.grid(alpha=.2)
        for ax in axes.flat[len(subset):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"{group} | raw-trace QC | no smoothing / interpolation", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .97))
        name = f"{group}_before_after_{start // 12 + 1:02d}.png"
        fig.savefig(REPORT / name, dpi=150)
        plt.close(fig)
        names.append(name)
    return names


def finalize(group):
    source = ROOT / "data/processed" / qc.BUNDLES[group]
    data, batteries, parent = load_bundle(source)
    features, evidence = features_for(group)
    cleaned = qc.apply_quality(data, features)
    # File boundaries alone are not missing cycles. Keep existing chronology and
    # cycle IDs; invalid cycles remain in the table and always break windows.
    if data.dataset.iloc[0] == "calce":
        cleaned["source_segment_id"] = data.segment_id
        for _, rows in data.groupby("battery_id", sort=False):
            cleaned.loc[rows.index, "segment_id"] = observation_segments(rows)
            data.loc[rows.index, "segment_id"] = observation_segments(rows)
    validate_tables(cleaned, batteries)
    audit = []
    for battery, before in data.groupby("battery_id", sort=False):
        after = cleaned[cleaned.battery_id == battery]
        previous = valid_mask(before)
        meta = batteries[batteries.battery_id == battery].iloc[0]
        ref = after[after.cycle == meta.reference_cycle]
        # Initial protocol tests may remain normalization references. Flag any
        # additional raw-trace exclusion at the reference, without changing Qref.
        reference_reasons = "" if ref.empty else ref.exclusion_reasons.iloc[0].replace("initial_capacity_test", "").strip(";")
        audit.append({"group": group, "battery_id": battery, "rows": len(before),
                      "eligible_before": int(previous.sum()), "eligible_after": int(after.quality_eligible.sum()),
                      "newly_excluded": int((previous.to_numpy() & ~after.quality_eligible.to_numpy()).sum()),
                      "quarantined": int(after.quality_status.eq("quarantined").sum()),
                      "excluded": int(after.quality_status.eq("excluded").sum()),
                      "shape_review_retained": int((after.shape_review & after.quality_eligible).sum()),
                      "windows_64_16_before": len(observed_windows(before)["target"]),
                      "windows_64_16_after": len(observed_windows(after)["target"]),
                      "reference_review": reference_reasons})
    audit = pd.DataFrame(audit)
    destination = DEST / group
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "status": "running", "group": group,
                "stage": "raw_trace_quality_control", "source_files": parent["source_files"],
                "parent_manifest_sha256": sha256(source / "manifest.json"),
                "raw_evidence_sha256": evidence["feature_sha256"],
                "n_rows": len(cleaned), "n_batteries": len(batteries),
                "n_eligible": int(cleaned.quality_eligible.sum()), "interpolation": "none", "smoothing": "none",
                "capacity_policy": "Native Q and Qref unchanged; uncertain/non-comparable measurements excluded, not repaired by guessing",
                "cycle_policy": "Original cycle IDs retained. Invalid observations break both context and target windows.",
                "selection_bias_note": "Strict windows condition on completed valid tests; report coverage and compare models on identical windows.",
                "implementation_sha256": {name: sha256(ROOT / name) for name in ["src/strict_preprocessing.py", "src/soh_data.py", "scripts/preprocess_strict_batteries.py"]}}
    write_json(destination / "manifest.json", manifest)
    for name, table in [("soh.csv", cleaned), ("batteries.csv", batteries), ("audit.csv", audit)]:
        table.to_csv(destination / name, index=False, float_format="%.17g")
    schema = json.loads((source / "schema.json").read_text(encoding="utf-8"))
    schema["strict_quality_extension"] = {"quality_eligible": "boolean: mandatory window mask",
                                          "exclusion_reasons": "semicolon-separated raw-QC and prior exclusion reasons",
                                          "shape_review": "past-only absolute SOH change >0.03; not an exclusion"}
    write_json(destination / "schema.json", schema)
    manifest.update(status="complete", artifact_sha256={name: sha256(destination / name)
                    for name in ["soh.csv", "batteries.csv", "audit.csv", "schema.json"]})
    write_json(destination / "manifest.json", manifest)
    loaded, _, _ = load_bundle(destination)
    np.testing.assert_array_equal(loaded.cycle, data.cycle)
    np.testing.assert_allclose(loaded.capacity_ah, data.capacity_ah, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(loaded.soh, data.soh, rtol=1e-12, equal_nan=True)
    images = plots(group, cleaned)
    print(f"BUNDLE COMPLETE {group}: {int(cleaned.quality_eligible.sum())}/{len(cleaned)} eligible; {int(audit.windows_64_16_after.sum())} windows", flush=True)
    return audit, cleaned, images


def report_all(outputs):
    audits = pd.concat([o[0] for o in outputs], ignore_index=True)
    audits.to_csv(REPORT / "battery_summary.csv", index=False)
    selected = []
    for audit, data, images in outputs:
        bad = data[~data.quality_eligible].copy()
        bad.insert(0, "group", audit.group.iloc[0])
        selected.append(bad[["group", "dataset", "battery_id", "cycle", "capacity_ah", "soh", "quality_status", "exclusion_reasons"]])
    excluded = pd.concat(selected, ignore_index=True)
    excluded.to_csv(REPORT / "excluded_cycles.csv", index=False)
    reason = excluded.assign(reason=excluded.exclusion_reasons.str.split(";")).explode("reason")
    counts = reason.groupby(["group", "reason"]).size().rename("cycles").reset_index()
    counts.to_csv(REPORT / "exclusion_counts.csv", index=False)
    totals = audits.groupby("group", sort=False).agg(batteries=("battery_id", "size"), rows=("rows", "sum"),
                before=("eligible_before", "sum"), after=("eligible_after", "sum"), newly_excluded=("newly_excluded", "sum"),
                quarantined=("quarantined", "sum"), excluded=("excluded", "sum"),
                windows_before=("windows_64_16_before", "sum"), windows_after=("windows_64_16_after", "sum"))
    totals.to_csv(REPORT / "dataset_summary.csv")
    cases = [("calce_cs2_1c", "CS2_35", 110, 185, "CALCE: incomplete charge cycles"),
             ("calce_cs2_1c", "CS2_35", 680, 755, "CALCE: late-life charge omissions"),
             ("nasa_aging", "B0005", 1, 168, "NASA: verified recovery retained"),
             ("xjtu_2c", "batch1_2C_battery-1", 360, 390, "XJTU: final discharge not completed")]
    frames = {a.group.iloc[0]: d for a, d, _ in outputs}
    if all(g in frames for g, *_ in cases):
        fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
        for ax, (g, battery, lo, hi, title) in zip(axes.flat, cases):
            d = frames[g]
            d = d[d.battery_id.eq(battery) & d.cycle.between(lo, hi)]
            ax.plot(d.cycle, d.soh, color=".65", lw=1.2, label="Original")
            ax.plot(d.cycle, d.soh.where(d.quality_eligible), color="#1167a8", lw=1.8, label="Eligible")
            for status, color, label in [("excluded", "#d53e4f", "Excluded"), ("quarantined", "#df9d27", "Unverified")]:
                selected = d[d.quality_status.eq(status)]
                ax.scatter(selected.cycle, selected.soh, color=color, marker="x", s=45, zorder=4, label=label)
            ax.set(title=title, xlabel="Original cycle", ylabel="SOH (Q/Qref)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle("Raw-trace QC: exclude non-comparable measurements, preserve verified variation", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(REPORT / "key_before_after.png", dpi=170)
        plt.close(fig)
    lines = ["# 전체 배터리 원시 기록 기반 전처리", "", "원본 용량·SOH·cycle 번호는 보존하고, 품질 기준을 통과한 관측만 예측에 사용한다. 보간·평활화·모델 실행 없음.", "",
             "| 데이터셋 | 기록 수 | 전체 cycle | 기존 유효 | 정제 후 유효 | 추가 제외·보류 | 확인 보류 | 64→16 창: 전→후 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g, r in totals.iterrows():
        lines.append(f"| {g} | {r.batteries} | {r.rows} | {r.before} | {r.after} | {r.newly_excluded} | {r.quarantined} | {r.windows_before} → {r.windows_after} |")
    if (REPORT / "key_before_after.png").exists():
        lines += ["", "![주요 전처리 전후](key_before_after.png)", ""]
    lines += ["", "## 사용과 해석", "", "- 새 데이터는 `data/processed/strict_v2/<group>`에 저장했다. 기존 데이터·모델 결과는 유지했다.",
              "- `soh.csv`는 모든 관측을 보존한다. **quality_eligible=True만 사용**하며, 공통 `observed_windows`가 이를 강제한다. 제외 행을 삭제한 뒤 cycle을 다시 번호 매기면 안 된다.",
              "- 원래의 64→16 연속 창에서 제외 cycle을 건너뛰거나 채우지 않는다. 유효 창 감소는 실제 품질 제약이다.",
              "- `quality_status=quarantined`는 **오류 확정이 아니라 확인 보류**다. 엄격한 기본 마스크에서는 제외하지만, 삭제·보정하지 않는다. 빨강은 제외, 주황은 확인 보류다.",
              "- 충전 미완료, 방전 종료 미도달, 비정상 전압/전류, 시간·카운터 오류, 용량 대조 실패를 구분했다. 재구성 근거가 부족한 용량은 임의로 수정하지 않았다.",
              "- 형태만 튀는 점은 삭제하지 않았다. 과거 SOH 대비 0.03 이상 변한 점은 검토 표시만 한다. 온도·휴지시간·조건 변화로 인한 실제 용량 변동이 남을 수 있다.",
              "- 미래 SOH를 이용한 보정과 정답 평활화는 없다. 단, 평가 대상은 미래 16개 cycle의 측정 품질까지 확인된 창이므로 '유효한 완전 충·방전 조건의 SOH 예측' 평가임을 명시해야 한다.",
              "- 이전 결과와 정제 후 결과는 평가 창이 달라 직접 개선율을 비교할 수 없다. 재평가 시 모든 모델에 동일한 정제 창을 사용해야 한다.",
              "- MIT 연속 기록 5쌍은 기존 physical_battery_id를 유지한다. train/validation/test에서 같은 물리 배터리를 분리하면 안 된다.",
              "", "## 근거와 규칙", "", "- [상세 규칙](../../docs/strict_preprocessing.md)",
              "- [배터리별 요약](battery_summary.csv), [제외 cycle 및 이유](excluded_cycles.csv), [이유별 집계](exclusion_counts.csv)",
              "- `evidence/`에 모든 cycle의 충전 종료·방전 종료·적분·카운터·원본 대응 정보를 보관했다.", ""]
    lines += ["## 핵심 확인 사항", "",
              "- CALCE의 표시된 급락 중 cycle 127·146·157·169·708·716·738(CS2_35)에서는 CV step 4가 없고 충전이 약 0.55A에서 종료됐다. 정상 인접 cycle은 약 0.05A까지 감소한다. 같은 완전 충전 용량으로 비교하기 어려워 제외했다.",
              "- NASA B0005/6/7의 cycle 90은 충전 기록 없이 다음 방전이 시작된다. 용량 오류로 단정하지 않고 확인 보류했다. 이 한 점 때문에 각 배터리에서 연속 80개 관측 창이 크게 줄어든다. B0018과 다른 검증된 용량 회복은 유지했다.",
              "- MIT Batch 3에서는 native QDischarge와 Qd 카운터가 맞아도 전류 시간적분이 수 % 더 큰 관측이 다수 있다. 표본을 단계별로 확인하면 주로 CV 방전에서 차이가 커진다. 적분 차이만 있는 관측은 오류라고 단정하지 않고 보류했으며, **실제 열화 후반 데이터까지 빠질 수 있어 이 엄격본의 성능을 전체 수명 성능으로 해석하면 안 된다.** 원 배포본의 CV 기록·카운터 정의를 확인하기 전 대규모 삭제나 재적분 대체는 하지 않았다.",
              "- MIT 확인 보류까지 제외한 것은 보수적인 검증 정책이다. 보류 관측의 재사용 여부를 결정하기 전에는 원래 MIT 결과를 정제 완료 결과로 대체하거나 모델 우열을 다시 결론내리지 않는다.", ""]
    references = audits[audits.reference_review.ne("")]
    if len(references):
        lines += [f"기준점 검토 표시가 있는 기록 {len(references)}개(기존 noisy channel 포함). `battery_summary.csv`의 reference_review를 확인한다. 기준 용량은 임의 변경하지 않았다.", ""]
    for audit, data, images in outputs:
        lines += [f"## {audit.group.iloc[0]}", ""]
        for name in images:
            lines += [f"![전처리 전후]({name})", ""]
    (REPORT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(DEST / "index.json", {"status": "complete", "groups": {g: str((DEST / g).relative_to(ROOT)).replace("\\", "/") for g in audits.group.unique()},
                                     "context_length": 64, "prediction_length": 16,
                                     "required_quality_column": "quality_eligible"})
    print(totals.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", choices=list(qc.BUNDLES), default=list(qc.BUNDLES))
    parser.add_argument("--stage", choices=["features", "finalize", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    if args.stage in {"features", "all"}:
        for group in args.groups:
            features_for(group, force=args.force)
    if args.stage in {"finalize", "all"}:
        outputs = [finalize(group) for group in args.groups]
        report_all(outputs)


if __name__ == "__main__":
    main()
