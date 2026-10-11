"""Re-audit NASA raw capacity, retaining a verified discharge with an absent charge log."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.soh_data import load_bundle, observed_windows, sha256, write_json
from src.strict_preprocessing import extract_nasa, apply_quality, FEATURE_VERSION

SOURCE = ROOT / 'data/processed/Nasa/nasa_aging'
OLD = ROOT / 'data/processed/strict_v2/nasa_aging'
DEST = ROOT / 'data/processed/nasa_review_v3/nasa_aging'
REPORT = ROOT / 'results/preprocessing_nasa_review_v3'


def main():
    data, batteries, parent = load_bundle(SOURCE)
    old, _, old_manifest = load_bundle(OLD)
    raw = ROOT / 'data/raw/Nasa'
    raw_hashes = {name: sha256(raw/name) for name in parent['source_files']}
    assert raw_hashes == parent['source_files']
    evidence = extract_nasa(raw, data, batteries)
    revised = apply_quality(data, evidence)
    changed = old.quality_eligible.to_numpy() != revised.quality_eligible.to_numpy()
    actual = set(map(tuple, revised.loc[changed, ['battery_id', 'cycle']].to_numpy()))
    assert actual == {(b, 90) for b in ('B0005', 'B0006', 'B0007')}
    assert revised.quality_eligible.all() and len(revised) == 636
    np.testing.assert_array_equal(revised.cycle, data.cycle)
    np.testing.assert_array_equal(revised.capacity_ah, data.capacity_ah)
    np.testing.assert_array_equal(revised.soh, data.soh)
    rows = []
    for b, cell in revised.groupby('battery_id', sort=False):
        previous = old[old.battery_id.eq(b)]
        before = observed_windows(previous, 64, 1)['forecast_start']
        after = observed_windows(cell, 64, 1)['forecast_start']
        np.testing.assert_array_equal(after, np.arange(65, cell.cycle.max()+1))
        restored = sorted(set(after)-set(before))
        if b != 'B0018':
            assert restored == list(range(90,155))
        rows.append(dict(battery_id=b, observations=len(cell), eligible_before=int(previous.quality_eligible.sum()),
                         eligible_after=int(cell.quality_eligible.sum()), targets_before=len(before),
                         targets_after=len(after), restored_targets=len(restored),
                         restored_start=min(restored) if restored else None, restored_end=max(restored) if restored else None))
    audit = pd.DataFrame(rows)
    DEST.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(REPORT/'raw_features.csv.gz', index=False)
    evidence[evidence.cycle.eq(90)].to_csv(REPORT/'cycle90_evidence.csv', index=False)
    revised.loc[changed].to_csv(REPORT/'reclassified_observations.csv', index=False)
    audit.to_csv(REPORT/'coverage.csv', index=False)
    schema = json.loads((SOURCE/'schema.json').read_text(encoding='utf-8'))
    schema['nasa_review_policy'] = 'Absent charge logs alone are review notes when native discharge capacity passes independent trace checks. Charge completion is not asserted.'
    for name, table in [('soh.csv', revised), ('batteries.csv', batteries), ('audit.csv', audit)]:
        table.to_csv(DEST/name, index=False, float_format='%.17g')
    write_json(DEST/'schema.json', schema)
    manifest = dict(schema_version=1, status='complete', group='nasa_aging',
        source_files=raw_hashes, parent_manifest_sha256=sha256(SOURCE/'manifest.json'),
        previous_strict_manifest_sha256=sha256(OLD/'manifest.json'), n_rows=len(revised), n_batteries=len(batteries),
        n_eligible=int(revised.quality_eligible.sum()), feature_version=FEATURE_VERSION,
        scope='Forecast native measured discharge capacity. Missing charging metadata is distinguished from demonstrated capacity error. Not certification of identical full-charge protocols.',
        interpolation='none', smoothing='none', cycle_policy='Original observation numbers preserved; no deletion or reindexing.',
        quality_policy=schema['nasa_review_policy'],
        raw_evidence_sha256=sha256(REPORT/'raw_features.csv.gz'),
        implementation_sha256={name:sha256(ROOT/name) for name in ('src/strict_preprocessing.py','scripts/review_nasa_capacity.py')},
        artifact_sha256={name:sha256(DEST/name) for name in ('soh.csv','batteries.csv','audit.csv','schema.json')})
    write_json(DEST/'manifest.json', manifest)
    loaded, _, _ = load_bundle(DEST)
    np.testing.assert_allclose(loaded.soh, data.soh, rtol=1e-15)
    assert raw_hashes == {name: sha256(raw/name) for name in raw_hashes}
    fig, axes = plt.subplots(2, 2, figsize=(13,8))
    for ax, (b, cell) in zip(axes.flat, revised.groupby('battery_id', sort=False)):
        ax.plot(cell.cycle, cell.soh, color='black', label='All original measured SOH', lw=1.5)
        ax.axvspan(1,64,color='grey',alpha=.10,label='Initial context (no one-step target)')
        if b != 'B0018':
            ax.axvspan(90,154,color='#e99431',alpha=.14,label='Targets previously omitted')
            r=cell[cell.cycle.eq(90)]
            ax.scatter(r.cycle,r.soh,color='#e99431',marker='D',s=42,zorder=4,label='Charge log absent; capacity retained')
        ax.set(title=b,xlabel='Original discharge observation',ylabel='SOH (Q/Qref)')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('NASA audit: capacity values unchanged; no smoothing, interpolation, or cycle compression')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(REPORT/'nasa_restored_coverage.png',dpi=150)
    plt.close(fig)
    lines=['# NASA 중간 구간 누락 원인과 수정', '',
        '원본 636개 관측은 삭제되지 않았다. 기존 strict_v2는 B0005·B0006·B0007의 90번 방전에 연결되는 충전 로그가 없다는 이유만으로 3개 용량 관측을 보류했다. 연속 64개 입력을 요구하므로 각 배터리에서 정답 90~154번까지 65개, 총 195개 예측 시점이 추가로 제외됐다.', '',
        '예측 그래프의 실제값도 평가 시점만 표시한 별도의 표시 오류가 있었다. 전체 원본 SOH 곡선과 평가 가능한 예측 시점을 구분해서 표시해야 한다.', '',
        '## 원시 자료 재확인', '',
        '- 세 원본 방전은 4.22~4.23V에서 시작하고 약 2A로 방전하며 Capacity 정의의 2.7V까지 도달한다. 전압·전류·시간에 기존 제외 사유가 없고, 원본 Capacity와 전류 시간적분 차이는 약 0.17~0.18%다.',
        '- 충전 로그가 없으므로 직전 충전이 규정된 CC-CV 과정을 완료했는지는 확정할 수 없다. 그러나 이것만으로 기록된 방전 용량이 잘못됐다고 판단할 근거도 없다. 이 경우는 관측을 유지하고 `preceding_charge_unrecorded_native_capacity_retained` 검토 표시를 남긴다.',
        '- 실제로 기록된 충전 미완료, 방전 미완료, 비정상 전류, 시간 오류, 적분 불일치 등은 계속 제외한다. 다른 데이터셋의 누락 충전 기준을 이번 수정으로 해제하지 않는다.',
        '- 원시 파일 체크섬, Q, Qref, SOH, cycle 번호를 모두 유지했다. 보간·평활화·단조화·번호 압축이 없다. 새 bundle은 `data/processed/nasa_review_v3/nasa_aging`이며 strict_v2는 비교 증거로 보존한다.',
        '- 이 수정은 모델 점수를 보기 전에 원시 기록과 측정 의미를 기준으로 결정했다. 이전 NASA 점수는 중간 구간이 빠진 부분 평가이므로 전체 NASA 성능으로 사용하지 않는다.', '',
        '| 배터리 | 전체 관측 | 이전 평가 시점 | 수정 평가 시점 | 복구 시점 |', '|---|---:|---:|---:|---|']
    for r in audit.itertuples():
        lines.append(f'| {r.battery_id} | {r.observations} | {r.targets_before} | {r.targets_after} | '+('90~154 (65개)' if r.restored_targets else '변경 없음')+' |')
    lines+=['','![복구 범위](nasa_restored_coverage.png)','',
        '[90번 원시 검증 수치](cycle90_evidence.csv) · [판정 변경 3개 관측](reclassified_observations.csv) · [평가 범위](coverage.csv)','',
        'NASA 공식 설명에서 Capacity는 2.7V까지의 방전 용량(Ah)이다: [NASA 데이터 설명](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets). 이 문서의 검토·제외 정책은 원 제작자의 삭제 권고가 아니라 본 프로젝트의 명시적 정책이다.','']
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(REPORT/'verification.json',dict(raw_hashes_unchanged=True,capacity_soh_cycle_unchanged=True,
        reclassified_observations=3,targets_before=int(audit.targets_before.sum()),targets_after=int(audit.targets_after.sum()),
        restored_targets=int(audit.restored_targets.sum()),bundle_manifest_sha256=sha256(DEST/'manifest.json')))
    print(audit.to_string(index=False))


if __name__=='__main__':
    main()
