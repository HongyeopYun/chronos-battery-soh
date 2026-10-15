"""Build audited v4 forecast-safe and retrospective views; never edit v3."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_data import load_bundle, sha256, validate_tables, write_json
from src.strict_preprocessing import BUNDLES, RAW_FOLDERS
from src.preprocessing_v4 import (
    ML_REVISION, LIFE_REVISION, MIT_CONTINUATIONS, MIT_NOT_REACHED_80, PROFILES,
    apply_exclusions, centered_calce_rule, extract_calce_file, extract_xjtu_file,
    mit_exclusions, load_v4_bundle,
)

PARENT = ROOT / 'data/processed/reviewed_native_v3'
OUTPUT = ROOT / 'data/processed/reviewed_v4'
REPORT = ROOT / 'results/preprocessing_batterylife_v4'
SOURCES = ['src/preprocessing_v4.py', 'scripts/build_preprocessing_v4.py']
UPSTREAM = {
    'batteryml_CALCE.py.txt': f'microsoft/BatteryML/{ML_REVISION}/batteryml/preprocess/preprocess_CALCE.py',
    'batterylife_CALCE.py.txt': f'Ruifeng-Tan/BatteryLife/{LIFE_REVISION}/process_scripts/preprocess_CALCE.py',
    'batterylife_MATR.py.txt': f'Ruifeng-Tan/BatteryLife/{LIFE_REVISION}/process_scripts/preprocess_MATR.py',
    'batterylife_XJTU.py.txt': f'Ruifeng-Tan/BatteryLife/{LIFE_REVISION}/process_scripts/preprocess_XJTU.py',
    'batterylife_XJTU_tools.py.txt': f'Ruifeng-Tan/BatteryLife/{LIFE_REVISION}/process_scripts/Extract_life_labels_tools/XJTU_tools.py',
}


def csv(path, frame):
    frame.to_csv(path, index=False, float_format='%.17g', lineterminator='\n')


def get_evidence(name, paths, function, workers):
    folder = REPORT / 'evidence'
    folder.mkdir(parents=True, exist_ok=True)
    dest, info = folder / f'{name}.csv.gz', folder / f'{name}.json'
    expected = dict(raw_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in paths},
                    extractor_sha256=sha256(ROOT/'src/preprocessing_v4.py'))
    if dest.exists() and info.exists():
        prior = json.loads(info.read_text())
        if any(prior[k] != v for k, v in expected.items()) or prior['artifact_sha256'] != sha256(dest):
            raise ValueError(f'Stale evidence: {name}; preserve it and use a new report path')
        return pd.read_csv(dest, keep_default_na=False)
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(function, str(p)): p for p in paths}
        for done in as_completed(pending):
            p = pending[done]
            rows = done.result()
            records.extend(rows)
            print(f'EXTRACT {name} {p.name}: {len(rows)} cycles', flush=True)
    for relative, expected_hash in expected['raw_sha256'].items():
        assert sha256(ROOT/relative) == expected_hash, relative
    data = pd.DataFrame(records).sort_values(['source_file', 'source_cycle'] if name == 'calce' else ['battery_id', 'cycle'])
    csv(dest, data)
    write_json(info, dict(**expected, artifact_sha256=sha256(dest), n_rows=len(data)))
    return data


def extract(workers):
    calce = get_evidence('calce', sorted((ROOT/'data/raw/Calce').rglob('*.xlsx')), extract_calce_file, workers)
    xjtu = get_evidence('xjtu', sorted((ROOT/'data/raw/XJTU').glob('*.mat')), extract_xjtu_file, min(workers, 2))
    return calce, xjtu


def add_lineage(data, group):
    d = data.copy()
    d['source_group'] = group
    d['source_battery_id'] = d.battery_id
    d['source_original_cycle'] = d.cycle
    d['source_raw_cycle_index'] = d.groupby('battery_id', sort=False).cumcount()
    d['capacity_v3_ah'] = d.capacity_ah
    d['soh_v3'] = d.soh
    d['eligible_v3'] = d.quality_eligible
    d['physical_battery_id'] = d.battery_id
    return d


def build_calce(data, meta, evidence):
    d, m = data.copy(), meta.copy()
    records = []
    for row in d.itertuples():
        matching = []
        for source in row.source_file.split(';'):
            e = evidence[evidence.source_file.eq(Path(source).name) & evidence.source_cycle.eq(row.source_cycle)
                         & evidence.source_start_time.eq(row.source_start_time) & evidence.source_end_time.eq(row.source_end_time)]
            if len(e) != 1:
                raise ValueError(f'CALCE lineage mismatch: {row.battery_id}/{row.cycle}/{source}: {len(e)}')
            matching.append(e.iloc[0])
        q = np.array([x.integrated_discharge_ah for x in matching])
        np.testing.assert_allclose(q, q[0], rtol=0, atol=1e-12)
        records.append(matching[0])
    e = pd.DataFrame(records).reset_index(drop=True)
    d['capacity_ah'] = e.integrated_discharge_ah.to_numpy()
    d['charge_capacity_ah'] = e.integrated_charge_ah.to_numpy()
    d['capacity_method'] = 'right_endpoint_current_time_integral'
    d['capacity_delta_ah'] = d.capacity_ah - d.capacity_v3_ah
    d = apply_exclusions(d, [['calce_capacity_le_0p1ah'] if q <= .1 else [] for q in d.capacity_ah])
    for idx, battery in m.iterrows():
        rows = d.battery_id.eq(battery.battery_id)
        qref = float(d.loc[rows & d.cycle.eq(battery.reference_cycle), 'capacity_ah'].iloc[0])
        assert np.isfinite(qref) and qref > .1
        m.loc[idx, 'reference_capacity_v3_ah'] = battery.reference_capacity_ah
        m.loc[idx, 'reference_capacity_ah'] = qref
        m.loc[idx, 'reference_method'] = 'integrated_at_original_first_complete_discharge'
        d.loc[rows, 'soh'] = d.loc[rows, 'capacity_ah'] / qref
        keep, median, residual, scale = centered_calce_rule(d.loc[rows, 'capacity_ah'])
        d.loc[rows, 'retrospective_calce_keep'] = keep
        d.loc[rows, 'retrospective_median_ah'] = median
        d.loc[rows, 'retrospective_residual_ah'] = residual
        d.loc[rows, 'retrospective_global_scale_ah'] = scale
    # These chronological observation coordinates already deduplicate exports.
    d['source_segment_id'] = d.source_file
    d['segment_id'] = d.battery_id
    return d, m


def build_mit(parents):
    frames, metadata = [], []
    for group in ('mit_batch1', 'mit_batch2', 'mit_batch3'):
        d, m, manifest = parents[group]
        d = add_lineage(d, group)
        # Reuse independently extracted, hash-verified raw Qd counter evidence.
        p = ROOT/'results/preprocessing_reviewed_native_v3'/f'{group}_raw_features.csv.gz'
        assert sha256(p) == manifest['raw_evidence_sha256']
        e = pd.read_csv(p)
        aligned = d[['battery_id', 'cycle', 'capacity_ah']].merge(
            e[['battery_id', 'cycle', 'raw_capacity_ah', 'discharge_counter_ah']],
            on=['battery_id', 'cycle'], validate='one_to_one')
        assert len(aligned) == len(d)
        np.testing.assert_allclose(aligned.capacity_ah, aligned.raw_capacity_ah, rtol=1e-12, equal_nan=True)
        d['capacity_method'] = 'native_QDischarge_summary_verified_against_raw_Qd'
        d['physical_battery_id'] = d.battery_id.map(lambda b: MIT_CONTINUATIONS.get(b, b))
        d['battery_id'] = d.physical_battery_id
        frames.append(d)
        metadata.append(m)
    d = pd.concat(frames, ignore_index=True).sort_values(['battery_id', 'cycle']).reset_index(drop=True)
    d['canonical_raw_cycle_index'] = d.groupby('battery_id', sort=False).cumcount()
    assert not d.duplicated(['battery_id', 'cycle']).any()
    for _, r in d.groupby('battery_id'):
        np.testing.assert_array_equal(r.cycle, r.canonical_raw_cycle_index + 1)
    d['segment_id'] = d.battery_id
    d = apply_exclusions(d, [mit_exclusions(b, int(c)) for b, c in zip(d.battery_id, d.canonical_raw_cycle_index)])
    m = pd.concat(metadata, ignore_index=True).set_index('battery_id')
    for child, parent in MIT_CONTINUATIONS.items():
        np.testing.assert_allclose(m.loc[child, 'reference_capacity_ah'], m.loc[parent, 'reference_capacity_ah'])
        for col in ('source_file', 'source_sha256'):
            m.loc[parent, col] = str(m.loc[parent, col]) + ';' + str(m.loc[child, col])
        m.loc[parent, 'continuation_sources'] = child
        m.loc[parent, 'cycle_number_source'] = 'original_batch1_then_batch2_with_original_offset'
    m = m.drop(index=list(MIT_CONTINUATIONS)).reset_index()
    result = {}
    for group in ('mit_batch1', 'mit_batch2', 'mit_batch3'):
        batch = group.replace('mit_', '')
        subset = m[m.batch.eq(batch)].copy()
        result[group] = (d[d.battery_id.isin(subset.battery_id)].copy().reset_index(drop=True), subset.reset_index(drop=True))
    return result


def build_xjtu(data, meta, evidence):
    d, m = data.copy(), meta.copy()
    e = d[['battery_id', 'cycle']].merge(evidence, on=['battery_id', 'cycle'], validate='one_to_one', how='left')
    assert len(e) == len(d) and e.source_file.notna().all()
    np.testing.assert_allclose(d.capacity_ah, e.summary_capacity_ah, rtol=1e-12, atol=1e-12)
    assert (d.cycle_description.to_numpy() == e.description.to_numpy()).all()
    d['capacity_ah'] = e.discharge_capacity_split_ah.to_numpy()
    d['charge_capacity_ah'] = e.charge_capacity_split_ah.to_numpy()
    d['capacity_delta_ah'] = d.capacity_ah - d.capacity_v3_ah
    d['capacity_method'] = 'native_capacity_Ah_discharge_mask_le_minus_0p01C'
    refs = m.set_index('battery_id').reference_capacity_ah
    d['soh'] = d.capacity_ah / d.battery_id.map(refs)
    for col in ('source_file', 'attribute', 'measured_discharge_rate_c', 'measured_discharge_cutoff_v',
                'charge_end_voltage_v', 'measurement_condition', 'time_resets', 'time_duplicates', 'elapsed_span_s'):
        d[col] = e[col].to_numpy()
    reasons = []
    for r in e.itertuples():
        bad = list(filter(None, str(r.condition_reasons).split(';')))
        if r.cycle == 1:
            bad.append('batterylife_initial_rpt_excluded')
        elif r.attribute != 'Cycling':
            bad.append('xjtu_reference_test_separate_from_cycling')
        reasons.append(bad)
    d = apply_exclusions(d, reasons)
    d['segment_id'] = d.measurement_condition
    m['normalization_note'] = 'Fixed v3 initial RPT Q_ref retained for comparison; nominal-capacity SOH also exported'
    return d, m


def view(data, meta, profile):
    d, m = data.copy(), meta.copy()
    if profile == 'retrospective':
        reasons = []
        for r in d.itertuples():
            extra = []
            if r.dataset == 'calce' and not r.retrospective_calce_keep:
                extra.append('batterylife_centered_median21_global_residual')
            if r.dataset == 'mit' and r.battery_id in MIT_NOT_REACHED_80:
                extra.append('batterylife_unreached_80_cell_selection')
            reasons.append(extra)
        d = apply_exclusions(d, reasons)
    else:
        # Keep future-derived diagnostics out of the default forecast table.
        d = d.drop(columns=[c for c in d if c.startswith('retrospective_')])
    d['soh_nominal'] = d.capacity_ah / d.battery_id.map(m.set_index('battery_id').nominal_capacity_ah)
    for idx, row in m.iterrows():
        r = d[d.battery_id.eq(row.battery_id)]
        m.loc[idx, 'eligible_observations'] = int(r.quality_eligible.sum())
        m.loc[idx, 'included_for_prediction'] = bool(r.quality_eligible.any())
    return d, m


def write_bundle(group, profile, data, meta, source_hashes):
    dest = OUTPUT/profile/group
    if dest.exists():
        raise ValueError(f'Output exists; do not overwrite: {dest}')
    dest.mkdir(parents=True)
    validate_tables(data, meta)
    csv(dest/'soh.csv', data)
    csv(dest/'batteries.csv', meta)
    write_json(dest/'schema.json', dict(schema_version=1, cycle='Original record coordinates, never compressed',
        excluded='Rows retained for audit; quality_eligible=False means no target or plotted point',
        soh='capacity_ah / reference_capacity_ah', soh_nominal='capacity_ah / nominal_capacity_ah',
        missing='No interpolation, smoothing, invented observations or lifetime labels'))
    write_json(dest/'manifest.json', dict(schema_version=1, status='complete', preprocessing_version='reviewed_v4',
        group=group, profile=profile, future_dependent_qc=profile=='retrospective',
        n_rows=len(data), n_batteries=len(meta), n_eligible=int(data.quality_eligible.sum()),
        source_files=source_hashes, implementation_sha256={s: sha256(ROOT/s) for s in SOURCES},
        parent_manifests={g: sha256(PARENT/g/'manifest.json') for g in BUNDLES},
        upstream_revisions=dict(BatteryML=ML_REVISION, BatteryLife=LIFE_REVISION),
        artifact_sha256={n: sha256(dest/n) for n in ('soh.csv', 'batteries.csv', 'schema.json')}))
    loaded, _, _ = load_v4_bundle(dest, allow_retrospective=profile=='retrospective')
    np.testing.assert_allclose(loaded.capacity_ah, data.capacity_ah, rtol=1e-12, atol=1e-12, equal_nan=True)


def protected_snapshot():
    files = list(PARENT.rglob('*')) + list((ROOT/'results/c2-battery').rglob('*'))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in files if p.is_file()}


def build(calce_evidence, xjtu_evidence):
    if OUTPUT.exists():
        raise ValueError('A v4 output exists; use a new output directory rather than overwriting')
    protected = protected_snapshot()
    write_json(REPORT/'protected_before.json', protected)
    parents = {g: load_bundle(PARENT/g) for g in BUNDLES}
    raw_hashes = {}
    for group, (d, _, manifest) in parents.items():
        for file, expected in manifest['source_files'].items():
            p = ROOT/'data/raw'/RAW_FOLDERS[d.dataset.iloc[0]]/file
            actual = sha256(p)
            assert actual == expected, str(p)
            raw_hashes[p.relative_to(ROOT).as_posix()] = actual
    prepared = build_mit(parents)
    for group, (d, m, _) in parents.items():
        if group.startswith('mit_'):
            continue
        d = add_lineage(d, group)
        if group == 'calce_cs2_1c':
            d, m = build_calce(d, m, calce_evidence)
            csv(REPORT/'calce_filter_diagnostics.csv', d[['battery_id', 'cycle', 'capacity_v3_ah', 'capacity_ah',
                'capacity_delta_ah', 'retrospective_calce_keep', 'retrospective_median_ah',
                'retrospective_residual_ah', 'retrospective_global_scale_ah']])
        elif group.startswith('xjtu_'):
            d, m = build_xjtu(d, m, xjtu_evidence)
        else:
            d['capacity_method'] = 'unchanged_reviewed_v3_native_capacity'
        prepared[group] = (d, m)
    coverage, changes, lineage = [], [], []
    for profile in PROFILES:
        for group in BUNDLES:
            d, m = view(*prepared[group], profile)
            raw_names = set(d.source_file.str.split(';').explode()) if 'source_file' in d else set()
            used_raw = {p: h for p, h in raw_hashes.items() if Path(p).name in {Path(n).name for n in raw_names}}
            write_bundle(group, profile, d, m, used_raw)
            for b, r in d.groupby('battery_id', sort=False):
                coverage.append(dict(profile=profile, group=group, battery_id=b, rows=len(r),
                    v3_eligible_in_these_rows=int(r.eligible_v3.sum()), eligible=int(r.quality_eligible.sum()),
                    newly_excluded=int((r.eligible_v3 & ~r.quality_eligible).sum()),
                    capacity_changes=int((abs(r.capacity_ah-r.capacity_v3_ah)>1e-12).sum())))
            changed = d.eligible_v3.ne(d.quality_eligible) | (abs(d.capacity_ah-d.capacity_v3_ah)>1e-12)
            columns = ['source_group','source_battery_id','source_original_cycle','battery_id','cycle',
                       'capacity_v3_ah','capacity_ah','soh_v3','soh','eligible_v3','quality_eligible','exclusion_reasons']
            changes.append(d.loc[changed, columns].assign(profile=profile, group=group))
            if profile == 'forecast_safe':
                lineage.append(d[['source_group','source_battery_id','source_original_cycle','battery_id','cycle']].assign(group=group))
            print('COMPLETE', profile, group, len(d), int(d.quality_eligible.sum()), flush=True)
    coverage = pd.DataFrame(coverage)
    csv(REPORT/'battery_coverage.csv', coverage)
    csv(REPORT/'dataset_coverage.csv', coverage.groupby(['profile','group'], sort=False).agg(
        cells=('battery_id','size'), eligible_cells=('eligible',lambda x:int((x>0).sum())),
        rows=('rows','sum'), v3_eligible_in_these_rows=('v3_eligible_in_these_rows','sum'),
        eligible=('eligible','sum'), newly_excluded=('newly_excluded','sum'), capacity_changes=('capacity_changes','sum')).reset_index())
    csv(REPORT/'changed_observations.csv.gz', pd.concat(changes, ignore_index=True))
    lineage = pd.concat(lineage, ignore_index=True)
    csv(REPORT/'lineage.csv.gz', lineage)
    expected_keys = {(g, r.battery_id, int(r.cycle)) for g,(d,_,_) in parents.items() for r in d.itertuples()}
    actual_keys = set(lineage[['source_group','source_battery_id','source_original_cycle']].itertuples(index=False,name=None))
    assert len(lineage) == len(expected_keys) and actual_keys == expected_keys
    assert not lineage.duplicated(['source_group','source_battery_id','source_original_cycle']).any()
    cell_groups = lineage[['group','battery_id']].drop_duplicates()
    assert not cell_groups.battery_id.duplicated().any(), 'A physical cell must occur in exactly one group'
    for p,h in {**protected, **raw_hashes}.items():
        assert sha256(ROOT/p) == h, f'Protected input changed: {p}'
    report(coverage)
    write_json(REPORT/'verification.json', dict(status='passed', raw_files_unchanged=len(raw_hashes),
        v3_and_prior_result_files_unchanged=len(protected), input_rows=len(expected_keys), output_rows_per_profile=len(lineage),
        all_source_rows_mapped_exactly_once=True, physical_cells_in_one_group=True,
        completed_bundles=2*len(BUNDLES), no_model_runs=True,
        implementation_sha256={p:sha256(ROOT/p) for p in SOURCES},
        upstream={name:dict(url='https://raw.githubusercontent.com/'+url, sha256=sha256(REPORT/'upstream_sources'/name)) for name,url in UPSTREAM.items()},
        artifact_sha256={p.relative_to(REPORT).as_posix():sha256(p) for p in REPORT.rglob('*') if p.is_file() and p.name!='verification.json'}))


def report(coverage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples = [('calce_cs2_1c','CS2_35'), ('mit_batch1','b1c0'), ('mit_batch2','b2c12'), ('xjtu_2c','batch1_2C_battery-1')]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (group, b) in zip(axes.flat, examples):
        for profile, color, label in [('forecast_safe','#4c78a8','Forecast-safe v4'), ('retrospective','#e45756','Retrospective v4')]:
            d, _, _ = load_v4_bundle(OUTPUT/profile/group, allow_retrospective=True)
            r=d[d.battery_id.eq(b)]
            ax.plot(r.cycle, r.soh.where(r.quality_eligible), color=color, lw=1, label=label)
        ax.set(title=f'{group} | {b}', xlabel='Original record / cycle coordinate', ylabel='SOH (Q / fixed Q_ref)')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('V4 eligible measurements only; no smoothing, no cycle compression', fontsize=13)
    fig.tight_layout(); fig.savefig(REPORT/'eligible_trajectories.png', dpi=160);plt.close(fig)
    totals=coverage.groupby('profile')[['rows','eligible','newly_excluded']].sum()
    table=pd.read_csv(REPORT/'dataset_coverage.csv')
    lines=['# BatteryML / BatteryLife 규칙을 반영한 SOH 전처리 v4', '',
        '원본·v3·기존 모델 결과는 보존했다. 새 모델 학습이나 성능 재평가는 실행하지 않았다.', '',
        '## 사용할 데이터', '',
        '- 기본 예측용: `data/processed/reviewed_v4/forecast_safe/<group>/`. 현재 cycle의 측정 정보와 고정된 문헌 예외만 사용한다.',
        '- 사후 정제 비교용: `data/processed/reviewed_v4/retrospective/<group>/`. CALCE 중앙값 필터와 MIT 미도달 셀 선택까지 적용한다. 미래 정보를 이용하므로 인과적 예측 성능으로 보고하지 않는다.',
        '- `load_v4_bundle`은 retrospective 데이터를 명시적으로 허용하지 않으면 거부한다. 기존 C2 실행기는 고정된 v3를 계속 사용하며 v4 결과로 간주하면 안 된다.', '',
        '## 실제 적용', '',
        '- CALCE: 실제 날짜와 시험시간 순서를 확인하고 각 cycle에서 음의 전류를 우측 직사각형 적분해 방전량을 재계산했다. 기존 중복 export 제거와 원본 cycle 매핑은 유지했다. Q_ref도 같은 원래 기준 cycle의 재계산 용량으로 갱신했다.',
        '- CALCE retrospective: scipy.signal.medfilt(Q,21)의 0 padding과 전체 residual 중앙값을 포함한 원래 `<3s`, `Q>0.1Ah` 규칙을 그대로 적용했다. s=0이면 전부 제외되는 원래 규칙도 임의 보정하지 않는다. 기존 원시파형 QC 제외와 교집합을 사용한다.',
        '- MIT: b2c7/8/9/15/16을 b1c0/1/2/3/4로 합쳤다. 원시 인덱스 0은 합친 셀의 맨 처음에만 적용하며 continuation의 첫 관측은 추가 삭제하지 않는다.',
        '- MIT: b1c0:11, b2c12:252, b2c44:247, b1c18:39의 원시 0-based 인덱스를 제외했다. 이후 관측을 앞으로 당기지 않는다. Batch 3 지정 6셀 제외도 명시적으로 반영한다.',
        '- MIT: 80% 미도달 지정 5셀은 예측용에서 유지하며 retrospective에서만 제외한다. 미도달이 실제 용량 측정 오류를 뜻하지 않기 때문이다.',
        '- XJTU: 현재 보유·선정 범위는 Batch 1(2C 충전) / Batch 2(3C 충전) 23셀이다. ±0.01C로 원본 용량을 분리하고 RPT/Cycling, 실제 방전 전류·종료 전압을 검증했다. 초기 RPT는 입력/정답에서 제외하고 1C, 2.5V 운전 관측만 허용했다.',
        '- XJTU: 시간 리셋·중복은 샘플 순서를 유지하는 단조 elapsed 좌표로 진단했다. 이 좌표는 실제 시간 공백을 복원한 것이 아니며, 용량 적분이나 수명 정답에 사용하지 않는다. 원시 파형은 그대로 보존했다.',
        '- NASA: v3의 용량·SOH·적격성은 그대로 유지했다.', '',
        '## 의도적으로 유지한 정의', '',
        '- 제외 관측은 감사용 행으로 남기고 quality_eligible=False로 표시한다. 그래프와 평가에서는 숨기며, 시간축을 압축하거나 보간·평활화하지 않는다.',
        '- SOH 기본값은 기존 Q/Q_ref 정의이다. CALCE만 용량 계산 방식에 맞춰 같은 기준 cycle의 Q_ref를 재계산했다. MIT와 XJTU는 고정된 초기 교정값을 유지한다. XJTU 초기 RPT는 정규화 기준으로만 남고 운전 이력에 섞이지 않는다.',
        '- 별도 soh_nominal 열은 정격용량으로 나눈 값이다. 80% EOL을 정할 때 soh와 soh_nominal을 혼동하면 안 된다. 새 EOL 라벨은 이번에 만들지 않았다.',
        '- BatteryLife XJTU 수명 코드의 마지막 교차·보간·꼬리 외삽은 도입하지 않았다. 현재 프로젝트의 최초 실측 교차 정의와 다르다.',
        '- CALCE CX 계열, MIT Batch 4, XJTU Batch 3–6은 현재 선택된 원본에 없어 적용하지 않았다.', '',
        '## 커버리지', '', '| 프로필 | 그룹 | 셀(적격 셀) | 전체 행 | 적격 관측 | 추가 제외 | 용량 변경 |', '|---|---|---:|---:|---:|---:|---:|']
    for r in table.itertuples():
        lines.append(f'| {r.profile} | {r.group} | {r.cells} ({r.eligible_cells}) | {r.rows} | {r.eligible} | {r.newly_excluded} | {r.capacity_changes} |')
    lines += ['', '추가 제외는 합친 뒤 동일 원본 행에 대한 v3 적격성 대비이다. 용량 변경은 절대 차이가 1e-12 Ah를 넘는 행 수이다.', '',
        '![적격 관측만 표시](eligible_trajectories.png)', '',
        '[셀별 집계](battery_coverage.csv) · [CALCE 필터 진단](calce_filter_diagnostics.csv) · [검증](verification.json)', '',
        '재현: `.venv/Scripts/python.exe -X utf8 scripts/build_preprocessing_v4.py --output <새 출력 경로> --report <새 보고서 경로>`.',
        '완료된 출력은 덮어쓰지 않는다. evidence 캐시는 원본과 추출기 해시가 같을 때만 재사용한다.']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    global OUTPUT, REPORT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--extract-only', action='store_true')
    args=parser.parse_args()
    OUTPUT, REPORT = args.output.resolve(), args.report.resolve()
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1 to 4 workers')
    if (REPORT/'verification.json').exists() or OUTPUT.exists():
        raise ValueError('Preserve completed output; choose new output AND report paths')
    REPORT.mkdir(parents=True, exist_ok=True)
    # Fetch source text for provenance only. Never execute upstream scripts.
    from urllib.request import urlopen
    sources=REPORT/'upstream_sources';sources.mkdir(exist_ok=True)
    for name,url in UPSTREAM.items():
        if not (sources/name).exists():
            (sources/name).write_bytes(urlopen('https://raw.githubusercontent.com/'+url, timeout=30).read())
    calce, xjtu = extract(args.workers)
    if not args.extract_only:
        build(calce, xjtu)


if __name__ == '__main__':
    main()
