"""Independent raw-data and coverage audit. Never changes data or runs models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calce_windows import observation_segments
from src.soh_data import EXCLUDED_FLAGS, load_bundle, observed_windows, sha256, write_json
from src.strict_preprocessing import BUNDLES, RAW_FOLDERS

OUT = ROOT / 'results/preprocessing_review_all'


def inputs(group):
    data, meta, manifest = load_bundle(ROOT / 'data/processed' / BUNDLES[group])
    strict, _, _ = load_bundle(ROOT / 'data/processed/strict_v2' / group)
    evidence = ROOT / 'results/preprocessing_strict_v2/evidence' / f'{group}_raw_features.csv.gz'
    evidence_manifest = json.loads(evidence.with_name(f'{group}_raw_manifest.json').read_text())
    assert sha256(evidence) == evidence_manifest['feature_sha256']
    assert sha256(ROOT / 'data/processed' / BUNDLES[group] / 'manifest.json') == evidence_manifest['parent_manifest_sha256']
    features = pd.read_csv(evidence)
    return data, meta, manifest, strict, features


def usable(data):
    return (data.is_observed & data.capacity_ah.gt(0) & np.isfinite(data.soh)
            & data.capacity_kind.isin(['full_discharge', 'reference_test'])
            & ~data.quality_flags.map(lambda x: bool(set(str(x).split(';')) & EXCLUDED_FLAGS)))


def basic():
    files, cells, reasons, joins = [], [], [], []
    references = {}
    for group in BUNDLES:
        data, meta, manifest, strict, features = inputs(group)
        dataset = data.dataset.iloc[0]
        for name, expected in manifest['source_files'].items():
            actual = sha256(ROOT / 'data/raw' / RAW_FOLDERS[dataset] / name)
            assert actual == expected, (group, name)
            files.append(dict(group=group, source_file=name, sha256=actual, unchanged=True))
        pd.testing.assert_frame_equal(data[['battery_id', 'cycle']], strict[['battery_id', 'cycle']])
        np.testing.assert_allclose(data[['capacity_ah', 'soh']], strict[['capacity_ah', 'soh']], rtol=1e-14, atol=1e-14, equal_nan=True)
        assert not features.duplicated(['battery_id', 'cycle']).any()
        matched = data.merge(features, on=['battery_id', 'cycle'], suffixes=('', '_trace'), validate='one_to_one')
        assert len(matched) == len(data)
        finite = np.isfinite(matched.capacity_ah) & np.isfinite(matched.raw_capacity_ah)
        max_q_error = float(abs(matched.loc[finite, 'capacity_ah'] - matched.loc[finite, 'raw_capacity_ah']).max())
        # CALCE QC extends the discharge to -0.005 A, while native extraction
        # uses -0.1 A. Keep this sub-micro-Ah tail difference visible; the
        # independent raw-counter scan below checks extraction to 1e-12 Ah.
        assert max_q_error < (1e-6 if dataset == 'calce' else 1e-10), (group, max_q_error)
        for m in meta.itertuples():
            references[m.battery_id] = m.reference_capacity_ah
            before = data[data.battery_id.eq(m.battery_id)].copy()
            after = strict[strict.battery_id.eq(m.battery_id)].copy()
            if dataset == 'calce':
                before['segment_id'] = observation_segments(before)
                starts = pd.to_datetime(before.source_start_time)
                ends = pd.to_datetime(before.source_end_time)
                for idx in range(1, len(before)):
                    a, b = before.iloc[idx-1], before.iloc[idx]
                    if a.source_file != b.source_file:
                        joins.append(dict(battery_id=m.battery_id, previous_cycle=int(a.cycle), next_cycle=int(b.cycle),
                                          elapsed_hours=(starts.iloc[idx]-ends.iloc[idx-1]).total_seconds()/3600,
                                          previous_source_cycle=a.source_cycle, next_source_cycle=b.source_cycle,
                                          previous_file=a.source_file, next_file=b.source_file))
            valid = usable(before)
            ref = before[before.cycle.eq(m.reference_cycle)]
            if not ref.empty:
                np.testing.assert_allclose(ref.capacity_ah.iloc[0], m.reference_capacity_ah, rtol=1e-12)
            elif getattr(m, 'continuation_of', ''):
                assert references[m.continuation_of] == m.reference_capacity_ah
            else:
                raise AssertionError(('Missing Qref source', m.battery_id))
            row = dict(group=group, battery_id=m.battery_id, rows=len(before), original_usable=int(valid.sum()),
                       strict_usable=int(after.quality_eligible.sum()), added_rejections=int((valid.to_numpy() & ~after.quality_eligible.to_numpy()).sum()),
                       quarantined=int(after.quality_status.eq('quarantined').sum()),
                       only_integral_unverified=int(after.exclusion_reasons.eq('capacity_integral_unverified').sum()),
                       qref=m.reference_capacity_ah, reference_cycle=m.reference_cycle,
                       reference_reasons=after[after.cycle.eq(m.reference_cycle)].exclusion_reasons.str.cat(sep=';'),
                       raw_capacity_max_error=max_q_error)
            for horizon in (1, 16):
                original = observed_windows(before, 64, horizon)['forecast_start']
                selected = observed_windows(after, 64, horizon)['forecast_start']
                assert set(selected) <= set(original)
                row[f'windows_64_{horizon}_original'] = len(original)
                row[f'windows_64_{horizon}_strict'] = len(selected)
                row[f'windows_64_{horizon}_lost'] = len(original)-len(selected)
                row[f'windows_64_{horizon}_coverage'] = len(selected)/len(original) if len(original) else np.nan
            cells.append(row)
        for reason, n in strict[~strict.quality_eligible].exclusion_reasons.str.split(';').explode().value_counts().items():
            reasons.append(dict(group=group, reason=reason, observations=int(n)))
        print('BASE', group, len(data), 'raw files', len(manifest['source_files']), flush=True)
    pd.DataFrame(files).to_csv(OUT/'raw_hashes.csv', index=False)
    table = pd.DataFrame(cells)
    table.to_csv(OUT/'battery_coverage.csv', index=False)
    sums = [c for c in table if c in ('rows','original_usable','strict_usable','added_rejections','quarantined','only_integral_unverified') or c.startswith('windows_') and not c.endswith('coverage')]
    table.groupby('group', sort=False)[sums].sum().to_csv(OUT/'dataset_coverage.csv')
    pd.DataFrame(reasons).to_csv(OUT/'exclusion_reasons.csv', index=False)
    pd.DataFrame(joins).to_csv(OUT/'calce_file_boundaries.csv', index=False)
    plan = json.loads((ROOT/'results/c2-battery/one-step-64/protocol.json').read_text())
    split_checks = []
    for group, spec in plan['groups'].items():
        _, meta, _, _, _ = inputs(group)
        physical = dict(zip(meta.battery_id, meta.physical_battery_id if 'physical_battery_id' in meta else meta.battery_id))
        for fold in spec['folds']:
            ids = {part: {physical[b] for b in fold[f'{part}_ids']} for part in ('train','validation','test')}
            assert not (ids['train'] & ids['validation'] or ids['train'] & ids['test'] or ids['validation'] & ids['test'])
            split_checks.append(dict(group=group, fold=fold['fold_id'], physical_cells_disjoint=True))
    write_json(OUT/'basic_verification.json', dict(raw_file_entries=len(files), unique_raw_files=len(set(r['sha256'] for r in files)),
               records=len(cells), observations=int(table.rows.sum()), original_values_unchanged=True,
               raw_summary_matches_processed=True, normalization_references_checked=True,
               split_checks=split_checks, scope='All selected 7 groups; folds evaluated separately, not a pooled cross-group training experiment.'))


def mit():
    records = []
    for group in ('mit_batch1','mit_batch2','mit_batch3'):
        data, meta, _, strict, _ = inputs(group)
        with h5py.File(ROOT/'data/raw/MIT'/meta.source_file.iloc[0], 'r') as f:
            batch = f['batch']
            for m in meta.itertuples():
                cell = int(m.source_cell_index_zero_based)
                cycles = f[batch['cycles'][cell,0]]
                summary = f[batch['summary'][cell,0]]
                q = summary['QDischarge'][()].ravel()
                source_cycles = summary['cycle'][()].ravel()
                rows = data[data.battery_id.eq(m.battery_id)]
                before = strict[strict.battery_id.eq(m.battery_id)]
                np.testing.assert_allclose(q, rows.capacity_ah, rtol=1e-14)
                np.testing.assert_array_equal(source_cycles, rows.source_cycle)
                refs = {k:cycles[k][()].ravel() for k in ('I','V','t','Qd')}
                assert all(len(v) == len(rows) for v in refs.values())
                for idx, (r, old) in enumerate(zip(rows.itertuples(), before.itertuples())):
                    d = {k:f[ref[idx]][()].ravel() for k, ref in refs.items()}
                    i,v,t,c = (d[k] for k in ('I','V','t','Qd'))
                    dt, dq = np.diff(t)/60, np.diff(c)
                    pair = (i[1:] < -.01) & (i[:-1] < -.01) & (dt > 0)
                    charge = i > .005
                    loads = i < -.1
                    cc = pair & (v[:-1] > 2.1) & (v[1:] > 2.1)
                    cv = pair & (v[:-1] < 2.05) & (v[1:] < 2.05)
                    integration = -.5*(i[1:]+i[:-1])*dt
                    scales = {}
                    for name, mask in (('cc',cc), ('cv',cv)):
                        good = mask & (dq > 0) & (integration > 1e-10)
                        scales[f'{name}_scale_median'] = float(np.median(dq[good]/integration[good])) if good.any() else np.nan
                        scales[f'{name}_counter_increment_ah'] = float(dq[mask].sum())
                        scales[f'{name}_integral_1p1_ah'] = float(1.1*integration[mask].sum())
                    records.append(dict(group=group,battery_id=r.battery_id,cycle=r.cycle,source_cycle=r.source_cycle,
                        known_collection_issue=bool(m.known_collection_issue),old_eligible=old.quality_eligible,
                        old_reasons=old.exclusion_reasons,native_q=q[idx],counter_max=float(c.max()),
                        native_counter_difference=float(q[idx]-c.max()), all_load_integral_1p1=float(1.1*integration[pair].sum()),
                        raw_discharge_runs=int(np.count_nonzero(np.diff(np.r_[False,loads,False].astype(int))==1)),
                        negative_counter_increments=int((dq[pair]<-1e-5).sum()),time_backsteps=int((np.diff(t)<-1e-6).sum()),
                        end_discharge_current_raw=float(i[np.flatnonzero(loads)[-1]]) if loads.any() else np.nan,
                        end_charge_current_raw=float(i[np.flatnonzero(charge)[-1]]) if charge.any() else np.nan,**scales))
                print('MIT RAW',m.battery_id,len(rows),flush=True)
        pd.DataFrame([r for r in records if r['group']==group]).to_csv(OUT/f'{group}_independent.csv.gz',index=False)
    table = pd.DataFrame(records)
    summary = []
    for (group, issue), d in table.groupby(['group','known_collection_issue']):
        qpos = d[d.native_q.gt(0)]
        summary.append(dict(group=group,known_collection_issue=bool(issue),observations=len(d),
             native_counter_mismatches=int(d.native_counter_difference.abs().gt(.005).sum()),
             cc_scale_median=float(qpos.cc_scale_median.median()),cv_scale_median=float(qpos.cv_scale_median.median()),
             cc_error_median=float((qpos.cc_integral_1p1_ah-qpos.cc_counter_increment_ah).median()),
             cv_error_median=float((qpos.cv_integral_1p1_ah-qpos.cv_counter_increment_ah).median()),
             only_integral_unverified=int(d.old_reasons.eq('capacity_integral_unverified').sum())))
    pd.DataFrame(summary).to_csv(OUT/'mit_phase_summary.csv',index=False)


def xjtu():
    output = []
    for group in ('xjtu_2c','xjtu_3c'):
        data,meta,_,strict,_ = inputs(group)
        for m in meta.itertuples():
            raw = loadmat(ROOT/'data/raw/XJTU'/m.source_file,simplify_cells=True)
            rows = data[data.battery_id.eq(m.battery_id)]
            old = strict[strict.battery_id.eq(m.battery_id)]
            q = np.asarray(raw['summary']['discharge_capacity_Ah']).ravel()
            np.testing.assert_allclose(q,rows.capacity_ah,rtol=1e-14)
            assert len(raw['data']) == len(rows)
            for r,o,c in zip(rows.itertuples(),old.itertuples(),raw['data']):
                i,v,t,counter = (np.asarray(c[k]).ravel() for k in ('current_A','voltage_V','relative_time_min','capacity_Ah'))
                neg=np.flatnonzero(i<-.1)
                pos=np.flatnonzero(i>.005)
                mask=(i[1:]<-.1)&(i[:-1]<-.1)
                increments=np.diff(t)/60
                output.append(dict(group=group,battery_id=r.battery_id,cycle=r.cycle,native_q=r.capacity_ah,
                    counter_last_discharge=float(counter[neg[-1]]) if len(neg) else np.nan,
                    integral_ah=float((-.5*(i[1:]+i[:-1])*np.maximum(increments,0))[mask].sum()),
                    discharge_runs=int(np.count_nonzero(np.diff(np.r_[False,i<-.1,False].astype(int))==1)),
                    charge_end_voltage=float(v[pos[-1]]) if len(pos) else np.nan,
                    charge_end_current=float(i[pos[-1]]) if len(pos) else np.nan,
                    discharge_end_voltage=float(v[neg[-1]]) if len(neg) else np.nan,
                    voltage_min=float(v.min()),voltage_max=float(v.max()),
                    discharge_time_backsteps=int((increments[mask]<-1e-6).sum()),
                    old_eligible=o.quality_eligible,old_reasons=o.exclusion_reasons))
            print('XJTU RAW',m.battery_id,len(rows),flush=True)
    pd.DataFrame(output).to_csv(OUT/'xjtu_independent.csv.gz',index=False)


def calce():
    data,meta,_,strict,_=inputs('calce_cs2_1c')
    output=[]
    for m in meta.itertuples():
        rows=data[data.battery_id.eq(m.battery_id)].copy()
        rows['export_file']=rows.source_file.str.split(';')
        for name,wanted in rows.explode('export_file').groupby('export_file',sort=False):
            with pd.ExcelFile(ROOT/'data/raw/Calce'/m.battery_id/name) as book:
                sheets=[s for s in book.sheet_names if s.startswith('Channel')]
                assert len(sheets)==1
                raw=pd.read_excel(book,sheet_name=sheets[0])
            bycycle=dict(tuple(raw.groupby('Cycle_Index',sort=False)))
            assert set(bycycle)==set(wanted.source_cycle)
            for r in wanted.itertuples():
                d=bycycle[r.source_cycle]
                assert pd.Timestamp(d.Date_Time.iloc[0])==pd.Timestamp(r.source_start_time)
                assert pd.Timestamp(d.Date_Time.iloc[-1])==pd.Timestamp(r.source_end_time)
                i,v,t,q=(d[k].to_numpy() for k in ('Current(A)','Voltage(V)','Test_Time(s)','Discharge_Capacity(Ah)'))
                neg=np.flatnonzero(i<-.1)
                preceding=d.iloc[:neg[0]] if len(neg) else d.iloc[:0]
                charging=preceding[preceding['Current(A)']>.005]
                # Detect a taper by measured I and V, independently of Step_Index=4.
                physical_cv=charging[charging['Voltage(V)'].between(4.15,4.25)&charging['Current(A)'].lt(.5)]
                countq=float(q[neg[-1]]-(q[neg[0]-1] if neg[0] else q[neg[0]])) if len(neg) else 0.
                np.testing.assert_allclose(countq,r.capacity_ah,rtol=0,atol=1e-12)
                mask=(i[1:]<-.1)&(i[:-1]<-.1)
                integral=float((-.5*(i[1:]+i[:-1])*np.maximum(np.diff(t),0))[mask].sum()/3600)
                if len(neg):
                    integral+=float(-i[neg[0]]*d['Step_Time(s)'].iloc[neg[0]]/3600)
                o=strict[strict.battery_id.eq(r.battery_id)&strict.cycle.eq(r.cycle)].iloc[0]
                output.append(dict(battery_id=r.battery_id,cycle=r.cycle,source_cycle=r.source_cycle,source_file=name,
                    native_q=r.capacity_ah,counter_q=countq,integral_ah=integral,
                    step4_samples=int(((preceding.Step_Index==4)&(preceding['Current(A)']>.005)).sum()),
                    physical_taper_samples=len(physical_cv),
                    charge_end_current=float(charging['Current(A)'].iloc[-1]) if len(charging) else np.nan,
                    charge_end_voltage=float(charging['Voltage(V)'].iloc[-1]) if len(charging) else np.nan,
                    discharge_end_voltage=float(v[neg[-1]]) if len(neg) else np.nan,
                    discharge_runs=int(np.count_nonzero(np.diff(np.r_[False,i<-.1,False].astype(int))==1)),
                    start_time=str(d.Date_Time.iloc[0]),end_time=str(d.Date_Time.iloc[-1]),
                    old_eligible=bool(o.quality_eligible),old_reasons=o.exclusion_reasons))
            print('CALCE RAW',m.battery_id,name,len(wanted),flush=True)
        pd.DataFrame(output).to_csv(OUT/'calce_independent_exports.csv.gz',index=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('basic','mit','xjtu','calce'),required=True)
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    globals()[args.stage]()
    write_json(OUT/f'{args.stage}_complete.json',dict(status='complete',stage=args.stage,script_sha256=sha256(Path(__file__))))


if __name__=='__main__':
    main()
