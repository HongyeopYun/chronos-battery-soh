"""Build a separate v3 dataset from audited native capacities; never overwrite v2."""
from pathlib import Path
import argparse
import json
import sys
import h5py
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.soh_data import load_bundle,sha256,write_json,observed_windows
from src.strict_preprocessing import BUNDLES,RAW_FOLDERS
from src.calce_windows import observation_segments
from src.reviewed_soh import POLICY,mit_review_features,apply_reviewed_mit,masked_windows

OUT=ROOT/'data/processed/reviewed_native_v3'
REPORT=ROOT/'results/preprocessing_reviewed_native_v3'


def mit_features(raw, data,meta):
    records=[]
    with h5py.File(raw/meta.source_file.iloc[0],'r') as f:
        batch=f['batch']
        for m in meta.itertuples():
            cell=int(m.source_cell_index_zero_based)
            cycles=f[batch['cycles'][cell,0]]
            summary=f[batch['summary'][cell,0]]
            q=summary['QDischarge'][()].ravel()
            source=summary['cycle'][()].ravel()
            rows=data[data.battery_id.eq(m.battery_id)]
            np.testing.assert_array_equal(source,rows.source_cycle)
            np.testing.assert_allclose(q,rows.capacity_ah,rtol=1e-14)
            refs={k:cycles[k][()].ravel() for k in ('I','V','t','Qd')}
            if any(len(r)!=len(rows) for r in refs.values()):raise ValueError('Raw cycle alignment changed')
            for idx,row in enumerate(rows.itertuples()):
                arrays=[f[refs[k][idx]][()].ravel() for k in ('I','V','t','Qd')]
                evidence=mit_review_features(*arrays)
                evidence.update(battery_id=row.battery_id,cycle=int(row.cycle),source_cycle=int(row.source_cycle),raw_capacity_ah=float(q[idx]))
                records.append(evidence)
            print('REVIEW',m.battery_id,len(rows),flush=True)
    return pd.DataFrame(records)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--classify-cached',action='store_true',help='Preflight decision-only revision using checksummed existing raw features; forbidden after an experiment is locked')
    args=parser.parse_args()
    if args.classify_cached and (ROOT/'results/c2-battery/one-step-64-reviewed-v3/protocol.json').exists():
        raise ValueError('A model protocol is already locked; create a new dataset version')
    REPORT.mkdir(parents=True,exist_ok=True)
    write_json(REPORT/'policy.json',POLICY)
    implementation={name:sha256(ROOT/name) for name in ('src/reviewed_soh.py','src/strict_preprocessing.py','scripts/build_reviewed_soh.py')}
    summaries=[];changes=[]
    for group,path in BUNDLES.items():
        data,meta,source=load_bundle(ROOT/'data/processed'/path)
        old,_,_=load_bundle(ROOT/'data/processed/strict_v2'/group)
        raw=ROOT/'data/raw'/RAW_FOLDERS[data.dataset.iloc[0]]
        actual={name:sha256(raw/name) for name in source['source_files']}
        if actual!=source['source_files']:raise ValueError('Raw provenance changed')
        if group.startswith('mit_'):
            evidence=REPORT/f'{group}_raw_features.csv.gz'
            evidence_meta=REPORT/f'{group}_raw_features.json'
            expected=dict(raw_sha256=actual,implementation_sha256=implementation)
            if evidence.exists() and evidence_meta.exists():
                cached=json.loads(evidence_meta.read_text())
                if cached['raw_sha256']!=actual or cached['evidence_sha256']!=sha256(evidence):
                    raise ValueError('Raw data or extracted evidence changed')
                if cached['implementation_sha256']!=implementation and not args.classify_cached:
                    raise ValueError('Stale v3 evidence; use a new output version')
                features=pd.read_csv(evidence)
            else:
                features=mit_features(raw,data,meta)
                features.to_csv(evidence,index=False,float_format='%.17g',compression='gzip')
                write_json(evidence_meta,dict(**expected,evidence_sha256=sha256(evidence)))
            revised=apply_reviewed_mit(data,features)
        elif group=='nasa_aging':
            revised,_,_=load_bundle(ROOT/'data/processed/nasa_review_v3/nasa_aging')
        else:
            revised=old.copy()
        if group=='calce_cs2_1c':
            meta=meta.copy()
            meta['recording_boundaries']='chronological_observation_windows_may_cross_source_files; physical_missing_cycles_unknown'
        np.testing.assert_allclose(revised[['capacity_ah','soh']],data[['capacity_ah','soh']],rtol=1e-14,atol=1e-14,equal_nan=True)
        pd.testing.assert_frame_equal(revised[['battery_id','cycle']],data[['battery_id','cycle']])
        changed=old.quality_eligible.to_numpy()!=revised.quality_eligible.to_numpy()
        detail=revised.loc[changed,['battery_id','cycle','capacity_ah','exclusion_reasons','review_notes','quality_eligible']].copy()
        detail.insert(0,'group',group);detail['old_reasons']=old.loc[changed,'exclusion_reasons'].to_numpy()
        detail['old_eligible']=old.loc[changed,'quality_eligible'].to_numpy();changes.append(detail)
        audit=[]
        for b,d in revised.groupby('battery_id',sort=False):
            old_cell=old[old.battery_id.eq(b)]
            w=masked_windows(d)
            audit.append(dict(group=group,battery_id=b,rows=len(d),eligible_before=int(old_cell.quality_eligible.sum()),
                              eligible_after=int(d.quality_eligible.sum()),old_complete_64_1=len(observed_windows(old_cell,64,1)['target']),
                              revised_complete_64_1=len(observed_windows(d,64,1)['target']),
                              masked_64_1=len(w['target']),targets_with_missing_context=int((w['context_observed']<64).sum()),
                              min_context_observed=int(w['context_observed'].min()) if len(w['target']) else 0))
        audit=pd.DataFrame(audit);summaries.append(audit)
        dest=OUT/group;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'manifest.json').exists():
            previous=json.loads((dest/'manifest.json').read_text())
            if previous['implementation_sha256']!=implementation:
                if not args.classify_cached:raise ValueError('Do not overwrite a v3 bundle under changed code')
                backup=REPORT/'.runtime';backup.mkdir(exist_ok=True)
                write_json(backup/f'{group}_preflight_manifest.json',previous)
        for name,table in [('soh.csv',revised),('batteries.csv',meta),('audit.csv',audit)]:
            table.to_csv(dest/name,index=False,float_format='%.17g')
        write_json(dest/'schema.json',dict(schema_version=1,review_policy=POLICY,native_values_unchanged=True))
        manifest=dict(schema_version=1,status='complete',group=group,source_files=actual,n_rows=len(revised),
                      n_batteries=len(meta),review_policy=POLICY,implementation_sha256=implementation,
                      parent_manifest_sha256=sha256(ROOT/'data/processed'/path/'manifest.json'),
                      audit_sha256=sha256(ROOT/'results/preprocessing_review_all/verification.json'),
                      artifact_sha256={n:sha256(dest/n) for n in ('soh.csv','batteries.csv','audit.csv','schema.json')})
        if group.startswith('mit_'):manifest['raw_evidence_sha256']=sha256(evidence)
        write_json(dest/'manifest.json',manifest)
        reloaded,_,_=load_bundle(dest)
        np.testing.assert_allclose(reloaded.capacity_ah,data.capacity_ah,rtol=1e-14,equal_nan=True)
        print('COMPLETE',group,'labels',int(revised.quality_eligible.sum()),'targets',int(audit.masked_64_1.sum()),flush=True)
    table=pd.concat(summaries,ignore_index=True);table.to_csv(REPORT/'battery_coverage.csv',index=False)
    table.groupby('group',sort=False).sum(numeric_only=True).drop(columns='min_context_observed').to_csv(REPORT/'dataset_coverage.csv')
    pd.concat(changes,ignore_index=True).to_csv(REPORT/'reclassified_observations.csv',index=False)
    write_json(REPORT/'verification.json',dict(status='complete',policy=POLICY,raw_files_unchanged=True,native_q_soh_cycles_unchanged=True,
        implementation_sha256=implementation,artifacts={p.name:sha256(p) for p in REPORT.iterdir() if p.is_file() and p.name!='verification.json'}))


if __name__=='__main__':main()
