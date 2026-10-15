"""Independent saved-table audit of the v4 release, including raw spot checks."""
from pathlib import Path
import json
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_data import load_bundle, sha256, write_json

REPORT = ROOT/'results/preprocessing_batterylife_v4'
DATA = ROOT/'data/processed/reviewed_v4'


def main():
    verification=json.loads((REPORT/'verification.json').read_text())
    assert verification['status']=='passed'
    for name, expected in verification['artifact_sha256'].items():
        assert sha256(REPORT/name)==expected, name
    for name, expected in verification['implementation_sha256'].items():
        assert sha256(ROOT/name)==expected, name
    protected=json.loads((REPORT/'protected_before.json').read_text())
    for name,expected in protected.items():
        assert sha256(ROOT/name)==expected, name
    for family in ('calce','xjtu'):
        sources=json.loads((REPORT/'evidence'/f'{family}.json').read_text())
        for name,expected in sources['raw_sha256'].items():
            assert sha256(ROOT/name)==expected, name
    safe={p.name:load_bundle(p) for p in (DATA/'forecast_safe').iterdir() if p.is_dir()}
    retrospective={p.name:load_bundle(p) for p in (DATA/'retrospective').iterdir() if p.is_dir()}
    for group,(d,m,manifest) in safe.items():
        other=retrospective[group][0]
        pd.testing.assert_frame_equal(d[['battery_id','cycle']],other[['battery_id','cycle']])
        np.testing.assert_allclose(d.capacity_ah,other.capacity_ah,rtol=0,atol=0)
        assert not ((~d.eligible_v3) & d.quality_eligible).any()
        assert not ((~d.quality_eligible) & other.quality_eligible).any()
        assert not any(c.startswith('retrospective_') for c in d.columns)
        for b, r in d.groupby('battery_id'):
            assert (np.diff(r.cycle)==1).all(), (group,b)
        for name,expected in manifest['source_files'].items():
            assert sha256(ROOT/name)==expected, name
    nasa=load_bundle(ROOT/'data/processed/reviewed_native_v3/nasa_aging')[0]
    pd.testing.assert_frame_equal(safe['nasa_aging'][0][nasa.columns],nasa)

    # Independent scalar integrations on one complete acquisition file per cell.
    evidence=pd.read_csv(REPORT/'evidence/calce.csv.gz')
    raw_integral_checks=0
    for cell in ['CS2_35','CS2_36','CS2_37','CS2_38']:
        filename=sorted(evidence[evidence.source_file.str.startswith(cell+'_')].source_file.unique())[0]
        path=ROOT/'data/raw/Calce'/cell/filename
        book=pd.ExcelFile(path)
        frame=book.parse([s for s in book.sheet_names if s.startswith('Channel')][0])
        frame=frame.sort_values(['Date_Time','Test_Time(s)'],kind='stable')
        for source_cycle, rows in frame.groupby('Cycle_Index',sort=False):
            q=0.
            current=rows['Current(A)'].to_numpy()
            seconds=rows['Test_Time(s)'].to_numpy()
            for j in range(1,len(rows)):
                if current[j]<0:
                    q-=float(current[j])*float(seconds[j]-seconds[j-1])/3600.
            e=evidence[evidence.source_file.eq(filename)&evidence.source_cycle.eq(source_cycle)]
            assert len(e)==1
            np.testing.assert_allclose(q,e.integrated_discharge_ah.iloc[0],rtol=1e-12,atol=1e-12)
            raw_integral_checks+=1
        book.close()

    # Recompute centered decisions without scipy.signal.medfilt or v4 helpers.
    calce=safe['calce_cs2_1c'][0]
    offline=retrospective['calce_cs2_1c'][0]
    calce_decisions=0
    for battery, rows in calce.groupby('battery_id',sort=False):
        q=rows.capacity_ah.to_numpy()
        padded=np.pad(q,(10,10),constant_values=0.)
        med=np.array([np.median(padded[k:k+21]) for k in range(len(q))])
        residual=np.abs(q-med)
        limit=3*float(np.median(residual))
        keep=np.array([r<limit and value>.1 for r,value in zip(residual,q)])
        np.testing.assert_array_equal(offline[offline.battery_id.eq(battery)].quality_eligible,
                                      rows.quality_eligible.to_numpy()&keep)
        calce_decisions+=len(q)

    mit=pd.concat([v[0] for k,v in safe.items() if k.startswith('mit_')],ignore_index=True)
    mapping={'b2c7':'b1c0','b2c8':'b1c1','b2c9':'b1c2','b2c15':'b1c3','b2c16':'b1c4'}
    for child,parent in mapping.items():
        part=mit[mit.source_battery_id.eq(child)]
        assert len(part)>0 and part.battery_id.eq(parent).all()
        assert not mit.battery_id.eq(child).any()
        initial=part[part.source_raw_cycle_index.eq(0)]
        assert len(initial)==1
        assert not initial.exclusion_reasons.str.contains('initial_raw_cycle_0').any()
    named_raw_checks=0
    exceptions={'b1c0':11,'b2c12':252,'b2c44':247,'b1c18':39}
    for battery,index in exceptions.items():
        row=mit[mit.source_battery_id.eq(battery)&mit.source_raw_cycle_index.eq(index)].iloc[0]
        assert not row.quality_eligible
        assert 'batterylife_named_cycle_exception' in row.exclusion_reasons
        batch=int(battery[1]);cell=int(battery.split('c')[1])
        with h5py.File(ROOT/f'data/raw/MIT/MIT_Batch_{batch}.mat','r') as f:
            b=f['batch'];s=f[b['summary'][cell,0]]
            assert float(s['cycle'][()].ravel()[index])==row.source_cycle
            np.testing.assert_allclose(float(s['QDischarge'][()].ravel()[index]),row.capacity_ah,atol=1e-12)
        named_raw_checks+=1
    xjtu=pd.concat([v[0] for k,v in safe.items() if k.startswith('xjtu_')],ignore_index=True)
    valid=xjtu[xjtu.quality_eligible]
    assert valid.attribute.eq('Cycling').all()
    assert np.isclose(valid.measured_discharge_rate_c,1.,atol=.1,rtol=0).all()
    assert np.isclose(valid.measured_discharge_cutoff_v,2.5,atol=.05,rtol=0).all()
    assert not xjtu[xjtu.cycle.eq(1)].quality_eligible.any()
    np.testing.assert_allclose(xjtu.capacity_ah,xjtu.capacity_v3_ah,rtol=0,atol=0)
    checks=dict(status='passed', protected_file_hashes_checked=len(protected),
        bundles_checked=len(safe)+len(retrospective), calce_raw_integral_cycles=raw_integral_checks,
        calce_independent_filter_decisions=calce_decisions, mit_raw_exception_indices=named_raw_checks,
        mit_merged_continuations=5, xjtu_eligible_same_condition=len(valid), nasa_exactly_unchanged=True,
        verifier_sha256=sha256(Path(__file__)))
    write_json(REPORT/'independent_verification.json',checks)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
