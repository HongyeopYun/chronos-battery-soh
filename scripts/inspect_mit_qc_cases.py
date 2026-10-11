"""Inspect every non-collection MIT exclusion, separating waveform failure modes."""
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.soh_data import load_bundle
from src.strict_preprocessing import runs


def main():
    out=[]
    for number in (1,2,3):
        group=f'mit_batch{number}'
        data,meta,_=load_bundle(ROOT/'data/processed/strict_v2'/group)
        candidates=data[~data.quality_eligible & ~data.exclusion_reasons.str.contains('known_collection_issue|invalid_capacity')]
        with h5py.File(ROOT/f'data/raw/MIT/MIT_Batch_{number}.mat','r') as f:
            for m in meta.itertuples():
                selected=candidates[candidates.battery_id.eq(m.battery_id)]
                if selected.empty:
                    continue
                cy=f[f['batch']['cycles'][int(m.source_cell_index_zero_based),0]]
                # Batch 2 continuations keep the original source-cycle index.
                for r in selected.itertuples():
                    idx=int(r.source_cycle)-1
                    i,v,t,q=(f[cy[k][idx,0]][()].ravel() for k in ('I','V','t','Qd'))
                    extreme=(v<.5)|(v>3.9)
                    dis=i<-.005
                    spans=[(a,b) for a,b in runs(dis) if np.min(i[a:b])<-.1]
                    if not spans:
                        continue
                    a,b=max(spans,key=lambda s:s[1]-s[0])
                    dt=np.diff(t[a:b])*60
                    incr=-.5*(i[a+1:b]+i[a:b-1])*1.1*dt/3600
                    med=float(np.median(dt[dt>0])) if (dt>0).any() else 0.
                    gap=dt>max(60,10*med)
                    startq=q[a-1] if a else q[a]
                    out.append(dict(group=group,battery_id=r.battery_id,cycle=r.cycle,source_cycle=r.source_cycle,
                        reasons=r.exclusion_reasons,native_q=r.capacity_ah,
                        counter_max=float(q.max()),counter_principal_increment=float(q[b-1]-startq),
                        principal_integral=float(incr.sum()),large_gap_integral=float(incr[gap].sum()),
                        large_gap_count=int(gap.sum()),max_gap_s=float(dt.max()) if len(dt) else 0.,
                        voltage_extreme_samples=int(extreme.sum()),
                        voltage_extreme_discharge_samples=int((extreme&dis).sum()),
                        voltage_extreme_charge_samples=int((extreme&(i>.005)).sum()),
                        voltage_max=float(v.max()),voltage_min=float(v.min()),
                        principal_discharge_voltage_min=float(v[a:b].min()),principal_discharge_voltage_max=float(v[a:b].max()),
                        number_discharge_runs=len(spans)))
        print(group,len(candidates),flush=True)
    pd.DataFrame(out).to_csv(ROOT/'results/preprocessing_review_all/mit_exclusion_cases.csv.gz',index=False)


if __name__=='__main__':
    main()
