"""Audit-revised native-capacity QC and 64-position inputs with missingness.

No capacity replacement, interpolation, cycle compression, or score-based QC.
"""
import numpy as np
import pandas as pd

from src.soh_data import EXCLUDED_FLAGS, load_bundle
from src.strict_preprocessing import runs, trace_features, classify_trace

POLICY = {
    'name': 'reviewed_native_v3',
    'mit_capacity': 'Native QDischarge is primary. Integral disagreement alone is a review note, not label rejection. Summary/counter agreement is not an independent physical calibration.',
    'mit_trace': 'Aggregate discharge fragments; detect intervening charge and counter reset across gaps. Integrate only observed within-run intervals; never integrate long sampling gaps.',
    'other_datasets': 'NASA absent-charge-log review policy; CALCE and XJTU raw-QC exclusions retained.',
    'context': '64 original consecutive record positions. Rejected measurements are NaN inputs; target must be eligible. At least two finite inputs for the linear baseline. No filling or row compression.',
    'uncertainty': 'MIT CV export/current-counter mismatch unresolved; native capacity retained under an explicit measurement-source policy. CALCE unknown physical cycles between files remain unknown.',
}


def mit_review_features(current, voltage, minutes, counter):
    """Summarize all discharge fragments, without guessing activity in gaps."""
    i,v,t,c=[np.asarray(x,dtype=float).ravel() for x in (current,voltage,minutes,counter)]
    i=i*1.1; t=t*60
    f=trace_features(i,v,t,counter=c)
    if len({len(i),len(v),len(t),len(c)})!=1 or len(i)<3 or not all(np.isfinite(x).all() for x in (i,v,t,c)):
        return f
    spans=[(a,b) for a,b in runs(i<-.005) if i[a:b].min()<-.1]
    if not spans:
        return f
    start,end=spans[0][0],spans[-1][1]-1
    positions=np.concatenate([np.arange(a,b) for a,b in spans])
    pairs=np.concatenate([np.arange(a,b-1) for a,b in spans])
    delta=t[pairs+1]-t[pairs]
    median=float(np.median(delta[delta>0])) if (delta>0).any() else 0.
    gaps=delta>max(60.,10*median)
    observed=(delta>=0)&~gaps
    parts=-.5*(i[pairs+1]+i[pairs])*np.maximum(delta,0)/3600
    gap_between=sum(max(0.,t[b]-t[a-1]) for (_,a),(b,_) in zip(spans,spans[1:]))
    f.update(discharge_runs=len(spans),discharge_samples=len(positions),
             discharge_start_voltage_v=float(v[start]),discharge_end_voltage_v=float(v[end]),
             discharge_min_voltage_v=float(v[positions].min()),discharge_end_current_a=float(i[end]),
             discharge_cc_current_a=float(np.quantile(i[positions],.1)),
             discharge_time_backsteps=int((np.diff(t[start:end+1]) < -1e-6).sum()),
             discharge_counter_backsteps=int((np.diff(c[start:end+1]) < -1e-5).sum()),
             discharge_max_gap_s=float(delta.max()) if len(delta) else 0.,discharge_median_gap_s=median,
             discharge_integral_ah=float(parts[observed].sum()),
             integral_observed_only_ah=float(parts[observed].sum()),
             unobserved_interval_s=float(delta[gaps].sum()+gap_between),
             integral_long_gap_count=int(gaps.sum()),
             discharge_counter_ah=float(c.max()),
             discharge_counter_initial_ah=float(c[start]),
             intervening_charge_samples=int((i[start:end+1]>.005).sum()),
             current_scale_diagnostic_only=1.1)
    pos=np.flatnonzero(i[:start]>.005)
    f['charge_samples']=len(pos)
    f['charge_end_voltage_v']=float(v[pos[-1]]) if len(pos) else np.nan
    f['charge_end_current_a']=float(i[pos[-1]]) if len(pos) else np.nan
    return f


def reviewed_mit_decision(feature, cycle):
    bad,notes=classify_trace('mit',feature,cycle)
    if 'capacity_integral_unverified' in bad:
        bad.remove('capacity_integral_unverified')
        notes.append('native_capacity_retained_integral_not_calibrated')
    notes.append('mit_current_scale_not_verified_for_all_phases')
    if feature.get('integral_long_gap_count',0)>0:
        notes.append('long_sampling_gap_not_integrated')
    if feature.get('discharge_runs',0)>1:
        notes.append('multiple_discharge_fragments_reviewed')
    if feature.get('intervening_charge_samples',0)>0:
        bad.append('interleaved_charge_discharge')
    if feature.get('discharge_counter_initial_ah',0)>.005:
        # An unrecorded waveform prefix is not proof that the cumulative
        # native capacity is wrong; handle it like other missing trace spans.
        notes.append('discharge_prefix_unrecorded_native_capacity_retained')
    # Native-capacity preference never authorizes mismatched or absent counters.
    q=feature.get('raw_capacity_ah',np.nan)
    c=feature.get('discharge_counter_ah',np.nan)
    if np.isfinite(q) and q>0 and not np.isfinite(c):
        bad.append('native_counter_unverified')
    return sorted(set(bad)),sorted(set(notes))


def apply_reviewed_mit(data,features):
    aligned=data[['battery_id','cycle']].merge(features,on=['battery_id','cycle'],how='left',validate='one_to_one',indicator=True)
    if len(features)!=len(data) or not aligned._merge.eq('both').all():
        raise ValueError('Every MIT observation needs reviewed raw evidence')
    out=data.reset_index(drop=True).copy()
    reasons,notes=[],[]
    for row,f in zip(out.itertuples(),aligned.to_dict('records')):
        bad,review=reviewed_mit_decision(f,int(row.cycle))
        bad+=list(set(str(row.quality_flags).split(';'))&EXCLUDED_FLAGS)
        if not row.is_observed or row.capacity_kind not in ('full_discharge','reference_test'):
            bad.append('previously_unusable_measurement')
        if not np.isfinite(row.capacity_ah) or row.capacity_ah<=0:
            bad.append('invalid_capacity')
        if np.isfinite(f.get('raw_capacity_ah',np.nan)) and not np.isclose(row.capacity_ah,f['raw_capacity_ah'],rtol=1e-12,atol=1e-12):
            bad.append('extracted_capacity_mismatch')
        reasons.append(';'.join(sorted(set(bad))));notes.append(';'.join(review))
    out['exclusion_reasons']=reasons;out['review_notes']=notes
    out['quality_eligible']=out.exclusion_reasons.eq('')
    out['quality_status']=np.where(out.quality_eligible,'accepted_with_note','excluded')
    out['quality_flags']=[';'.join(filter(None,[str(flags),'strict_quality_excluded' if not ok else ''])) for flags,ok in zip(out.quality_flags,out.quality_eligible)]
    return out


def masked_windows(rows,context_length=64):
    """Use past positions t-64...t-1; valid y_t never appears in its input."""
    if len(rows.battery_id.unique())!=1 or rows.cycle.duplicated().any() or not rows.cycle.is_monotonic_increasing:
        raise ValueError('Expected one battery in original cycle order')
    if not rows.quality_eligible.isin([True,False]).all():
        raise ValueError('Quality eligibility must be boolean')
    valid=(rows.quality_eligible.to_numpy(dtype=bool)&rows.is_observed.to_numpy(dtype=bool)&
           np.isfinite(rows.soh)&rows.capacity_ah.gt(0)&rows.capacity_kind.isin(['full_discharge','reference_test'])).to_numpy().copy()
    valid &= np.array([not (set(str(x).split(';'))&EXCLUDED_FLAGS) for x in rows.quality_flags])
    shape=(0,context_length)
    if len(rows)<=context_length:
        return dict(context=np.empty(shape),context_cycle=np.empty(shape,dtype=int),target=np.empty((0,1)),
                    target_cycle=np.empty((0,1),dtype=int),forecast_start=np.empty(0,dtype=int),
                    context_observed=np.empty(0,dtype=int),last_observed_lag=np.empty(0,dtype=int))
    values=rows.soh.to_numpy(dtype=float)
    total=context_length+1
    c=np.lib.stride_tricks.sliding_window_view(rows.cycle.to_numpy(),total)
    v=np.lib.stride_tricks.sliding_window_view(np.where(valid,values,np.nan),total)
    n=np.isfinite(v[:,:-1]).sum(axis=1)
    selected=np.isfinite(v[:,-1])&(n>=2)&(np.diff(c,axis=1)==1).all(axis=1)
    if 'segment_id' in rows:
        s=np.lib.stride_tricks.sliding_window_view(rows.segment_id.to_numpy(),total)
        selected&=(s==s[:,:1]).all(axis=1)
    context=v[selected,:-1].copy()
    last=context_length-np.max(np.where(np.isfinite(context),np.arange(context_length),-1),axis=1)
    return dict(context=context,context_cycle=c[selected,:-1].copy(),target=v[selected,-1:].copy(),
                target_cycle=c[selected,-1:].copy(),forecast_start=c[selected,-1].copy(),
                context_observed=n[selected],last_observed_lag=last)


def prepare_reviewed_windows(bundle,config):
    if config.context_length!=64 or config.prediction_length!=1 or config.stride!=1 or config.battery_ids is not None or config.max_windows_per_battery is not None:
        raise ValueError('Reviewed run is full 64-position-to-1 only')
    data,meta,manifest=load_bundle(bundle)
    if manifest.get('review_policy')!=POLICY:
        raise ValueError('Expected locked reviewed-native policy')
    records,coverage=[],[]
    for index,m in enumerate(meta.to_dict('records'),1):
        rows=data[data.battery_id.eq(m['battery_id'])]
        w=masked_windows(rows)
        count=len(w['target'])
        record=dict(record_id=f'{index:04d}',dataset=m['dataset'],battery_id=m['battery_id'],
                    physical_battery_id=m.get('physical_battery_id') or m['battery_id'])
        coverage.append(dict(**record,n_rows=len(rows),eligible_windows=count,evaluated_windows=count,
                             status='ready' if count else 'skipped_no_eligible_windows',
                             min_context_observed=int(w['context_observed'].min()) if count else 0,
                             targets_with_missing_context=int((w['context_observed']<64).sum())))
        if count:records.append(dict(**record,windows=w))
    return records,pd.DataFrame(coverage),data,manifest
