"""Independent local re-score and annotated horizon diagnostic figure."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts import run_c2_rul_context_sweep as run
from src.c2_rul import eligible_mask

def main():
    plan,data,audit,prepared=run.prepare()
    result=run.verify(plan,prepared,check_weights=False)
    out=run.OUT
    saved=pd.read_csv(out/'horizon_diagnostics.csv')
    scores=pd.read_csv(out/'rul_predictions.csv')
    summary=pd.read_csv(out/'summary.csv')
    count=0
    with np.load(out/'forecast_quantiles.npz') as arrays:
        for length,(tasks,_) in prepared.items():
            for method,key in [('zero-shot','zero'),('head-only','head')]:
                for i,task in enumerate(tasks):
                    raw=data[data.battery_id.eq(task['battery_id'])].set_index('cycle')
                    target=raw.soh.where(eligible_mask(raw)).reindex(np.arange(65,321)).to_numpy(copy=True)
                    if task['true_rul'] is not None:target[int(task['true_rul']):]=np.nan
                    pred=arrays[f'L{length}_{key}'][i,:,4]
                    sub=saved[saved.context_length.eq(length)&saved.method.eq(method)&saved.battery_id.eq(task['battery_id'])]
                    for _,r in sub.iterrows():
                        sl=slice(int(r.start_step)-1,int(r.end_step));valid=np.isfinite(target[sl]);e=pred[sl][valid]-target[sl][valid]
                        assert r.n_scored==len(e)
                        metrics=[100*abs(e).mean(),100*np.sqrt(np.mean(e**2)),100*e.mean()] if len(e) else [np.nan]*3
                        np.testing.assert_allclose(r[['mae_soh_pp','rmse_soh_pp','bias_soh_pp']].to_numpy(dtype=float),metrics,rtol=1e-9,atol=1e-10,equal_nan=True)
                        count+=1
                part=scores[scores.context_length.eq(length)&scores.method.eq(method)]
                events=part[part.event_observed];finite=events[events.predicted_crossing]
                row=summary[summary.context_length.eq(length)&summary.method.eq(method)].iloc[0]
                assert row.n_finite_event_predictions==len(finite) and row.n_no_crossing_event_predictions==3-len(finite)
                np.testing.assert_allclose(row.conditional_mae_cycles,finite.error_cycles.abs().mean(),rtol=1e-10)
                np.testing.assert_allclose(row.mae_lower_bound_all_event_origins,events.absolute_error_lower_bound_cycles.mean(),rtol=1e-10)
    result.update(independent_horizon_metric_rows=count,independent_summary_recalculation=True,
                  checkpoint_note='12 file hashes and reload parity checked on desktop; weights retained there')
    run.write_json(out/'local_release_verification.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bands=saved[saved.kind.eq('band')]
    n=bands[bands.context_length.eq(64)&bands.method.eq('zero-shot')].groupby('end_step').n_scored.agg(lambda v:int(v.gt(0).sum())).reindex([16,32,64,128,256])
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    labels=[f'{band}\n{int(c)} cells' for band,c in zip(['1–16','17–32','33–64','65–128','129–256'],n)]
    for ax,method in zip(axes,['zero-shot','head-only']):
        for length,color in [(16,'#d67c23'),(32,'#2467ad'),(64,'#168575')]:
            s=bands[bands.context_length.eq(length)&bands.method.eq(method)].groupby('end_step').mae_soh_pp.mean().reindex([16,32,64,128,256])
            ax.plot(range(5),s,marker='o',label=f'L={length}',color=color)
        ax.set_xticks(range(5),labels);ax.set(title=method,xlabel='Forecast distance band / cells with observed labels')
        ax.grid(alpha=.2);ax.legend(frameon=False);ax.axvspan(2.5,4.4,color='#999',alpha=.08)
    axes[0].set_ylabel('Mean cell MAE (SOH percentage points)')
    fig.suptitle('Matched cells across input lengths; cell composition changes across distance bands')
    fig.tight_layout(rect=(0,.08,1,1))
    fig.text(.5,.025,'65–128: B0007 only (40 labels at h65–104). 129–256: no labels. Do not infer whole-cohort error growth.',ha='center',fontsize=9)
    fig.savefig(out/'horizon_errors_annotated.png',dpi=170);plt.close(fig)
    print(result)

if __name__=='__main__':main()
