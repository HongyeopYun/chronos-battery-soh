"""Matched t64 RUL comparison and horizon-specific failure diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.c2_rul import eligible_mask,score_trajectory,summarize_rul
from scripts.run_c2_rul_pilot import forecasts_to_frame,write_json

def trajectory_diagnostics(task,q,frame,length,method):
    rows=[]
    for kind,bands in [('prefix',[(1,h) for h in [16,32,64,128,256]]),
                       ('band',[(1,16),(17,32),(33,64),(65,128),(129,256)])]:
        for lo,hi in bands:
            a=frame[frame.score_eligible & frame.horizon_step.between(lo,hi)]
            e=a.q50.to_numpy()-a.target_soh.to_numpy()
            rows.append(dict(context_length=length,method=method,battery_id=task['battery_id'],
                kind=kind,start_step=lo,end_step=hi,n_scored=len(a),
                mae_soh_pp=float(abs(e).mean()*100) if len(e) else np.nan,
                rmse_soh_pp=float(np.sqrt(np.mean(e**2))*100) if len(e) else np.nan,
                bias_soh_pp=float(e.mean()*100) if len(e) else np.nan))
    valid=frame[frame.score_eligible]
    slopes={}
    if len(valid)>1:
        for label,col in [('actual','target_soh'),('predicted','q50')]:
            slopes[label+'_slope_soh_pp_per_record']=float(np.polyfit(valid.horizon_step,valid[col],1)[0]*100)
    true=task['true_rul']
    diagnostic=dict(context_length=length,method=method,battery_id=task['battery_id'],
        n_scored=len(valid),threshold_soh=task['threshold_soh'],
        minimum_predicted_soh=float(q[:,4].min()),
        minimum_threshold_margin_pp=float((q[:,4].min()-task['threshold_soh'])*100),
        predicted_soh_h1=float(q[0,4]),predicted_soh_h256=float(q[-1,4]),
        predicted_drift_h1_to_h256_pp=float((q[-1,4]-q[0,4])*100),
        threshold_margin_at_true_eol_pp=float((q[true-1,4]-task['threshold_soh'])*100) if true is not None and 0<true<=256 else np.nan,
        quantile_crossing_fraction=float((np.diff(q,axis=-1)<0).any(axis=-1).mean()),
        invalid_interval_fraction=float((q[:,0]>q[:,-1]).mean()),**slopes)
    return rows,diagnostic

def build_report(out,plan,data,prepared,arrays):
    out=Path(out);all_scores=[];forecasts=[];errors=[];diagnostics=[];summaries=[]
    for length,(tasks,_) in prepared.items():
        for method,key in [('zero-shot','zero'),('head-only','head')]:
            values=arrays[f'L{length}_{key}']
            scores=pd.DataFrame([dict(context_length=length,method=method,**score_trajectory(t,q[:,4])) for t,q in zip(tasks,values)])
            all_scores.append(scores)
            f=forecasts_to_frame(tasks,values,data);f['context_length']=length;f['method']=method;forecasts.append(f)
            summaries.append(dict(context_length=length,method=method,**summarize_rul(scores)))
            for task,q in zip(tasks,values):
                rows,diag=trajectory_diagnostics(task,q,f[f.battery_id.eq(task['battery_id'])],length,method)
                errors.extend(rows);diagnostics.append(diag)
    scores=pd.concat(all_scores,ignore_index=True);pd.concat(forecasts,ignore_index=True).to_csv(out/'soh_predictions.csv.gz',index=False)
    scores.to_csv(out/'rul_predictions.csv',index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(errors).to_csv(out/'horizon_diagnostics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'trajectory_diagnostics.csv',index=False)
    # Read-only comparison against the old XPU experiment, never use it to fit or select heads.
    prior=Path(out).parents[2]/'results/c2-battery/rul-nasa-l64-h256-head-v1'
    old=pd.read_csv(prior/'rul_predictions.csv');old=old[old.origin_cycle.eq(64)]
    new=scores[scores.context_length.eq(64)]
    old[['battery_id','method','predicted_rul','error_cycles']].merge(
        new[['battery_id','method','predicted_rul','error_cycles']],on=['battery_id','method'],suffixes=('_old_xpu','_new_cuda')).to_csv(out/'backend_l64_comparison.csv',index=False)
    plots(out,data,prepared,arrays,pd.DataFrame(errors))
    fmt=lambda v:'N/A' if pd.isna(v) else f'{v:.2f}'
    lines=['# NASA RUL: 입력 16·32·64, 동일 t=64 비교','',
        '모든 조건은 같은 데스크톱 CUDA에서 새로 실행했다. 입력 길이만 바꾸고 미래 SOH 256개를 예측한다.','',
        '- NASA 4개 셀, 실제 EOL 관측 3개. B0007은 우측 검열이다.',
        '- EOL은 유효 용량 최초 1.4 Ah 이하. 미도달은 RUL > 256으로 유지한다.',
        '- 각 fold의 학습 2개 셀·검증 1개 셀·테스트 1개 셀은 서로 다르다. 모두 t=64이며 미래 학습 정답도 길이 간 완전히 같다.',
        '- Head-only는 LR 1e-6/1e-5 각 100 steps, seed 42. 검증 SOH 손실 및 분위수 역전 기준으로 step 0 포함 선택. 12개 head를 테스트 전에 모두 고정했다.',
        '- 16/32개 입력은 64개 이력의 마지막 위치를 잘랐다. t=16/32에서 시작한 실험이 아니다.',
        '- 미래 실측 재입력·보간·평활화·단조화·임의 외삽 없음. 제외 관측은 그림에서 숨겼다.',
        '- 장기 SOH 오차는 동일 H256 예측의 구간별 진단이다. H16/H32를 따로 생성한 결과가 아니다.',
        '- 실제 EOL 또는 관측 종료까지만 채점한다. 정답이 없는 구간은 N/A이며 0점으로 취급하지 않는다.',
        '- 조건부 MAE는 유한하게 EOL을 예측한 사례만의 점수다. 미도달 개수와 함께 해석해야 한다.',
        '- 테스트 결과를 이미 본 탐색 연구, 단일 seed, 적은 셀 수이므로 최적 길이 또는 범용 RUL 성능을 확정하지 않는다.','',
        '## 배터리별 RUL','',
        '| 셀 | 실제 RUL | 16 zero | 16 head | 32 zero | 32 head | 64 zero | 64 head |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for task in prepared[64][0]:
        b=task['battery_id'];truth=str(task['true_rul']) if task['true_rul'] is not None else f"> {task['observed_rul_lower_bound']}"
        vals=[]
        for length in [16,32,64]:
            for method in ['zero-shot','head-only']:
                v=scores[scores.battery_id.eq(b)&scores.context_length.eq(length)&scores.method.eq(method)].iloc[0].predicted_rul
                vals.append('> 256' if pd.isna(v) else str(int(v)))
        lines.append('| '+b+' | '+truth+' | '+' | '.join(vals)+' |')
    lines+=['','## 사건 관측 셀 3개 요약','','| 입력 | 모델 | 유한 예측 수 | 미도달 수 | 조건부 MAE | 전체 MAE 하한 |','|---:|---|---:|---:|---:|---:|']
    for _,r in summary.iterrows():
        lines.append(f'| {r.context_length} | {r.method} | {r.n_finite_event_predictions} | {r.n_no_crossing_event_predictions} | {fmt(r.conditional_mae_cycles)} | {fmt(r.mae_lower_bound_all_event_origins)} |')
    lines+=['','MAE 하한은 정확한 전체 MAE의 대체값이 아니다. B0007의 실제 RUL은 미관측이므로 별도 검열 모순 여부를 CSV에 기록한다.','',
        '![SOH 경로 비교](trajectories.png)','','![예측 거리별 오차](horizon_errors.png)','',
        '[RUL 원점수](rul_predictions.csv) · [구간별 오차](horizon_diagnostics.csv) · [기울기·임계값 진단](trajectory_diagnostics.csv) · [이전 XPU와 새 CUDA L64 비교](backend_l64_comparison.csv) · [검증](verification.json)','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')

def plots(out,data,prepared,arrays,errors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,4,figsize=(17,10),sharex=True,sharey='col')
    for row,length in enumerate([16,32,64]):
        for col,t in enumerate(prepared[length][0]):
            ax=axes[row,col];raw=data[data.battery_id.eq(t['battery_id'])].sort_values('cycle')
            ax.plot(raw.cycle,raw.soh.where(eligible_mask(raw)),color='black',lw=1.1,label='Eligible measurements')
            ax.axvspan(65-length,64,color='#9e9e9e',alpha=.14)
            for key,label,color in [('zero','zero-shot','#2467ad'),('head','head-only','#a64e93')]:
                ax.plot(np.arange(65,321),arrays[f'L{length}_{key}'][col,:,4],color=color,lw=1.3,label=label)
            ax.axhline(t['threshold_soh'],color='#555',ls='--',lw=.8,label='EOL threshold')
            ax.axvline(64,color='#999',ls=':',lw=.8)
            if t['true_eol_cycle'] is not None:ax.axvline(t['true_eol_cycle'],color='#bd6b2c',ls=':',lw=1)
            ax.set_title(f"{t['battery_id']} | L={length}",fontsize=11);ax.grid(alpha=.15)
            if col==0:ax.set_ylabel('SOH')
            if row==2:ax.set_xlabel('Discharge record ordinal')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.96),ncol=4,frameon=False)
    fig.suptitle('Same origin t=64, forecast H=256; all conditions on desktop CUDA',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(out/'trajectories.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,method in zip(axes,['zero-shot','head-only']):
        for length,color in [(16,'#d67c23'),(32,'#2467ad'),(64,'#168575')]:
            part=errors[errors.kind.eq('band')&errors.method.eq(method)&errors.context_length.eq(length)]
            g=part.groupby('end_step').mae_soh_pp.mean().reindex([16,32,64,128,256])
            ax.plot(range(5),g,marker='o',label=f'L={length}',color=color)
        ax.set_xticks(range(5),['1–16','17–32','33–64','65–128','129–256'],rotation=20)
        ax.set(title=method,xlabel='Forecast distance band');ax.grid(alpha=.2);ax.legend(frameon=False)
        ax.text(.98,.94,'129–256: no observed labels',transform=ax.transAxes,ha='right',va='top',fontsize=9)
    axes[0].set_ylabel('Mean cell MAE (SOH percentage points)')
    fig.suptitle('Error within each band, eligible labels through EOL/censoring only')
    fig.tight_layout();fig.savefig(out/'horizon_errors.png',dpi=170);plt.close(fig)
