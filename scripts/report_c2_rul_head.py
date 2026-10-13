"""Matched zero-shot/head-only RUL tables and held-out-cell figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.c2_rul import eligible_mask,score_trajectory,summarize_rul
from scripts.run_c2_rul_pilot import forecasts_to_frame,write_json


def number(value,digits=2):
    return f'{value:.{digits}f}' if value is not None and pd.notna(value) else 'N/A'


def build_report(out,plan,data,audit,tasks,zero,head,selected):
    out=Path(out);score_blocks=[];forecast_blocks=[];summaries={};cell_rows=[];prefix=[]
    for method,values in [('zero-shot',zero),('head-only',head)]:
        score=pd.DataFrame([dict(method=method,**score_trajectory(task,q[:,4])) for task,q in zip(tasks,values)])
        score_blocks.append(score)
        forecast=forecasts_to_frame(tasks,values,data);forecast['method']=method;forecast_blocks.append(forecast)
        primary=score[score.origin_cycle.eq(64)]
        summary=dict(primary_t64=summarize_rul(primary),secondary_pooled_origins=summarize_rul(score),
                     quantile_crossing_fraction=float((np.diff(values,axis=-1)<0).any(axis=-1).mean()),
                     q10_q90_inversion_fraction=float((values[...,0]>values[...,-1]).mean()))
        event=primary[primary.event_observed]
        r2=None
        if len(event)>1 and event.predicted_crossing.all():
            y=event.true_rul.to_numpy(dtype=float);e=event.predicted_rul.to_numpy(dtype=float)-y
            sst=np.square(y-y.mean()).sum()
            if sst>0:r2=float(1-np.square(e).sum()/sst)
        summary['primary_t64']['descriptive_cross_cell_r2']=r2
        cell=[]
        for battery,part in score.groupby('battery_id'):
            cell.append(dict(method=method,battery_id=battery,**summarize_rul(part)))
        cell_rows.extend(cell)
        event_cells=pd.DataFrame(cell).query('n_event_origins > 0')
        summary['secondary_equal_cell']={key:float(event_cells[key].mean()) if event_cells[key].notna().any() else None
                                         for key in ['conditional_mae_cycles','conditional_rmse_cycles','conditional_bias_cycles','mae_lower_bound_all_event_origins']}
        summaries[method]=summary
        for (battery,origin),block in forecast.groupby(['battery_id','origin_cycle']):
            for h in [16,32,64,128,256]:
                use=block[block.score_eligible & block.horizon_step.le(h)]
                y,p=use.target_soh.to_numpy(),use.q50.to_numpy();e=p-y
                sst=float(np.square(y-y.mean()).sum()) if len(y) else 0.
                prefix.append(dict(method=method,battery_id=battery,origin_cycle=origin,prefix_horizon=h,n_scored=len(y),
                    mae_soh_pp=float(abs(e).mean()*100) if len(e) else np.nan,
                    rmse_soh_pp=float(np.sqrt(np.square(e).mean())*100) if len(e) else np.nan,
                    r2=float(1-np.square(e).sum()/sst) if len(y)>1 and sst>0 else np.nan))
    scores=pd.concat(score_blocks,ignore_index=True);forecasts=pd.concat(forecast_blocks,ignore_index=True)
    scores.to_csv(out/'rul_predictions.csv',index=False)
    forecasts.to_csv(out/'soh_predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(cell_rows).to_csv(out/'battery_metrics.csv',index=False)
    pd.DataFrame(prefix).to_csv(out/'soh_prefix_metrics.csv',index=False)
    write_json(out/'summary.json',summaries)
    main=scores[scores.origin_cycle.eq(64)]
    comparisons=[]
    for battery,part in main.groupby('battery_id'):
        z,h=part.set_index('method').loc[['zero-shot','head-only']].itertuples(index=False)
        comparisons.append(dict(battery_id=battery,true_rul=z.true_rul,event_observed=z.event_observed,
                                true_rul_lower_bound=z.observed_rul_lower_bound,zero_rul=z.predicted_rul,head_rul=h.predicted_rul,
                                zero_error=z.error_cycles,head_error=h.error_cycles))
    comparison=pd.DataFrame(comparisons);comparison.to_csv(out/'primary_t64_comparison.csv',index=False)
    plots(out,data,audit,scores,forecasts)
    lines=['# NASA RUL: zero-shot / few-shot head-only 비교','',
           '같은 64개 입력 → 256개 미래 SOH → 1.4 Ah 최초 도달 시점으로 RUL을 계산한 NASA 4개 배터리 파일럿이다.',
           '각 테스트 배터리를 제외한 2개 배터리의 열화 경로로 학습하고, 별도 1개 배터리로 head를 선택했다.','',
           '## 학습과 평가 조건','',
           '- 모델: 동일한 원본 Chronos-2 revision, float32. 본체 고정, output_patch_embedding만 학습.',
           '- 각 source 배터리에서 t=64 한 창만 사용. 학습 2창·검증 1창. 학습 정답은 fold별 172 또는 208개 SOH 점이며 입력 SOH 128개도 관측 정보다.',
           '- NASA 기록은 t=64 이후 68~104개만 남는다. 실제로 관측된 정답만 사용하고 이후 256까지는 NaN으로 마스킹했다. 가용 기록에는 EOL 이후 측정도 포함된다. 미관측 미래값은 만들지 않았다.',
           '- RUL 정답을 학습하지 않는다. 미래 SOH의 native 21분위수 손실을 셀별 유효 정답 수로 정규화하여 학습했다.',
           '- fold 및 학습률별 원본에서 독립 초기화. LR 1e-6/1e-5, 각 100 steps, seed 42, 20 steps마다 검증. 두 학습 경로를 한 batch로 사용.',
           '- 검증 SOH loss로만 선택하며 검증 분위수 역전이 순정보다 늘어나면 제외한다. step 0도 후보다. 네 head를 모두 확정한 뒤 테스트 추론을 수행했다.',
           '- 테스트 입력·정답·EOL·17개 평가 시점은 이전 zero-shot과 동일하다. t=64가 주 평가이고 이후 겹치는 시점은 보조 평가다.',
           '- 검증은 미래 SOH 손실 기준이다. 검증 RUL이나 테스트 RUL로 head를 고르지 않았다.',
           '- 예측값 보정·평활화·단조화·추가 외삽 없음. excluded point는 그림에서 숨기고 원래 cycle 위치를 유지했다.',
           '- 미래 SOH 분위수는 시점별 구간이며 RUL 신뢰구간으로 해석하지 않는다. 미래 운전 조건을 별도로 입력하지 않는다.',
           '- 같은 노트북 XPU에서 수행했으며 cached frozen-encoder 계산을 native forward 및 loss와 대조했다. 완료 head 파일을 다시 읽어 검증 결과를 재현했다.','',
           '## 동일한 t=64에서의 RUL','',
           '| 배터리 | 실제 RUL | Zero-shot | Head-only | Zero 오차 | Head 오차 |',
           '|---|---:|---:|---:|---:|---:|']
    for row in comparison.itertuples():
        truth=number(row.true_rul,0) if row.event_observed else f'> {int(row.true_rul_lower_bound)}'
        z=number(row.zero_rul,0) if pd.notna(row.zero_rul) else '> 256'
        h=number(row.head_rul,0) if pd.notna(row.head_rul) else '> 256'
        lines.append(f'| {row.battery_id} | {truth} | {z} | {h} | {number(row.zero_error,0)} | {number(row.head_error,0)} |')
    lines+=['','단위: 방전 기록 수. 오차 = 예측−실제이며 양수는 늦은 EOL 예측이다. B0007은 실제 EOL이 관측되지 않아 정확한 오차를 계산하지 않는다.','',
            '| 주 평가 지표 | Zero-shot | Head-only |','|---|---:|---:|']
    for label,key in [('실제 EOL 관측 셀 수','n_event_batteries'),('유한 RUL 예측 수','n_finite_event_predictions'),
                      ('256 내 미도달 수','n_no_crossing_event_predictions'),('유한 사례 조건부 MAE','conditional_mae_cycles'),
                      ('유한 사례 조건부 RMSE','conditional_rmse_cycles'),('유한 사례 조건부 bias','conditional_bias_cycles'),
                      ('모든 사건 셀의 MAE 하한','mae_lower_bound_all_event_origins'),('기술적 셀 간 R²','descriptive_cross_cell_r2')]:
        lines.append('| '+label+' | '+' | '.join(number(summaries[m]['primary_t64'][key]) for m in ['zero-shot','head-only'])+' |')
    lines+=['','미도달이 있는 모델의 MAE/RMSE는 유한 예측 사례에 한정된다. 모델 간 유한 사례 수가 다르면 조건부 평균만 직접 비교하면 안 된다. MAE 하한도 실제 MAE의 대체값이 아니다. R²는 실제 EOL 관측 3개 셀 모두에서 유한한 예측이 있을 때만 제시한다.','',
            '![t64 예측 비교](trajectories_t64_comparison.png)','',
            '## 선택한 head','',
            '| 테스트 배터리 | 학습 배터리 | 검증 배터리 | LR | Step | 검증 loss 전 → 후 |',
            '|---|---|---|---:|---:|---|']
    for s in selected:
        lines.append(f"| {s['test_ids'][0]} | {', '.join(s['train_ids'])} | {s['validation_ids'][0]} | {s['learning_rate']:g} | {s['step']} | {s['baseline']['validation_loss']:.4f} → {s['validation_loss']:.4f} |")
    lines+=['','## 보조 평가와 한계','',
            '| 보조 지표 | Zero-shot | Head-only |','|---|---:|---:|']
    for label,key in [('실제 EOL 관측 배터리의 전체 평가 시점','n_event_origins'),('그중 256 내 미도달','n_no_crossing_event_predictions'),
                      ('B0007 검열 시점 수','censored_origins'),('B0007의 관측된 생존 기간과 모순되는 EOL 예측','censored_contradictions')]:
        lines.append('| '+label+' | '+' | '.join(number(summaries[m]['secondary_pooled_origins'][key],0) for m in ['zero-shot','head-only'])+' |')
    lines+=['','![시점별 RUL 비교](rul_by_origin_comparison.png)','',
            '- 전체 RUL 결과는 `rul_predictions.csv`, SOH 경로는 `soh_predictions.csv.gz`, 거리별 SOH 오차는 `soh_prefix_metrics.csv`에 있다.',
            '- SOH prefix 점수는 같은 256개 예측의 앞부분을 평가한 것으로, 별도 짧은 horizon 실행 결과와 다르다.',
            '- 실제 EOL 관측 3개 셀, 단일 seed, 검증 1개 셀에 한정된다. 일반적인 RUL 우위나 타 데이터셋 성능을 주장할 수 없다.',
            '- 이 연구는 이전 NASA 결과를 보고 설계한 탐색적 후속 실험이다. 별도 미사용 외부 테스트셋 검증은 아니다.',
            '- 학습 정답은 최대 104개 미래 관측까지만 존재한다. 256개 전체 열화 경로가 학습된 것으로 해석하지 않는다.',
            '- 다음 설정을 고르기 위해 이번 테스트 결과를 반복적으로 최적화하지 않았다. 기존 zero-shot 파일과 전처리·현재 SOH 실험은 보존했다.','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def plots(out,data,audit,scores,forecasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'zero-shot':('#1768b5','Zero-shot'),'head-only':('#9453b4','Head-only (held-out cell)')}
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for ax,(battery,raw) in zip(axes.flat,data.groupby('battery_id')):
        line=raw.sort_values('cycle');measured=np.where(eligible_mask(line),line.soh*100,np.nan)
        a=audit.set_index('battery_id').loc[battery]
        ax.plot(line.cycle,measured,color='#22272e',lw=1.7,label='Eligible measured SOH')
        for method,(color,label) in styles.items():
            p=forecasts[forecasts.method.eq(method)&forecasts.battery_id.eq(battery)&forecasts.origin_cycle.eq(64)]
            ax.plot(p.target_cycle,p.q50*100,color=color,lw=1.7,label=label)
        ax.axhline(a.threshold_soh*100,color='#c34b37',ls='--',lw=1.2,label='EOL = 1.4 Ah')
        ax.axvline(64,color='#888',ls=':',lw=1.2)
        if a.event_observed:ax.axvline(a.eol_cycle,color='#c34b37',ls=':',lw=1.2)
        primary=scores[scores.battery_id.eq(battery)&scores.origin_cycle.eq(64)].set_index('method')
        text=[]
        for method in styles:
            r=primary.loc[method];text.append(number(r.predicted_rul,0) if r.predicted_crossing else '>256')
        true=number(primary.iloc[0].true_rul,0) if a.event_observed else f'>{int(primary.iloc[0].observed_rul_lower_bound)}'
        ax.set_title(f'{battery} | true {true} | zero {text[0]} | head {text[1]}',fontsize=12)
        ax.set_xlabel('Discharge record index');ax.set_ylabel('SOH (% of first measured capacity)')
        ax.grid(alpha=.18);ax.set_xlim(1,320)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=4,fontsize=10)
    fig.suptitle('NASA RUL at t=64 | 64 past -> 256 future | zero-shot vs head-only\nTest cell excluded from head training and checkpoint selection',fontsize=14)
    fig.savefig(out/'trajectories_t64_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(battery,part) in zip(axes.flat,scores.groupby('battery_id')):
        truth=part[part.method.eq('zero-shot')]
        if truth.event_observed.all():ax.plot(truth.origin_cycle,truth.true_rul,'o-',color='#22272e',label='True RUL')
        else:ax.plot(truth.origin_cycle,truth.observed_rul_lower_bound,'o--',color='#22272e',label='True RUL strictly above bound')
        for method,(color,label) in styles.items():
            method_rows=part[part.method.eq(method)]
            # NaN breaks lines at non-hits; do not bridge them as finite predictions.
            ax.plot(method_rows.origin_cycle,method_rows.predicted_rul,'s-',color=color,label=label)
            miss=method_rows[~method_rows.predicted_crossing]
            if len(miss):ax.scatter(miss.origin_cycle,np.full(len(miss),256),marker='^',color=color,label=f'{method} >256')
        ax.set_title(battery);ax.set_xlabel('Prediction origin (discharge record index)');ax.set_ylabel('Remaining discharge records')
        ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle('Matched prediction origins | each forecast uses only observations through its origin',fontsize=12)
    fig.savefig(out/'rul_by_origin_comparison.png',dpi=160);plt.close(fig)
