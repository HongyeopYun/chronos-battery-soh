"""Independently re-score saved v4 predictions and render the final report."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.c2_v4_experiments import load_cells
from src.battery_zeroshot import sha256


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def r2(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    denominator=((y-y.mean())**2).sum()
    return float(1-((y-p)**2).sum()/denominator) if len(y)>1 and denominator>0 else np.nan


def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-8,equal_nan=True)


def verify(out,require_weights=True,persist=True):
    def save_table(frame,name):
        if persist:frame.to_csv(out/name,index=False)
        else:pd.testing.assert_frame_equal(frame.reset_index(drop=True),pd.read_csv(out/name),check_dtype=False,rtol=1e-6,atol=1e-8)
    protocol=read(out/'protocol.json');ph=sha256(out/'protocol.json')
    assert (out/'execution_finished.json').exists()
    finished=read(out/'execution_finished.json');assert finished['protocol_sha256']==ph
    lock=read(out/'selection_lock.json');assert lock['protocol_sha256']==ph
    cells,fingerprints=load_cells(ROOT/'data/processed/reviewed_v4/forecast_safe')
    assert fingerprints==protocol['data_artifact_sha256']
    for name,h in protocol['source_sha256'].items():assert sha256(ROOT/name)==h,name
    weights_checked=0; selection_rows=[]
    remote=read(out/'remote_artifact_manifest.json') if (out/'remote_artifact_manifest.json').exists() else {}
    for name,h in lock['selected_sha256'].items():
        p=out/name;assert sha256(p)==h
        s=read(p);assert s['protocol_sha256']==ph and s['reload_verified'] and s['frozen_verified']
        assert s['train_labels']==8 and s['validation_labels']==4
        wp=p.with_name('weights.pt')
        if wp.exists():assert sha256(wp)==s['weights_sha256'];weights_checked+=1
        elif require_weights:raise ValueError(f'Missing weights: {wp}')
        else:assert remote['artifacts'][str(wp.relative_to(out)).replace('\\','/')]['sha256']==s['weights_sha256']
        hist=pd.read_csv(p.with_name('history.csv'))
        if 'validation_mae' in s:
            close(s['validation_mae'],hist.validation_mae.min())
            for audit in s['gradient_audit'].values():
                assert audit['head_nonzero_gradient']
                assert audit['backbone_nonzero_gradient']==(s['method']!='head')
                if s['method']=='lora':assert all(n.startswith('head.') or 'lora_' in n for n in audit['trainable_names'])
                if s['method']=='head':assert all(n.startswith('head.') for n in audit['trainable_names'])
            selection_rows.append(dict(task='rul',dataset=p.parents[2].name,fold=p.parents[1].name,method=s['method'],
                step=s['step'],lr=s['lr'],validation_mae=s['validation_mae'],parameters=next(iter(s['gradient_audit'].values()))['trainable_parameters']))
        else:
            eligible=hist[(hist.crossing<=hist.iloc[0].crossing)&(hist.invalid<=hist.iloc[0].invalid)]
            close(s['loss'],eligible.loss.min())
            selection_rows.append(dict(task='soh',dataset=p.parents[2].name,fold=p.parent.name,method='head-only',
                context_length=int(p.parents[1].name[1:]),step=s['step'],lr=s['lr'],parameters=s['trainable_parameters']))
    assert len(selection_rows)==117
    save_table(pd.DataFrame(selection_rows),'selected_models.csv')
    budget=pd.read_csv(out/'label_budget.csv')
    # Check labels independently against original record positions, including EOL.
    for row in budget.itertuples():
        c=cells[(row.dataset,row.battery_id)]
        if row.task=='soh':close(row.target,c['values'][row.cycle-1])
        else:
            hits=np.flatnonzero(np.isfinite(c['values'])&(c['values']<=.8))
            assert len(hits);close(row.target,int(hits[0])+1-row.cycle)
            assert row.target>0
    for task,groups in protocol['split_design'].items():
        if task not in ('soh','rul'):continue
        for group,folds in groups.items():
            for f in folds:
                sets=[{cells[(group,b)]['physical_battery_id'] for b in f[k]} for k in ('train_ids','validation_ids','test_ids')]
                assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    expected_soh=set()
    for group,folds in protocol['split_design']['soh'].items():
        for f in folds:
            for b in f['test_ids']:
                for length in protocol['soh_lengths']:expected_soh.add((group,b,length))
    bm=pd.read_csv(out/'soh_battery_metrics.csv');pred_files=list((out/'soh_predictions').rglob('*.npz'))
    found=set();counts={};metrics=[]
    for p in pred_files:
        group=p.parent.name;length_text,battery=p.stem.split('_',1);length=int(length_text[1:]);found.add((group,battery,length))
        c=cells[(group,battery)]
        expected=np.asarray([t for t in range(65,c['last']+1) if np.isfinite(c['values'][t-1]) and np.isfinite(c['values'][t-3:t-1]).any()])
        with np.load(p) as z:
            assert str(z['protocol_sha256'])==ph;np.testing.assert_array_equal(z['cycle'],expected)
            close(z['target'],c['values'][expected-1])
            last=np.asarray([c['values'][t-length-1:t-1][np.isfinite(c['values'][t-length-1:t-1])][-1] for t in expected])
            close(z['last_value'],last)
            for method,pred in [('LastValue',z['last_value']),('zero-shot',z['zero'][:,1]),('head-only',z['head'][:,1])]:
                assert np.isfinite(pred).all()
                error=pred-z['target'];mse=float((error**2).mean());lv=float(((z['last_value']-z['target'])**2).mean())
                row=bm[(bm.dataset==group)&(bm.battery_id==battery)&(bm.context_length==length)&(bm.method==method)].iloc[0]
                close([row.mae,row.rmse,row.temporal_r2,row.mse_reduction_pct],
                      [np.abs(error).mean(),np.sqrt(mse),r2(z['target'],pred),100*(1-mse/lv)])
                metrics.append(dict(dataset=group,battery_id=battery,length=length,method=method))
            counts[(group,battery,length)]=len(expected)
    assert found==expected_soh
    assert len(bm)==len(found)*3
    summary=pd.read_csv(out/'soh_summary.csv')
    for row in summary.itertuples():
        b=bm[(bm.dataset==row.dataset)&(bm.context_length==row.context_length)&(bm.method==row.method)]
        close([row.mae,row.rmse,row.temporal_r2,row.mse_reduction_pct],[b.mae.mean(),b.rmse.mean(),b.temporal_r2.mean(),b.mse_reduction_pct.mean()])
    zero=pd.read_csv(out/'rul_zero_predictions.csv');direct=pd.read_csv(out/'rul_direct_predictions.csv')
    trajectory_origins=0
    for p in (out/'rul_trajectories').rglob('*.npz'):
        g,b=p.parent.name,p.stem;c=cells[(g,b)]
        with np.load(p) as z:
            assert str(z['protocol_sha256'])==ph
            expected=np.asarray([t for t in range(32,c['last']+1,16) if np.isfinite(c['values'][t-1]) and (c['eol'] is None or t<c['eol'])])
            np.testing.assert_array_equal(z['cycle'],expected)
            assert z['quantiles'].shape==(len(expected),3,1024) and np.isfinite(z['quantiles']).all()
            rows=zero[(zero.dataset==g)&(zero.battery_id==b)].set_index('origin_cycle')
            for i,t in enumerate(expected):
                row=rows.loc[t];hits=np.flatnonzero(z['quantiles'][i,1]<=.8)
                if len(hits):close(row.predicted_rul,hits[0]+1)
                else:assert np.isnan(row.predicted_rul)
                if c['eol'] is None:
                    assert not row.event_observed and np.isnan(row.true_rul)
                    close(row.observed_rul_lower_bound,c['last']-t)
                else:close(row.true_rul,c['eol']-t)
            trajectory_origins+=len(expected)
    assert trajectory_origins==len(zero)
    keycols=['dataset','battery_id','origin_cycle']
    keys=[]
    for mode in ('head','full','lora'):
        d=direct[direct.method==mode]
        assert not d.duplicated(keycols).any()
        keys.append(set(map(tuple,d[keycols].itertuples(index=False,name=None))))
    assert keys[0]==keys[1]==keys[2]
    for row in direct.itertuples():
        c=cells[(row.dataset,row.battery_id)]
        assert np.isfinite(row.predicted_rul) and row.predicted_rul>=0
        if c['eol'] is None:
            assert np.isnan(row.true_rul) and np.isnan(row.error_cycles)
            assert row.contradicts_censored_followup==(row.predicted_rul<=c['last']-row.origin_cycle)
        else:
            close(row.true_rul,c['eol']-row.origin_cycle)
            close(row.error_cycles,row.predicted_rul-row.true_rul)
        folds=protocol['split_design']['rul'][row.source_dataset]
        f=next(f for f in folds if f['fold_id']==row.fold)
        if row.scope=='within_dataset':assert row.battery_id in f['test_ids']
        else:assert row.battery_id in f['transfer_test'][row.dataset]
    matched=pd.read_csv(out/'rul_zero_matched_predictions.csv')
    assert set(map(tuple,matched[keycols].itertuples(index=False,name=None)))==keys[0]
    # Provide both origin-weighted and equal-cell summaries. Do not turn
    # non-crossing trajectory forecasts into finite RUL point estimates.
    battery_rows=[]
    all_rows=pd.concat([matched.assign(source_dataset=matched.dataset),direct],ignore_index=True)
    origin32=[]
    for (g,scope,method),d in all_rows[all_rows.origin_cycle==32].groupby(['dataset','scope','method']):
        observed=d[d.event_observed.eq(True)];finite=observed.predicted_rul.notna()
        errors=observed.loc[finite,'predicted_rul']-observed.loc[finite,'true_rul']
        lower=np.where(finite,np.abs(observed.predicted_rul-observed.true_rul),np.maximum(0,1025-observed.true_rul))
        origin32.append(dict(dataset=g,scope=scope,method=method,event_cells=len(observed),
            no_crossing=int((~finite).sum()),conditional_mae=float(np.abs(errors).mean()) if len(errors) else np.nan,
            mae_lower_bound=float(np.mean(lower)) if len(lower) else np.nan))
    save_table(pd.DataFrame(origin32),'rul_origin32_summary.csv')
    for (g,b,scope,method),d in all_rows.groupby(['dataset','battery_id','scope','method']):
        event=bool(d.event_observed.iloc[0]);finite=d.predicted_rul.notna()
        exact=d[finite] if event else d.iloc[:0]
        errs=exact.predicted_rul-exact.true_rul
        lower=np.where(finite,np.abs(d.predicted_rul-d.true_rul),np.maximum(0,1025-d.true_rul)) if event else np.full(len(d),np.nan)
        battery_rows.append(dict(dataset=g,battery_id=b,scope=scope,method=method,n_origins=len(d),event_observed=event,
            n_no_crossing=int((~finite).sum()),conditional_mae=float(np.abs(errs).mean()) if len(errs) else np.nan,
            mae_lower_bound=float(np.mean(lower)) if event else np.nan,
            temporal_r2=r2(d.true_rul,d.predicted_rul) if event and finite.all() else np.nan,
            censored_contradictions=int(d.contradicts_censored_followup.sum()) if not event else 0))
    rb=pd.DataFrame(battery_rows);save_table(rb,'rul_battery_metrics.csv')
    macro=rb.groupby(['dataset','scope','method'],as_index=False).agg(n_batteries=('battery_id','size'),n_event_batteries=('event_observed','sum'),
        n_origins=('n_origins','sum'),n_no_crossing=('n_no_crossing','sum'),macro_conditional_mae=('conditional_mae','mean'),
        macro_mae_lower_bound=('mae_lower_bound','mean'),mean_cell_r2=('temporal_r2','mean'),
        n_defined_cell_r2=('temporal_r2','count'),censored_contradictions=('censored_contradictions','sum'))
    save_table(macro,'rul_macro_summary.csv')
    result=dict(status='passed',verified_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=ph,selected_models=117,
        checkpoints_sha256_checked_locally=weights_checked,checkpoints_sha256_checked_remotely=117 if remote else None,
        soh_prediction_files=len(pred_files),soh_test_cells=len({(g,b) for g,b,l in found}),
        soh_targets_per_length=sum(n for (g,b,l),n in counts.items() if l==2),
        rul_all_zero_origins=len(zero),rul_matched_origins_per_method=len(keys[0]),
        direct_test_cells=len(direct[keycols[:2]].drop_duplicates()),source_and_test_disjoint=True,
        censored_not_exact_labels=True,identical_direct_method_test_origins=True,
        scores_independently_recomputed=True,selection_histories_verified=True,source_and_data_hashes_verified=True)
    write(out/('verification.json' if persist else 'local_verification.json'),result)
    return cells,result


def table(frame,columns,formats=None):
    formats=formats or {};lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,row in frame.iterrows():
        vals=[]
        for name in columns:
            v=row[name]
            vals.append('—' if pd.isna(v) else formats[name].format(v) if name in formats else str(v))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def render(out,cells,verification):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    protocol=read(out/'protocol.json');summary=pd.read_csv(out/'soh_summary.csv')
    mean=summary.groupby(['context_length','method'],as_index=False).mse_reduction_pct.mean()
    mean.to_csv(out/'soh_mean_reduction.csv',index=False)
    colors={'zero-shot':'#0075C4','head-only':'#A040C0','head':'#A040C0','full':'#D55E00','lora':'#009E73','trajectory-zero-shot':'#0075C4'}
    fig,ax=plt.subplots(figsize=(8,4.5))
    for method in ('zero-shot','head-only'):
        d=mean[mean.method==method];ax.plot(d.context_length,d.mse_reduction_pct,'o-',label=method,color=colors[method])
    ax.set_xscale('log',base=2);ax.set_xticks([2,4,8,16,32,64],labels=[2,4,8,16,32,64])
    ax.axhline(0,color='.5',lw=.8);ax.set(xlabel='Input SOH positions (L)',ylabel='Mean MSE reduction vs LastValue (%)',title='v4 forecast_safe | equal weight to 7 dataset groups')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'soh_input_reduction.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(14,12));axes=axes.ravel()
    for ax,(g,folds) in zip(axes,protocol['split_design']['soh'].items()):
        b=folds[0]['test_ids'][0];c=cells[(g,b)]
        ax.plot(c['cycles'],c['values'],color='black',lw=1,label='Eligible measured SOH')
        with np.load(out/'soh_predictions'/g/f'L32_{b}.npz') as z:
            for method,key in [('zero-shot','zero'),('head-only','head')]:
                line=np.full(len(c['cycles']),np.nan);line[z['cycle']-1]=z[key][:,1]
                ax.plot(c['cycles'],line,color=colors[method],lw=.8,label=method)
        ax.set(title=f'{g} | {b} | L=32 -> 1',xlabel='Original cycle / observation index',ylabel='SOH');ax.grid(alpha=.2)
    axes[-1].axis('off');axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'soh_representatives.png',dpi=160);plt.close(fig)
    direct=pd.read_csv(out/'rul_direct_predictions.csv');zero=pd.read_csv(out/'rul_zero_matched_predictions.csv')
    fig,axes=plt.subplots(3,2,figsize=(13,10));axes=axes.ravel()
    event_groups=['nasa_aging','calce_cs2_1c','mit_batch2','xjtu_2c','xjtu_3c']
    for ax,g in zip(axes,event_groups):
        candidates=direct[(direct.dataset==g)&direct.event_observed.eq(True)]
        b=sorted(candidates.battery_id.unique())[0];d=candidates[candidates.battery_id==b]
        ref=d[d.method=='head'];ax.plot(ref.origin_cycle,ref.true_rul,color='black',lw=1.6,label='True RUL')
        for mode in ('head','full','lora'):
            q=d[d.method==mode];ax.plot(q.origin_cycle,q.predicted_rul,'o-',ms=3,color=colors[mode],label=mode)
        q=zero[(zero.dataset==g)&(zero.battery_id==b)]
        ax.plot(q.origin_cycle,q.predicted_rul,'s--',ms=3,color=colors['trajectory-zero-shot'],label='Trajectory zero-shot (finite only)')
        missing=q.predicted_rul.isna().sum()
        ax.set(title=f'{g} | {b} | trajectory non-crossing {missing}/{len(q)}',xlabel='Prediction origin',ylabel='Remaining record positions');ax.grid(alpha=.2)
    axes[-1].axis('off');axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'rul_direct_representatives.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(14,12));axes=axes.ravel()
    for ax,g in zip(axes,protocol['split_design']['soh']):
        z=zero[zero.dataset==g];b=sorted(z.battery_id.unique())[0];c=cells[(g,b)]
        with np.load(out/'rul_trajectories'/g/f'{b}.npz') as saved:
            t=int(saved['cycle'][0]);q=saved['quantiles'][0];future=t+np.arange(1,1025)
            ax.plot(c['cycles'],c['values'],'k-',lw=1,label='Eligible measured SOH')
            ax.plot(future,q[1],color='#0075C4',label='Zero-shot q50')
            # Quantiles are neither sorted nor repaired; omit crossed interval fills.
            valid=q[0]<=q[2];ax.fill_between(future,q[0],q[2],where=valid,color='#0075C4',alpha=.12,label='q10-q90 where ordered')
            ax.axhline(.8,color='#D55E00',ls='--',lw=1);ax.axvline(t,color='.5',ls=':',lw=.8)
            if c['eol'] is not None:ax.axvline(c['eol'],color='black',ls='--',lw=.8)
            ax.set(title=f'{g} | {b} | origin {t}, H=1024',xlabel='Original cycle / observation index',ylabel='SOH');ax.grid(alpha=.2)
    axes[-1].axis('off');axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'rul_zero_trajectories.png',dpi=160);plt.close(fig)
    rul=pd.read_csv(out/'rul_summary.csv');macro=pd.read_csv(out/'rul_macro_summary.csv')
    selected=pd.read_csv(out/'selected_models.csv');environment=read(out/'environment.json')
    lines=['# C2 v4 GPU 실험: 현재 SOH · 궤적 RUL · 직접 RUL','',
        '새 전처리 `reviewed_v4/forecast_safe`로 새로 학습·예측한 결과다. 이전 v3 점수의 재채점이 아니다.','',
        f"검증: {verification['selected_models']}개 모델 선택, SOH {verification['soh_test_cells']}셀 × 6개 입력 길이, 길이당 {verification['soh_targets_per_length']:,}개 공통 시점. 직접 RUL은 {verification['direct_test_cells']}개 시험 셀, 방법당 {verification['rul_matched_origins_per_method']:,}개 동일 시점.",'',
        '## 실행 조건','',
        f"- SSH `home-pc`의 {environment['gpu']}, CUDA {environment['cuda']}, PyTorch {environment['packages']['torch']}; float32, seed 42.",
        '- SOH: 과거 L개 기록 위치 → 현재 한 점. L=2·4·8·16·32·64, LastValue / zero-shot / native SOH head-only. 매 시점 실측 과거 입력을 갱신한다.',
        '- MIT 연속 기록을 물리 셀로 병합한 v4에서 분할을 새로 고정했다. v3와 시험 셀 구성도 달라 기존 점수와의 차이를 전처리만의 개선 효과로 해석하지 않는다.',
        '- RUL: 과거 32개 SOH, t=32부터 16개 기록 위치 간격. Q/Q_ref ≤ 0.8 최초 적격 관측이 EOL이다. 정격용량 80%와 혼동하지 않는다.',
        '- 궤적 RUL: 순정 C2 q50 미래 1,024개 위치를 한 번에 출력한다. 미래 실측 투입·평활화·단조화·보간 없이 최초 80% 교차를 찾는다. 미도달은 >1024로 남긴다.',
        '- 직접 RUL: C2 REG token + 입력 정규화 loc/scale + 마지막 적격 SOH → 새 RUL head. 입력에 cycle, 데이터셋 ID, EOL, RUL 정답을 넣지 않는다. 출력은 softplus로 비음수다.',
        '- Head 51,015 / LoRA 1,230,663 / Full 115,875,399개 파라미터 학습. Full은 사용하지 않는 원래 SOH 출력층을 제외한 본체 전체와 새 RUL head를 학습한다. LoRA는 attention q/k/v/o에 rank 8, alpha 16을 적용한다.',
        '- 매 fold: 다른 학습 셀 2개 × 정답 4개 = 8 labels, 별도 검증 셀 1개 × 4개 = 4 labels. 입력 이력과 완료된 source 셀의 EOL 관측 비용은 별도다. 8개 독립 배터리를 학습했다는 뜻이 아니다.',
        '- SOH 2개 LR × 100 steps; RUL 방식별 2개 LR × 200 steps, batch 4, 20 steps마다 검증, step 0 허용. RUL은 train-label median으로 스케일한 Huber loss, 검증 MAE로 선택한다.',
        '- 세 직접 방식의 source 셀·라벨·초기 head·미니배치 순서가 같다. 모든 모델 선택을 잠근 뒤 시험 평가했다. 내부 역전파 범위·동결 가중치·체크포인트 재로딩을 확인했다.',
        '- CALCE의 cycle 단위는 보존한 관측 순서다. 파일 사이 미관측 물리 cycle 수가 불명확하므로 물리 cycle 수명과 동일시하지 않는다.',
        '- 한 seed 및 작은 고정 예산의 첫 비교다. 최적 튜닝 성능이나 통계적으로 확정된 순위로 해석하지 않는다.','',
        '## 현재 SOH','',
        '배터리별 MSE 감소율을 평균한 다음, 7개 데이터셋에 같은 가중치를 준 평균이다. R²는 셀의 여러 단일 시점 정답·예측을 모은 temporal R²이며 음수도 그대로 유지한다.','',
        table(mean[mean.method!='LastValue'],['context_length','method','mse_reduction_pct'],{'mse_reduction_pct':'{:.2f}'}),'',
        '![입력별 평균 감소율](soh_input_reduction.png)','',
        '[데이터셋별 SOH 요약](soh_summary.csv) · [셀별 점수](soh_battery_metrics.csv)','',
        '![현재 SOH 대표 예측](soh_representatives.png)','',
        '## RUL 정확한 정답이 있는 시험 셀','',
        '아래 MAE는 원점별 집계다. 궤적 방식의 conditional MAE는 **80% 교차를 예측한 시점에 한정**되므로 직접 방식의 전체 MAE와 단독 비교하면 안 된다. 미도달 수와 전체 오차 하한을 함께 본다. 직접 방식의 하한 열은 정확한 MAE와 같다.','',
        table(rul[rul.n_event_origins>0],['dataset','scope','method','n_event_batteries','n_event_origins','n_no_crossing_event_predictions','conditional_mae_cycles','mae_lower_bound_all_event_origins'],
            {'conditional_mae_cycles':'{:.2f}','mae_lower_bound_all_event_origins':'{:.2f}'}),'',
        '[같은 시험 시점의 궤적 결과](rul_zero_matched_predictions.csv) · [직접 예측 원시 결과](rul_direct_predictions.csv)','',
        '[최초 공통 예측 시점 t=32만의 별도 비교](rul_origin32_summary.csv) — 전체 원점 결과와 구분한다. 이전 v3의 t=64·H=256 결과와는 데이터·시점·horizon이 모두 달라 직접 개선율을 계산하지 않는다.','',
        '![직접 RUL 대표 예측](rul_direct_representatives.png)','',
        '## RUL 셀별 동일 가중 평균 및 R²','',
        'R²는 셀의 여러 예측 원점을 모아 계산한 뒤 셀별 평균한다. 궤적 방식에서 미도달 예측이 있는 셀은 전체 R²가 정의되지 않아 제외하고, 정의된 셀 수를 명시한다. 서로 다른 포함 셀 수의 R²를 순위로 비교하지 않는다.','',
        table(macro[macro.n_event_batteries>0],['dataset','scope','method','macro_conditional_mae','macro_mae_lower_bound','mean_cell_r2','n_defined_cell_r2'],
              {'macro_conditional_mae':'{:.2f}','macro_mae_lower_bound':'{:.2f}','mean_cell_r2':'{:.3f}'}),'',
        '## 미도달 셀과 전이 예측','',
        '- MIT Batch 1·3은 현재 데이터에서 SOH 80% 도달 정답이 없다. MIT Batch 2의 2개 학습 셀·1개 검증 셀로 정한 모델을 그대로 적용한 전이 예측이다.',
        '- XJTU 3C는 도달 셀이 2개뿐이다. 별도 학습/검증/시험 분리를 만들 수 없어 XJTU 2C 첫 fold 모델을 그대로 적용했다. 이 2개 셀의 MAE는 조건이 달라진 전이 성능이다.',
        '- XJTU 2C의 미도달 셀은 첫 fold의 시험에만 포함했다. 미도달 셀에는 정확한 RUL MAE/R²를 만들지 않는다.',
        '- 아래 모순 수는 예측 RUL이 실제 무고장 관측 기간보다 짧은 경우다. 모순이 없다고 정확한 RUL을 맞혔다고 볼 수 없다.',
        '- EOL이 관측된 source 셀만으로 정확한 회귀를 학습했으므로 관측 종료 전에 고장난 셀을 선택한 편향이 있다. 검열을 직접 학습에 활용하는 생존 모델은 이번 범위에 포함하지 않았다.','',
        table(rul[rul.censored_origins>0],['dataset','scope','method','censored_origins','censored_contradictions']),'',
        '![순정 미래 SOH 궤적](rul_zero_trajectories.png)','',
        '## 선택·재현성','',
        '- 원본·v3·전처리 v4 데이터와 이전 모델 결과를 덮어쓰지 않았다. 전처리 및 실행 코드 해시, 입력 bundle 해시, split, 라벨 위치, 환경, 예측값, checkpoint 해시를 저장했다.',
        '- 그래프는 적격 실측만 표시한다. 제외된 point는 그리지 않으며 원래 cycle과 NaN 공백을 유지한다.',
        '- 첫 개발 실행(v1)은 새 캐시 학습 손실의 pinball 방향 오류를 시험 평가 전에 발견해 폐기했다. 이 v2는 예측값·학습 손실 모두 공식 C2와 일치하는 검사를 통과한 별도 실행이다.',
        f"- 모델 선택 중 step 0: SOH {int(((selected.task=='soh')&(selected.step==0)).sum())}/78, 직접 RUL {int(((selected.task=='rul')&(selected.step==0)).sum())}/39. Step 0 직접 RUL은 학습 source의 평균 RUL로 초기화한 회귀 head이며 zero-shot이라 부르지 않는다.",
        '- 원격 실행 폴더: `C:\\Users\\master\\Lab\\chronos-battery-soh\\experiments\\c2-v4-suite-v2`. 실행 환경은 상위 프로젝트 `.venv`다.',
        '- 새 출력 폴더에서 재실행: `python -X utf8 scripts/run_c2_v4_suite.py --out <새 결과 폴더> --cache-root <Chronos-2 캐시>`; `--smoke-only`로 먼저 CUDA·역전파·재로딩을 검사할 수 있다. 완료 결과는 덮어쓰지 않는다.',
        '- 검증/보고서 재생성: `python -X utf8 scripts/verify_report_c2_v4_suite.py --out <결과 폴더>`. 모든 checkpoint 파일이 있어야 기본 검증이 통과한다.',
        '', '[고정 프로토콜](protocol.json) · [정답 예산](label_budget.csv) · [EOL 감사](eol_audit.csv) · [모델 선택](selected_models.csv) · [검증](verification.json) · [환경](environment.json)','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--remote-weights',action='store_true')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--read-only',action='store_true');a=p.parse_args();out=a.out.resolve()
    cells,result=verify(out,require_weights=not a.remote_weights,persist=not a.read_only)
    if not a.verify_only and not a.read_only:render(out,cells,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
