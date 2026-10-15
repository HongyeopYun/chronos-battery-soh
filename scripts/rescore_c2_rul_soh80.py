"""Rescore saved NASA RUL trajectories at SOH <= 0.8; no inference or fitting."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.c2_rul import eligible_mask, first_crossing, score_trajectory, summarize_rul
from src.soh_data import load_bundle, sha256

OUT = ROOT/'results/c2-battery/rul-nasa-soh80-rescore-v1'
BUNDLE = ROOT/'data/processed/reviewed_native_v3/nasa_aging'
RUNS = ['rul-nasa-l64-h256-zero-v1', 'rul-nasa-l64-h256-head-v1',
        'rul-nasa-context-sweep-cuda-v1', 'rul-nasa-multi-origin-xpu-v1']
PRIMARY = RUNS[-1]
THRESHOLD = .8


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def eol_audit(data, metadata):
    rows = []
    for b, raw in data.groupby('battery_id', sort=True):
        f = raw.sort_values('cycle')
        keep = eligible_mask(f)
        qref = float(metadata.set_index('battery_id').loc[b, 'reference_capacity_ah'])
        np.testing.assert_allclose(f.soh, f.capacity_ah/qref, rtol=1e-12)
        eol = first_crossing(f.cycle.to_numpy(), f.soh.where(keep).to_numpy(), THRESHOLD)
        old = first_crossing(f.cycle.to_numpy(), f.capacity_ah.where(keep).to_numpy(), 1.4)
        rows.append(dict(battery_id=b, reference_capacity_ah=qref, threshold_soh=THRESHOLD,
            threshold_capacity_ah=THRESHOLD*qref, old_eol_1p4ah=old, eol_cycle=eol,
            last_eligible_cycle=int(f.loc[keep, 'cycle'].max()),
            eligible_at_t64=eol is None or eol>64))
    return pd.DataFrame(rows)


def rethreshold(task, audit_row):
    """Exclude origins at/after first observed EOL, even if SOH later recovers."""
    eol = None if pd.isna(audit_row.eol_cycle) else int(audit_row.eol_cycle)
    if eol is not None and eol <= task['origin_cycle']:
        return None
    return dict(task, threshold_soh=THRESHOLD, event_observed=eol is not None,
        true_eol_cycle=eol, true_rul=None if eol is None else eol-task['origin_cycle'],
        observed_rul_lower_bound=int(audit_row.last_eligible_cycle)-task['origin_cycle'] if eol is None else None)


def conditions(run, plan, arrays):
    if run == RUNS[0]:
        yield 64, 'zero-shot', plan['tasks'], arrays['quantiles']
    elif run == RUNS[1]:
        for method, key in [('zero-shot','zero_shot'), ('head-only','head_only')]:
            yield 64, method, plan['tasks'], arrays[key]
    elif run == RUNS[2]:
        for length in [16,32,64]:
            for method, key in [('zero-shot','zero'), ('head-only','head')]:
                yield length, method, plan['tasks'][str(length)], arrays[f'L{length}_{key}']
    else:
        for method in ['zero-shot','single','multi']:
            yield 64, method, plan['tasks'], arrays[method]


def summaries(scores):
    rows = []
    for (run, length, method), group in scores.groupby(['source_run','context_length','method'], sort=False):
        for scope, part in [('primary_t64',group[group.origin_cycle.eq(64)]), ('all_saved_eligible_origins',group)]:
            rows.append(dict(source_run=run, context_length=length, method=method, scope=scope,
                **summarize_rul(part)))
    return pd.DataFrame(rows)


def plot_and_report(audit, scores, summary, data, paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(13,8),sharex=True)
    styles=[('zero-shot','Zero-shot','#2467ad'),('single','Head: source t64','#9d55aa'),
            ('multi','Head: source t64/80/96','#138572')]
    for ax, row in zip(axes.flat, audit.itertuples()):
        f=data[data.battery_id.eq(row.battery_id)].sort_values('cycle')
        ax.plot(f.cycle,f.soh.where(eligible_mask(f)),color='black',lw=1.4,label='Eligible measurements')
        ax.axhline(.8,color='#555555',ls='--',lw=1,label='EOL: SOH 0.80')
        ax.axvline(64,color='#777777',ls=':',lw=1)
        if not pd.isna(row.eol_cycle):
            ax.axvline(row.eol_cycle,color='#c17d23',ls=':',lw=1,label='Observed EOL')
        if row.eligible_at_t64:
            ax.axvspan(1,64,color='#999999',alpha=.10)
            for key,label,color in styles:
                ax.plot(np.arange(65,321),paths[(row.battery_id,key)][:,4],label=label,color=color,lw=1.4)
            ax.set_title(f'{row.battery_id} | true RUL at t64 = {int(row.eol_cycle)-64}')
        else:
            # Hide forecasts at invalid origins; never present a negative RUL as a test target.
            ax.set_title(f'{row.battery_id} | t64 not evaluated')
            ax.text(.98,.92,f'EOL at record {int(row.eol_cycle)}\nAlready reached before t64',
                transform=ax.transAxes,ha='right',va='top',fontsize=11,
                bbox=dict(facecolor='white',edgecolor='#cccccc',alpha=.9))
        ax.set(xlabel='Discharge record ordinal',ylabel='SOH');ax.grid(alpha=.18)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
    fig.suptitle('EOL at SOH 80%: same saved forecasts, no retraining',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.87));fig.savefig(OUT/'trajectories.png',dpi=170);plt.close(fig)
    selected=scores[scores.source_run.eq(PRIMARY)&scores.origin_cycle.eq(64)]
    s=summary[summary.source_run.eq(PRIMARY)&summary.scope.eq('primary_t64')]
    fmt=lambda v:'>256' if pd.isna(v) else str(int(v))
    lines=['# NASA RUL: SOH 80% 기준 재평가','',
        '기존 미래 SOH 예측은 변경하지 않고, EOL을 최초 유효 SOH ≤ 0.8로 다시 정의했다. 재학습·재추론·체크포인트 재선택은 하지 않았다.','',
        '- SOH = Q / Q_ref. 기준은 셀별 Q_ref의 80%이며 정격 2Ah의 80%(1.6Ah)가 아니다.',
        '- EOL은 처음 관측된 임계값 도달 기록이다. 이후 용량이 회복해도 이벤트를 취소하지 않는다.',
        '- 해당 시점까지 EOL이 이미 관측된 예측 원점은 제외한다. B0006은 EOL=61이므로 t64에서 평가하지 않는다.',
        '- B0007은 새 기준에서 EOL=124가 관측된다. 이전 1.4Ah 평가의 우측 검열 사례와 구분한다.',
        '- 주 결과는 최신 XPU 학습 시점 비교의 저장된 t64 예측이다. CUDA 입력 16/32/64 및 초기 다중 평가 시점도 별도 표에 재채점했다.',
        '- Head의 기존 학습·검증 조건은 유지했다. 원래 학습 시점은 1.4Ah 기준으로 설정되어 일부 창은 새 80% EOL 이후에 있다. 따라서 80% 기준으로 새롭게 설계·학습한 모델 결과는 아니다.',
        '- 1.4Ah 결과와 평가 셀 및 정답이 달라졌으므로 점수 차이를 모델 개선으로 해석하지 않는다.',
        '- 제외된 측정점은 그림에서 숨겼으며, 평가 제외 원점의 예측 경로도 표시하지 않았다.','',
        '| 셀 | 기존 EOL: 1.4Ah | 새 EOL: SOH 80% | t64 실제 RUL |',
        '|---|---:|---:|---:|']
    for r in audit.itertuples():
        old='미관측' if pd.isna(r.old_eol_1p4ah) else str(int(r.old_eol_1p4ah))
        truth=str(int(r.eol_cycle)-64) if r.eligible_at_t64 else '평가 제외: 이미 EOL 도달'
        lines.append(f'| {r.battery_id} | {old} | {int(r.eol_cycle)} | {truth} |')
    lines+=['','| 셀 | 실제 RUL | Zero-shot | 기존 head | 학습 시점 확장 head |','|---|---:|---:|---:|---:|']
    for b,g in selected.groupby('battery_id'):
        vals=[fmt(g[g.method.eq(m)].iloc[0].predicted_rul) for m in ['zero-shot','single','multi']]
        lines.append(f'| {b} | {int(g.iloc[0].true_rul)} | '+' | '.join(vals)+' |')
    lines+=['','| 방법 | EOL 미도달 / 사건 수 | 유한 사례 조건부 MAE | 전체 MAE 하한 |','|---|---:|---:|---:|']
    for r in s.itertuples():
        lines.append(f'| {r.method} | {r.n_no_crossing_event_predictions}/{r.n_event_origins} | {r.conditional_mae_cycles:.2f} | ≥{r.mae_lower_bound_all_event_origins:.2f} |')
    lines+=['','단위는 방전 기록 수. 조건부 MAE는 도달 시점이 산출된 사례만 포함하며 전체 성능을 대표하지 않는다. 미도달 RUL>256의 오차 하한은 max(0,257−실제 RUL)로 계산한다.',
        '','![SOH 80% 경로](trajectories.png)','','입력 길이 비교는 [primary_context_sweep.csv](primary_context_sweep.csv), 기존 다중 시점 재평가는 [summary.csv](summary.csv)에 있다. 다중 시점 요약은 원점 단위의 기술통계이며 독립 배터리 평균으로 해석하지 않는다.',
        '','[배터리별 EOL](eol_audit.csv) · [전체 RUL 원점수](rul_predictions.csv) · [제외 원점](excluded_origins.csv) · [검증](verification.json)','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    if OUT.exists(): raise ValueError('Preserve completed rescore; use a new output directory')
    data, meta, _ = load_bundle(BUNDLE)
    audit=eol_audit(data,meta);by_id=audit.set_index('battery_id')
    tracked=[BUNDLE/n for n in ['soh.csv','batteries.csv','manifest.json']]
    for run in RUNS:
        folder=ROOT/'results/c2-battery'/run
        assert json.loads((folder/'status.json').read_text())['status']=='complete'
        tracked.extend(folder/n for n in ['protocol.json','forecast_quantiles.npz','status.json'])
    original={p.relative_to(ROOT).as_posix():sha256(p) for p in tracked}
    scores=[];excluded=[];diagnostics=[];paths={};checks=0
    for run in RUNS:
        folder=ROOT/'results/c2-battery'/run;plan=json.loads((folder/'protocol.json').read_text())
        with np.load(folder/'forecast_quantiles.npz') as arrays:
            for length,method,tasks,q in conditions(run,plan,arrays):
                assert q.shape==(len(tasks),256,9) and np.isfinite(q).all()
                for task,forecast in zip(tasks,q):
                    keys=dict(source_run=run,context_length=length,method=method,battery_id=task['battery_id'],origin_cycle=task['origin_cycle'])
                    updated=rethreshold(task,by_id.loc[task['battery_id']])
                    if updated is None:
                        excluded.append(dict(**keys,eol_cycle=int(by_id.loc[task['battery_id'],'eol_cycle']),reason='EOL already observed at or before origin'))
                        continue
                    score=score_trajectory(updated,forecast[:,4]);scores.append(dict(**score,source_run=run,context_length=length,method=method))
                    # Independently check crossing index using scalar comparisons.
                    hits=[i+1 for i,v in enumerate(forecast[:,4]) if float(v)<=THRESHOLD]
                    assert score['predicted_rul']==(hits[0] if hits else None)
                    assert score['true_rul'] is None or score['true_rul']>0
                    if score['predicted_rul'] is None and score['true_rul'] is not None:
                        assert score['absolute_error_lower_bound_cycles']==max(0,257-score['true_rul'])
                    checks+=1
                    f=data[data.battery_id.eq(task['battery_id'])].sort_values('cycle')
                    origin=task['origin_cycle'];f=f[eligible_mask(f)&f.cycle.gt(origin)&f.cycle.le(origin+256)]
                    if updated['true_eol_cycle'] is not None:f=f[f.cycle.le(updated['true_eol_cycle'])]
                    e=forecast[(f.cycle.to_numpy(dtype=int)-origin-1),4]-f.soh.to_numpy()
                    diagnostics.append(dict(**keys,n_scored=len(f),mae_soh_pp=float(abs(e).mean()*100) if len(e) else np.nan,
                        rmse_soh_pp=float(np.sqrt(np.mean(e**2))*100) if len(e) else np.nan))
                    if run==PRIMARY:paths[(task['battery_id'],method)]=forecast.copy()
    scores=pd.DataFrame(scores);excluded=pd.DataFrame(excluded);summary=summaries(scores)
    assert set(scores[scores.origin_cycle.eq(64)].battery_id)=={'B0005','B0007','B0018'}
    OUT.mkdir(parents=True)
    audit.to_csv(OUT/'eol_audit.csv',index=False);scores.to_csv(OUT/'rul_predictions.csv',index=False)
    excluded.to_csv(OUT/'excluded_origins.csv',index=False);summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'soh_metrics_through_eol.csv',index=False)
    summary[summary.source_run.eq(RUNS[2])&summary.scope.eq('primary_t64')].to_csv(OUT/'primary_context_sweep.csv',index=False)
    plot_and_report(audit,scores,summary,data,paths)
    # All source runs and input data must remain byte-identical after reporting.
    for p,digest in original.items():assert sha256(ROOT/p)==digest,p
    sources=['scripts/rescore_c2_rul_soh80.py','src/c2_rul.py','src/soh_data.py']
    write_json(OUT/'protocol.json',dict(threshold_soh=THRESHOLD,definition='First eligible SOH <= 0.8; SOH=Q/Q_ref',
        primary_origin=64,source_runs=RUNS,source_artifacts_sha256=original,
        source_sha256={p:sha256(ROOT/p) for p in sources},no_retraining=True,no_new_inference=True,
        unchanged_model_selection=True,not_a_fresh_80_percent_training_protocol=True))
    write_json(OUT/'verification.json',dict(status='passed',independent_forecast_crossings=checks,
        scored_predictions=len(scores),excluded_predictions=len(excluded),original_artifacts_unchanged=True,
        nonpositive_true_rul_excluded=True,artifacts_sha256={p.name:sha256(p) for p in OUT.iterdir() if p.is_file()}))
    write_json(OUT/'status.json',dict(status='complete',protocol_sha256=sha256(OUT/'protocol.json')))
    print(summary[summary.source_run.eq(PRIMARY)&summary.scope.eq('primary_t64')].to_string(index=False))
    print('COMPLETE',OUT)

if __name__=='__main__':main()
