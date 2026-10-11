"""32->1 follow-up with the frozen reviewed 64->1 labels and test timestamps."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts import run_c2_one_step_context as base
from src.c2_one_step import crop_windows
from src.c2_context_comparison import METHODS, paired_metrics, summarize_paired, compare_lengths

PRIOR = ROOT / 'results/c2-battery/one-step-64-reviewed-v3'
OUT = ROOT / 'results/c2-battery/one-step-32-reviewed-v3'
BUNDLES = ROOT / 'data/processed/reviewed_native_v3'
WEIGHTS = ROOT / 'models/c2-head-one-step-32-reviewed-v3'


def cropped(windows, length):
    result = crop_windows(windows, length)
    finite = np.isfinite(result['context'])
    if not finite.any(axis=1).all():
        raise ValueError('A matched target has no observed input in its shorter history')
    result['context_observed'] = finite.sum(axis=1)
    result['last_observed_lag'] = length-np.where(finite, np.arange(length), -1).max(axis=1)
    return result


def persistence(context):
    context = np.asarray(context)
    finite = np.isfinite(context)
    if context.ndim != 2 or np.isinf(context).any() or not finite.any(axis=1).all():
        raise ValueError('A finite past observation is required')
    last = np.where(finite, np.arange(context.shape[1]), -1).max(axis=1)
    return {'LastValue': context[np.arange(len(context)), last]}


def configure():
    # Keep the frozen 64-run source and outputs intact; reuse its training engine.
    base.OUT, base.WEIGHTS, base.BUNDLES = OUT, WEIGHTS, BUNDLES
    base.CONTEXTS, base.METHODS, base.REVIEWED = (32,), METHODS, True
    base.crop_windows, base.baselines = cropped, persistence
    base.plot_sweep = lambda summary: None
    base.write_reviewed_report, base.verify = write_report, verify


def check_frozen_prior():
    prior = base.read(PRIOR / 'protocol.json')
    release = base.read(PRIOR / 'release_verification.json')
    if base.read(PRIOR / 'status.json')['status'] != 'complete':
        raise ValueError('The 64->1 reference must be complete')
    assert base.sha256(PRIOR / 'protocol.json') == release['protocol_sha256']
    for path, digest in prior['source_sha256'].items():
        assert base.sha256(ROOT / path) == digest, path
    for group, digest in release['predictions_sha256'].items():
        assert base.sha256(PRIOR / group / 'predictions.csv.gz') == digest, group
    return prior, release


def prepare():
    prior, release = check_frozen_prior()
    master = base.ZeroShotConfig(**prior['config'])
    cfg = replace(master, context_length=32)
    plan = deepcopy(prior)
    plan.update(schema_version=2, config=asdict(cfg), master_window_config=asdict(master),
                contexts=[32], methods=list(METHODS), reference_context_length=64,
                reference_protocol_sha256=release['protocol_sha256'],
                reference_predictions_sha256=release['predictions_sha256'],
                matching='Exact 64-run target times, folds, label indices and label values. Input is the final 32 positions of each 64-position window; preserve NaN masks and original positions.',
                selection='Same validation-only loss and crossing guard as 64->1; step zero allowed. Fresh original C2 initialization for each fold/LR. Lock all 13 heads before test inference.',
                aggregation='Equal record weight for MAE/RMSE in SOH ratio units. Compute 1-MSE_model/MSE_LastValue separately per record, then average; negative skills retained. Zero reference error is undefined and counted. R2 is auxiliary only.')
    sources = set(prior['source_sha256']) | {
        'scripts/run_c2_context32.py', 'src/c2_context_comparison.py',
        'src/c2_partial_tuning.py', 'src/battery_fewshot_tuning.py', 'src/c2_head_protocol.py'}
    plan['source_sha256'] = {p: base.sha256(ROOT/p) for p in sorted(sources)}
    prepared, coverage, budgets = {}, [], []
    for group, spec in plan['groups'].items():
        assert base.sha256(BUNDLES/group/'manifest.json') == spec['bundle_sha256']
        assert base.sha256(BUNDLES/group/'soh.csv') == spec['data_sha256']
        records, _, data, _ = base.prepare_reviewed_windows(BUNDLES/group, master)
        by_id = {r['battery_id']: r for r in records}
        source = {b for f in spec['folds'] for k in ('train_ids','validation_ids') for b in f[k]}
        test = {b for f in spec['folds'] for b in f['test_ids']}
        old = pd.read_csv(PRIOR/group/'predictions.csv.gz')
        old = old[old.method.eq('LastValue')]
        for b in sorted(source | test):
            w = cropped(by_id[b]['windows'], 32)
            if b in test:
                ref = old[old.battery_id.eq(b)].sort_values('target_cycle')
                np.testing.assert_array_equal(ref.target_cycle, w['target_cycle'][:,0])
                np.testing.assert_allclose(ref.target, w['target'][:,0], rtol=1e-12)
                np.testing.assert_allclose(ref.prediction, persistence(w['context'])['LastValue'], rtol=1e-12)
            coverage.append(dict(dataset=group,battery_id=b,role='held_out_test' if b in test else 'source_or_validation',
                                 context_length=32,n_targets=len(w['target']),
                                 min_context_observed=int(w['context_observed'].min()),
                                 targets_with_missing_context=int((w['context_observed']<32).sum())))
        for fold in spec['folds']:
            base.check_physical_split(fold, {b:r['physical_battery_id'] for b,r in by_id.items()})
            for label in fold['label_budget']:
                w = cropped(by_id[label['battery_id']]['windows'],32)
                i = label['index']
                assert int(w['target_cycle'][i,0]) == label['target_cycle']
                budgets.append(dict(**label,context_length=32,context_start=int(w['context_cycle'][i,0]),
                                    context_end=int(w['context_cycle'][i,-1]),
                                    observed_inputs=int(w['context_observed'][i])))
        assert len(old) == spec['test_windows']
        prepared[group] = dict(by_id=by_id,data=data)
        print(f"PLAN {group}: 32->1, {spec['train_windows']}/{spec['validation_windows']} train/val, {len(old)} matched test targets",flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists() and base.read(OUT/'protocol.json') != plan:
        raise ValueError('Locked protocol changed; use a new output directory')
    base.write_json(OUT/'protocol.json',plan)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    budget = pd.DataFrame(budgets)
    old_budget = pd.read_csv(PRIOR/'label_budget.csv')
    label_keys = ['dataset','fold_id','split','battery_id','index','target_cycle']
    pd.testing.assert_frame_equal(budget[label_keys].reset_index(drop=True),old_budget[label_keys].reset_index(drop=True))
    budget.to_csv(OUT/'label_budget.csv',index=False)
    runtime=OUT/'.runtime';runtime.mkdir(exist_ok=True)
    (runtime/'executed_runner.py').write_bytes(Path(__file__).read_bytes())
    return cfg,plan,prepared


def write_report(plan, summary):
    frames=[]
    for length,folder in [(64,PRIOR),(32,OUT)]:
        for group in plan['groups']:
            p=pd.read_csv(folder/group/'predictions.csv.gz')
            frames.append(paired_metrics(p[p.method.isin(METHODS)]))
    battery=pd.concat(frames,ignore_index=True)
    paired=summarize_paired(battery)
    comparison=compare_lengths(battery)
    battery.to_csv(OUT/'lastvalue_battery_metrics.csv',index=False)
    paired.to_csv(OUT/'lastvalue_comparison.csv',index=False)
    comparison.to_csv(OUT/'context32_vs64_battery.csv',index=False)
    context=comparison.groupby(['dataset','method','phase'],sort=False).agg(
        n_batteries=('battery_id','size'),mse_skill_32_vs_64=('mse_skill_32_vs_64','mean'),
        median_skill=('mse_skill_32_vs_64','median'),wins=('mse_skill_32_vs_64',lambda s:int(s.gt(0).sum())))
    context.to_csv(OUT/'context32_vs64_summary.csv')
    make_comparison_plot(paired)
    selected=pd.read_csv(OUT/'selected_checkpoints.csv')
    coverage=pd.read_csv(OUT/'coverage.csv')
    counts=coverage[coverage.role.eq('held_out_test')].groupby('dataset').agg(
        records=('battery_id','size'),targets=('n_targets','sum'),minimum=('min_context_observed','min'),
        missing=('targets_with_missing_context','sum'))
    lines=['# C2 현재 SOH: 입력 이력 64개에서 32개로 축소','',
        '**32→1 실험이다.** 과거 t−32~t−1의 SOH 기록 위치로 y_t 하나를 예측한다. 64→32 미래 다중 예측이 아니다.','',
        '- Linear는 이번 실험과 비교표에서 제외했다. LastValue, C2 zero-shot, C2 head-only만 비교한다.',
        '- reviewed_native_v3의 Q/Qref·SOH·정제 마스크를 그대로 사용했다. 64개 입력 창의 뒤쪽 32개 위치만 남겼으며 결측은 NaN으로 유지했다. 보간하거나 시간을 압축하지 않았다.',
        '- 64→1 실험과 같은 배터리 분할·학습/검증 정답·97,868개 테스트 정답 시점이다. 32개로 예측 가능해진 초기 시점을 추가하지 않았다. LastValue 예측도 두 입력 길이에서 동일함을 확인했다.',
        '- NASA/CALCE 4-fold, 나머지는 기존 고정 분할. 각 fold는 학습 배터리 2개×4 labels=8개, 검증 배터리 1개×4 labels=4개다. 테스트 배터리 자체를 calibration한 실험은 아니다.',
        '- 각 fold/LR는 순정 C2에서 독립 시작하고 output_patch_embedding만 학습했다. LR 1e-6/1e-5, 각각 100 steps, batch 4, seed 42. 검증 손실과 분위수 역전 guard로 선택하며 step 0도 허용한다. 모든 head를 고른 뒤 테스트를 실행했다.',
        '- 실제 과거 SOH로 매 시점 입력을 갱신하는 rolling one-step이다. CALCE의 관측 간 누락 물리 cycle과 MIT CV 계측 불일치에 대한 기존 제한은 그대로다.','',
        '## LastValue 대비 성능','',
        '배터리 기록마다 `1 − MSE_model / MSE_LastValue`를 계산한 뒤 동일 가중 평균했다. 아래는 그 값을 100배 한 **MSE 감소율(%)**이다. 양수는 개선, 음수는 악화다. 평균 RMSE의 비율이나 R²가 아니다. 분모 0은 NaN으로 두고 CSV에 제외 수를 기록한다.','',
        '| 데이터셋 | 64 zero-shot | 64 head-only | 32 zero-shot | 32 head-only |',
        '|---|---:|---:|---:|---:|']
    for group in plan['groups']:
        s=paired[paired.dataset.eq(group)&paired.phase.eq('all')].set_index(['context_length','method'])
        vals=[100*s.loc[(l,m),'mse_skill_lastvalue'] for l in (64,32) for m in ('zero-shot','head-only')]
        lines.append('| '+group+' | '+' | '.join(f'{v:+.2f}%' for v in vals)+' |')
    lines+=['','![LastValue 대비 MSE 감소율](comparison.png)','','## 32→1 절대 오차','',
        'RMSE·MAE는 SOH 비율 단위이며 배터리 기록별 오차의 동일 가중 평균이다. R²는 summary.csv의 보조 지표로만 남겼다.','',
        '| 데이터셋 | LastValue RMSE | Zero-shot RMSE | Head-only RMSE | Head-only MAE |',
        '|---|---:|---:|---:|---:|']
    for group in plan['groups']:
        s=summary[summary.dataset.eq(group)&summary.phase.eq('all')].set_index('method')
        lines.append('| '+group+' | '+' | '.join(f'{s.loc[m,"rmse"]:.6f}' for m in METHODS)+f' | {s.loc["head-only","mae"]:.6f} |')
    lines+=['','## 입력 축소와 head 학습의 효과','',
        '입력 축소 효과는 같은 모델 조건의 32개 입력 MSE를 64개 입력 MSE와 비교한다. Head 학습 효과는 32개 입력 안에서 head-only를 zero-shot과 비교한다. 아래도 배터리 기록별 감소율의 평균이며 양수가 개선이다.','',
        '| 데이터셋 | 32 vs 64 zero-shot | 32 vs 64 head-only | 32 head vs zero-shot |',
        '|---|---:|---:|---:|']
    for group in plan['groups']:
        z=context.loc[(group,'zero-shot','all'),'mse_skill_32_vs_64']
        h=context.loc[(group,'head-only','all'),'mse_skill_32_vs_64']
        d=paired[paired.dataset.eq(group)&paired.context_length.eq(32)&paired.method.eq('head-only')&paired.phase.eq('all')].iloc[0]
        lines.append(f'| {group} | {100*z:+.2f}% | {100*h:+.2f}% | {100*d.mse_skill_zero:+.2f}% |')
    lines+=['','단일 seed의 점추정이다. 상세 CSV에 배터리별 값, 중앙값, 개선한 기록 수, 분모 0 제외 수를 함께 기록했다. 일부 MIT 물리 배터리는 연속 기록이 나뉘어 있어 여기의 동일 가중 단위는 기존 실험과 같은 배터리 기록이다.','',
        '## 평가 범위','', '| 데이터셋 | 테스트 기록 | 정답 시점 | 최소 유효 입력 / 32 | 결측 입력이 있는 시점 |','|---|---:|---:|---:|---:|']
    for group in plan['groups']:
        c=counts.loc[group]
        lines.append(f'| {group} | {c.records} | {c.targets:,} | {c.minimum} | {c.missing:,} |')
    lines+=['','## 선택한 head','', '| 데이터셋 / fold | LR | step |','|---|---:|---:|']
    for r in selected.itertuples():lines.append(f'| {r.dataset} / {r.fold_id} | {r.learning_rate:g} | {r.step} |')
    lines+=['','step 0은 검증에서 학습 전 순정 가중치가 선택되어 zero-shot과 같은 예측을 썼다는 뜻이다. 분위수는 사후 정렬하지 않으며 역전이 있으면 구간 지표에 드러낸다.','',
        '## 대표 예측','', '대표 배터리는 이전과 같은 첫 테스트 배터리다. 전체 측정 곡선을 보이고 예측선만 평가 가능한 시점에 표시한다.','']
    for group in plan['groups']:lines += [f'![{group}]({group}/representative.png)','']
    lines+=['## 상세 파일','',
        '[LastValue 대비 전체·구간별 비교](lastvalue_comparison.csv) · [배터리별 비교](lastvalue_battery_metrics.csv) · [입력 32 vs 64 비교](context32_vs64_summary.csv)',
        '[절대 오차](summary.csv) · [평가 범위](coverage.csv) · [학습 정답](label_budget.csv) · [프로토콜](protocol.json) · [검증](verification.json)',
        '재현: `.venv\\Scripts\\python.exe scripts/run_c2_context32.py --stage all`','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def make_comparison_plot(paired):
    plt=base.plt
    groups=list(base.GROUPS)
    fig,axes=plt.subplots(1,2,figsize=(12,5.7),sharey=True)
    for ax,method,title in zip(axes,['zero-shot','head-only'],['C2 zero-shot','C2 head-only']):
        for length,offset,color in [(64,-.17,'#90a4ae'),(32,.17,'#2467ad')]:
            d=paired[paired.context_length.eq(length)&paired.method.eq(method)&paired.phase.eq('all')].set_index('dataset').loc[groups]
            bars=ax.barh(np.arange(len(groups))+offset,100*d.mse_skill_lastvalue,height=.29,color=color,label=f'{length} past positions')
            for bar,v in zip(bars,100*d.mse_skill_lastvalue):
                ax.text(v+(1 if v>=0 else -1),bar.get_y()+bar.get_height()/2,f'{v:+.1f}%',ha='left' if v>=0 else 'right',va='center',fontsize=8)
        ax.axvline(0,color='#555555',lw=.8)
        vals=100*paired[paired.method.eq(method)&paired.phase.eq('all')].mse_skill_lastvalue
        ax.set_xlim(min(-15,float(vals.min())-12),max(88,float(vals.max())+12))
        ax.set(title=title,xlabel='MSE reduction vs LastValue (%) — higher is better')
        ax.set_yticks(np.arange(len(groups)),groups)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='lower right',fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('Past SOH history: 64 vs 32 positions -> one target',fontsize=14)
    fig.text(.5,.015,'Same 97,868 targets; compute the MSE ratio per battery record, then average. Negative = worse than LastValue.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.95));fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)


def verify(plan):
    check_frozen_prior()
    for path,digest in plan['source_sha256'].items():assert base.sha256(ROOT/path)==digest,path
    battery=pd.read_csv(OUT/'battery_metrics.csv')
    paired=pd.read_csv(OUT/'lastvalue_battery_metrics.csv')
    total=0
    for group,spec in plan['groups'].items():
        assert base.sha256(BUNDLES/group/'soh.csv')==spec['data_sha256']
        p=pd.read_csv(OUT/group/'predictions.csv.gz')
        assert set(p.method)==set(METHODS) and set(p.context_length)=={32}
        old=pd.read_csv(PRIOR/group/'predictions.csv.gz')
        old=old[old.method.eq('LastValue')].sort_values(['battery_id','target_cycle'])
        ref=p[p.method.eq('LastValue')].sort_values(['battery_id','target_cycle'])
        for col in ('battery_id','target_cycle','target','prediction','phase'):
            np.testing.assert_array_equal(ref[col].to_numpy(),old[col].to_numpy())
        records,_,_,_=base.prepare_reviewed_windows(BUNDLES/group,base.ZeroShotConfig(**plan['master_window_config']))
        for record in records:
            b=record['battery_id']
            if b not in set(ref.battery_id):continue
            w=record['windows'];x=w['context'][:,-32:]
            last=np.where(np.isfinite(x),np.arange(32),-1).max(axis=1)
            assert (last>=0).all()
            for method in METHODS:
                a=p[p.battery_id.eq(b)&p.method.eq(method)].sort_values('target_cycle')
                np.testing.assert_array_equal(a.target_cycle,w['target_cycle'][:,0])
                np.testing.assert_allclose(a.target,w['target'][:,0],rtol=1e-12)
                np.testing.assert_array_equal(a.context_observed,np.isfinite(x).sum(axis=1))
                np.testing.assert_array_equal(a.last_observed_lag,32-last)
                if method=='LastValue':np.testing.assert_allclose(a.prediction,x[np.arange(len(x)),last],rtol=1e-12)
                for phase in ('all','early','middle','late'):
                    part=a if phase=='all' else a[a.phase.eq(phase)]
                    if part.empty:continue
                    baseline=ref[ref.battery_id.eq(b)].set_index('target_cycle').loc[part.target_cycle]
                    e=part.prediction.to_numpy()-part.target.to_numpy()
                    be=baseline.prediction.to_numpy()-baseline.target.to_numpy()
                    s=battery[battery.dataset.eq(group)&battery.battery_id.eq(b)&battery.method.eq(method)&battery.phase.eq(phase)].iloc[0]
                    np.testing.assert_allclose([s.mae,s.rmse],[np.mean(abs(e)),np.sqrt(np.mean(e**2))],rtol=1e-9,atol=1e-12)
                    r=paired[paired.dataset.eq(group)&paired.context_length.eq(32)&paired.battery_id.eq(b)&paired.method.eq(method)&paired.phase.eq(phase)].iloc[0]
                    expected=1-np.sum(e**2)/np.sum(be**2) if np.sum(be**2)>0 else np.nan
                    np.testing.assert_allclose(r.mse_skill_lastvalue,expected,rtol=1e-8,atol=1e-12)
        total+=len(ref)
    summary=pd.read_csv(OUT/'summary.csv').set_index(['dataset','context_length','method','phase']).sort_index()
    expected=battery.groupby(['dataset','context_length','method','phase'])[['mae','rmse']].mean().sort_index()
    np.testing.assert_allclose(summary[['mae','rmse']],expected,rtol=1e-10,atol=1e-12)
    comparison=pd.read_csv(OUT/'lastvalue_comparison.csv').set_index(['dataset','context_length','method','phase']).sort_index()
    expected=paired.groupby(['dataset','context_length','method','phase'])[['mse_skill_lastvalue','mae_skill_lastvalue','mse_skill_zero']].mean().sort_index()
    np.testing.assert_allclose(comparison[expected.columns],expected,rtol=1e-10,atol=1e-12,equal_nan=True)
    assert total==97868
    return dict(matched_targets_per_condition=total,conditions=3,context_length=32,prediction_length=1,
                identical_targets_to_context64=True,identical_lastvalue_to_context64=True,
                identical_source_labels_to_context64=True,reviewed_data_and_context_masks_verified=True,
                independent_metric_recalculation=True,linear_excluded=True,verified_at=base.now())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','train','finish','report','all'],default='all')
    args=parser.parse_args();configure()
    cfg,plan,prepared=prepare()
    if args.stage=='prepare':return
    if args.stage=='report':
        base.summarize(plan)
        for group,spec in plan['groups'].items():base.plot_predictions(group,pd.read_csv(OUT/group/'predictions.csv.gz'),spec)
        base.write_json(OUT/'verification.json',verify(plan));return
    state=base.read(OUT/'status.json') if (OUT/'status.json').exists() else dict(started_at=base.now())
    pipeline=base.load_model(cfg)
    initial={n:p.detach().cpu().clone() for n,p in pipeline.model.named_parameters() if n.startswith('output_patch_embedding.')}
    base.update(state,status='loaded',device=str(pipeline.model.device),protocol_sha256=base.sha256(OUT/'protocol.json'))
    try:
        if args.stage in ('train','all'):base.fit_all(pipeline,initial,plan,prepared,state)
        if args.stage in ('finish','all'):
            if 'all_heads_locked_at' not in state:raise ValueError('Lock all heads before testing')
            base.score(pipeline,cfg,initial,plan,prepared,state)
    except Exception as error:
        base.update(state,status='failed',error=repr(error));raise


if __name__=='__main__':main()
