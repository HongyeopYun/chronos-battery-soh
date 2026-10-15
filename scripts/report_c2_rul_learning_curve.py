"""Independently audit learning-curve artifacts, then produce comparison plots."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.battery_zeroshot import sha256
from src.c2_v4_experiments import load_cells, DirectRUL
from src.c2_rul_learning_curve import assemble, cell_weights, constants, fit_ridge, predict_ridge
from scripts.run_c2_v4_suite import read, write, now, load_model, tensor

NAMES={'nasa_aging':'NASA','calce_cs2_1c':'CALCE','mit_batch2':'MIT Batch 2','xjtu_2c':'XJTU 2C'}
COLORS={'head':'#903bb0','ridge':'#078776','source_mean':'#ce6a17','source_median':'#657487'}


def audit(out,gpu=False,cache_root=None):
    p=read(out/'protocol.json'); ph=sha256(out/'protocol.json')
    finish=read(out/'execution_finished.json');lock=read(out/'selection_lock.json')
    assert finish['protocol_sha256']==lock['protocol_sha256']==ph
    assert finish['backbone_frozen_verified'] and finish['selection_unchanged']
    assert finish['predictions_sha256']==sha256(out/'predictions.csv')
    assert all(sha256(ROOT/k)==v for k,v in p['source_sha256'].items())
    cells,fingerprints=load_cells(ROOT/'data/processed/reviewed_v4/forecast_safe')
    assert fingerprints==p['data_artifact_sha256']
    predictions=pd.read_csv(out/'predictions.csv',dtype={'budget':str})
    labels=pd.read_csv(out/'label_budget.csv')
    model=None
    if gpu:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        model=DirectRUL(load_model(cache_root).model,'head',100.,100.)
    native_points=0;expected_labels=0;expected_rows=0
    for c in p['cases']:
        directory=out/'checkpoints'/c['case_id'];s=read(directory/'selected.json')
        assert lock['selected_sha256'][(directory/'selected.json').relative_to(out).as_posix()]==sha256(directory/'selected.json')
        assert s['protocol_sha256']==ph and s['weights_sha256']==sha256(directory/'weights.pt')
        assert s['reload_verified'] and s['trainable_parameters']==51015
        roles=[set(c[k]) for k in ('train_ids','validation_ids','test_ids')]
        assert not any(roles[i]&roles[j] for i,j in ((0,1),(0,2),(1,2)))
        arrays={}
        for split,ids,budget,stride in [('train',c['train_ids'],c['budget'],1),('validation',c['validation_ids'],'all',1),('test',c['test_ids'],'all',16)]:
            a=assemble(cells,c['dataset'],ids,budget,stride);arrays[split]=a
            if split!='test':
                z=labels[(labels.case_id==c['case_id'])&(labels.split==split)]
                np.testing.assert_array_equal(z.battery_id,a['battery'])
                np.testing.assert_array_equal(z.origin_cycle,a['cycle'])
                np.testing.assert_allclose(z.target,a['target'],rtol=1e-12,atol=1e-12)
                expected_labels+=len(z)
        assert s['train_labels']==len(arrays['train']['target']) and s['validation_labels']==len(arrays['validation']['target'])
        const=constants(arrays['train']['target'],arrays['train']['battery'])
        for k,v in const.items():assert np.isclose(v,s['constants'][k],atol=1e-10)
        ridge=fit_ridge(arrays['train'],arrays['validation'])
        assert ridge['alpha']==s['ridge']['alpha']
        np.testing.assert_allclose(ridge['coef'],s['ridge']['coef'],rtol=1e-7,atol=1e-7)
        a=arrays['test'];d=predictions[predictions.case_id==c['case_id']]
        assert set(d.method)=={'head','ridge','source_mean','source_median'}
        assert set(d.battery_id)==set(c['test_ids'])
        assert (d.seed==c['seed']).all() and (d.budget==c['budget']).all()
        assert (d.n_train_cells==c['n_train_cells']).all() and (d.dataset==c['dataset']).all()
        for method in ('head','ridge','source_mean','source_median'):
            z=d[d.method==method]
            np.testing.assert_array_equal(z.battery_id,a['battery'])
            np.testing.assert_array_equal(z.origin_cycle,a['cycle'])
            np.testing.assert_array_equal(z.true_rul,a['target'])
            assert np.isfinite(z.predicted_rul).all() and (z.predicted_rul>=0).all()
            expected_rows+=len(z)
            if method in const:np.testing.assert_allclose(z.predicted_rul,const[method],rtol=1e-12,atol=1e-12)
            if method=='ridge':np.testing.assert_allclose(z.predicted_rul,predict_ridge(s['ridge'],a['context']),rtol=1e-9,atol=1e-8)
        state=torch.load(directory/'weights.pt',weights_only=True,map_location='cpu')
        assert all(torch.isfinite(t).all() for t in state.values())
        if gpu:
            model.head.load_state_dict(state);model.scale=s['scale'];model.eval()
            # Recompute sampled test predictions through the original native
            # encoder+head, without the training runner's feature cache.
            idx=np.unique(np.linspace(0,len(a['target'])-1,min(5,len(a['target'])),dtype=int))
            with torch.no_grad():q=model(tensor(a['context'][idx])).cpu().numpy()
            np.testing.assert_allclose(q,d[d.method=='head'].predicted_rul.to_numpy()[idx],rtol=2e-4,atol=.03)
            native_points+=len(idx)
    assert len(predictions)==expected_rows and len(labels)==expected_labels
    # Every budget/source count uses identical test labels and timestamps.
    for group,d in predictions.groupby('dataset'):
        reference=None
        for _,z in d[d.method=='head'].groupby(['seed','n_train_cells','budget']):
            values=z[['battery_id','origin_cycle','true_rul']].sort_values(['battery_id','origin_cycle']).reset_index(drop=True)
            if reference is None:reference=values
            else:pd.testing.assert_frame_equal(reference,values)
    result=dict(status='passed',verified_at=now(),protocol_sha256=ph,cases=len(p['cases']),
        weight_files=len(p['cases']),predictions=len(predictions),label_rows=len(labels),
        gpu_native_reloaded_test_points=native_points,cell_disjoint=True,matched_test_origins=True,
        constants_and_ridge_recomputed=True,source_and_data_hashes=True)
    write(out/('verification_gpu.json' if gpu else 'verification_local.json'),result)
    return predictions,p


def metrics(pred,out):
    rows=[]
    keys=['dataset','fold','seed','n_train_cells','budget','method','battery_id']
    for key,d in pred.groupby(keys,sort=True):
        y=d.true_rul.to_numpy();error=d.predicted_rul.to_numpy()-y
        sst=float(np.sum((y-y.mean())**2))
        rows.append(dict(zip(keys,key),n_origins=len(y),mae=float(np.mean(abs(error))),
            rmse=float(np.sqrt(np.mean(error**2))),bias=float(error.mean()),
            r2=1-float(np.sum(error**2))/sst if sst>0 and len(y)>1 else np.nan))
    cells=pd.DataFrame(rows);cells.to_csv(out/'cell_metrics.csv',index=False)
    seed=cells.groupby(['dataset','seed','n_train_cells','budget','method'],as_index=False).agg(
        mae=('mae','mean'),rmse=('rmse','mean'),mean_cell_r2=('r2','mean'),bias=('bias','mean'),n_cells=('battery_id','size'),n_origins=('n_origins','sum'))
    seed.to_csv(out/'seed_metrics.csv',index=False)
    summary=seed.groupby(['dataset','n_train_cells','budget','method'],as_index=False).agg(
        mae=('mae','mean'),mae_seed_sd=('mae','std'),rmse=('rmse','mean'),mean_cell_r2=('mean_cell_r2','mean'),bias=('bias','mean'),n_cells=('n_cells','first'))
    summary.to_csv(out/'summary.csv',index=False)
    return summary


def figures(summary,pred,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    budgets=['4','16','all']
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,(group,name) in zip(axes.ravel(),NAMES.items()):
        n=16 if group=='mit_batch2' else 2
        d=summary[(summary.dataset==group)&(summary.n_train_cells==n)]
        for method in COLORS:
            z=d[d.method==method].set_index('budget').loc[budgets]
            ax.errorbar(range(3),z.mae,yerr=z.mae_seed_sd,marker='o',capsize=3,color=COLORS[method],label=method)
        ax.set(title=f'{name} | {n} training cells',xticks=range(3),xticklabels=['4','16','All'],xlabel='Labels per training cell',ylabel='Equal-cell MAE (record positions)')
        ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('RUL label learning curves | fixed test cells | mean ± SD across 3 seeds')
    fig.tight_layout();fig.savefig(out/'label_learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.3))
    for ax,budget in zip(axes,budgets):
        d=summary[(summary.dataset=='mit_batch2')&(summary.budget==budget)]
        for method in COLORS:
            z=d[d.method==method].sort_values('n_train_cells')
            ax.errorbar(z.n_train_cells,z.mae,yerr=z.mae_seed_sd,marker='o',capsize=3,color=COLORS[method],label=method)
        ax.set(title=f'{budget} labels per training cell',xticks=[2,4,8,16],xlabel='Training cells',ylabel='Equal-cell MAE (record positions)')
        ax.grid(alpha=.2);ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=9)
    fig.suptitle('MIT Batch 2 | same 20 test cells | nested source cells | mean ± SD across 3 seeds')
    fig.tight_layout();fig.savefig(out/'mit_cell_learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7.5))
    for ax,(group,name) in zip(axes.ravel(),NAMES.items()):
        d=pred[(pred.dataset==group)&(pred.seed==42)&(pred.n_train_cells==(16 if group=='mit_batch2' else 2))]
        battery=sorted(d.battery_id.unique())[0];d=d[d.battery_id==battery]
        ref=d[(d.budget=='all')&(d.method=='head')]
        ax.plot(ref.origin_cycle,ref.true_rul,color='black',label='True RUL',lw=2)
        for budget,color in [('4','#db8b22'),('16','#2388b5'),('all','#903bb0')]:
            z=d[(d.budget==budget)&(d.method=='head')]
            ax.plot(z.origin_cycle,z.predicted_rul,'o-',ms=3,color=color,label=f'Head: {budget} labels/cell')
        z=d[(d.budget=='all')&(d.method=='ridge')]
        ax.plot(z.origin_cycle,z.predicted_rul,'--',color=COLORS['ridge'],label='Ridge: all labels')
        ax.set(title=f'{name} | {battery} | seed 42',xlabel='Prediction origin',ylabel='Remaining record positions')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'representative_rul.png',dpi=180);plt.close(fig)


def markdown_table(df):
    columns=list(df.columns)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def report(summary,protocol,out):
    chosen=summary[((summary.dataset=='mit_batch2')&(summary.n_train_cells==16))|((summary.dataset!='mit_batch2')&(summary.n_train_cells==2))]
    table=chosen.pivot(index=['dataset','n_train_cells','budget'],columns='method',values='mae').reset_index()
    mit=summary[summary.dataset=='mit_batch2'].pivot(index=['n_train_cells','budget'],columns='method',values='mae').reset_index()
    table=table.assign(order=table.budget.map({'4':0,'16':1,'all':2})).sort_values(['dataset','n_train_cells','order']).drop(columns='order')
    mit=mit.assign(order=mit.budget.map({'4':0,'16':1,'all':2})).sort_values(['n_train_cells','order']).drop(columns='order')
    observations=[]
    for group,name in NAMES.items():
        n=16 if group=='mit_batch2' else 2
        d=summary[(summary.dataset==group)&(summary.n_train_cells==n)]
        h=d[d.method=='head'].set_index('budget')
        r=d[(d.method=='ridge')&(d.budget=='all')].iloc[0]
        observations.append(f"- {name} 학습{n}셀: Head MAE 4정답 {h.loc['4','mae']:.2f} → 16정답 {h.loc['16','mae']:.2f} → 전체 {h.loc['all','mae']:.2f}. 동일 전체 정답 Ridge는 {r.mae:.2f}.")
    h=summary[(summary.dataset=='mit_batch2')&(summary.method=='head')&(summary.budget=='all')].set_index('n_train_cells')
    observations.append('- MIT 전체 정답 조건의 Head MAE: '+ ' → '.join(f'{n}셀 {h.loc[n,"mae"]:.2f}' for n in (2,4,8,16))+'.')
    lines=['# RUL 학습 셀 수·정답 수 후속 실험','',
        '새 전처리와 SOH 80% EOL 정의를 유지한 Head-only 진단 실험이다. Full·LoRA·SOH 현재 예측을 재실행한 결과가 아니다. 기존 v4 결과는 보존했다.','',
        '## 비교 조건','',
        '- 입력: 최근 SOH 32개. 미래 SOH 궤적 없이 RUL을 직접 출력한다. 셀의 cycle 번호나 미래 EOL은 모델 입력에 넣지 않는다.',
        '- NASA·CALCE·XJTU 2C: EOL 관측 셀 4개씩. 기존 LOBO 4분할, 학습2/검증1/시험1을 유지한다. 이 자료만으로 학습 셀 수를 늘리는 실험은 불가능하다.',
        '- MIT Batch 2: 기존 학습2개를 유지하고, 기존 시험40개에서 고정 RNG로 공통 시험20개·추가 검증2개·학습 후보18개를 나눈다. 기존 검증1개와 합쳐 검증3개다. 수명이나 이전 오차를 기준으로 셀을 선택하지 않았다.',
        '- MIT 학습 셀 수 2·4·8·16개. 모든 데이터셋에서 셀당 정답4·16·전체를 비교한다. 각 단계의 정답과 MIT 학습 셀은 중첩된다.',
        '- seed42·43·44: Head 초기화/미니배치 반복; MIT는 추가 학습 셀 순서도 바뀐다. 시험 셀은 고정되어 독립된 시험 집단 3개가 아니다.',
        '- 셀당 전체 정답은 적격 pre-EOL stride1 관측이다. 인접 창은 상관되어 있고 하나의 완료된 셀 EOL에서 유도되므로 독립 배터리 수가 늘어나는 것은 아니다. 전체 정답 조건은 few-shot으로 부르지 않는다.',
        '- 모든 조건에 동일한 1,000 steps × LR2개, batch16, 셀 균등 샘플링. 모든 검증 시점의 셀 균등 MAE로 step0 포함 선택하며 전체144개 모델 선택 후 시험 예측한다.',
        '- 기존 실험의 200 steps·검증4정답과 달라, 이전 보고서와의 변화 전부를 데이터 증가 효과로 설명할 수 없다. 이 보고서 안의 통제된 조건끼리 비교한다.',
        '- 기준: 학습 셀 균등 RUL 평균/중앙값과, 동일 SOH 입력의 수준·기울기 등 6개 특징을 쓰는 Ridge. Ridge의 정규화는 학습 자료로만 계산하고 alpha는 검증으로 선택한다.',
        '- MAE/RMSE는 셀별 계산 후 같은 가중치로 평균한다. R²는 셀별 여러 단일 RUL 정답·예측을 모아 계산하고 음수를 유지한다. 표의 점수는 3seed 평균, 그래프 막대는 seed 간 표준편차이며 신뢰구간이 아니다.',
        '- CALCE 축은 보존된 관측 레코드이며 파일 사이 미관측 물리 cycle 수를 알 수 없다. 다른 데이터셋과 MAE 숫자를 직접 비교하지 않는다.',
        '- 기존에 살펴본 데이터로 하는 후속 탐색이다. 새로운 외부 검증셋의 성능을 입증하지 않는다. 우측 검열 셀과 데이터셋 간 전이는 이 진단에서 제외했다.','',
        '## 관측된 변화','',*observations,'',
        '자료량을 늘릴수록 반드시 좋아진다고 가정하지 않는다. 큰 정답 예산에서의 정체·악화와 Ridge 대비 우위 여부를 함께 확인해야 한다. 반복 간 산포는 표준편차이며 통계적 유의성 검정 결과가 아니다.','',
        '## 셀당 정답 수 비교 — MAE','',markdown_table(table),'',
        '![정답 수 학습곡선](label_learning_curves.png)','',
        '## MIT 학습 셀 수 비교 — MAE','',markdown_table(mit),'',
        '![MIT 학습 셀 수 학습곡선](mit_cell_learning_curves.png)','',
        '## 대표 셀의 RUL 갱신','',
        '각 데이터셋 시험 셀의 문자열 ID 첫 번째, seed42를 고정 표시한다. 결과를 보고 좋은 셀을 골라 표시하지 않는다. 제외된 관측을 보간하거나 평활화하지 않는다.','',
        '![대표 셀](representative_rul.png)','',
        '[전체 지표·seed 표준편차·R²](summary.csv) · [seed별 지표](seed_metrics.csv) · [셀별 지표](cell_metrics.csv) · [고정 프로토콜](protocol.json)','',
        '## 재현','',
        '```powershell',
        'python scripts/run_c2_rul_learning_curve.py --out results/c2-battery/rul-learning-curve-v1',
        'python scripts/report_c2_rul_learning_curve.py --out results/c2-battery/rul-learning-curve-v1 --gpu-verify',
        '```','',
        '원본 새 전처리 데이터와 고정 Chronos revision 캐시가 필요하며 학습은 CUDA에서 실행된다. 완료 폴더는 덮어쓰지 않는다. 동일 프로토콜의 중단된 학습은 저장된 체크포인트를 검증한 뒤 재개한다.','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gpu-verify',action='store_true');parser.add_argument('--cache-root',type=Path)
    args=parser.parse_args();out=args.out.resolve()
    pred,p=audit(out,args.gpu_verify,args.cache_root)
    summary=metrics(pred,out);figures(summary,pred,out);report(summary,p,out)
    if args.gpu_verify:
        write(out/'complete.json',dict(status='complete',completed_at=now(),cases=144,
            protocol_sha256=sha256(out/'protocol.json'),report_source_sha256=sha256(Path(__file__)),
            gpu_verification_sha256=sha256(out/'verification_gpu.json')))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()
