"""Audit matched context sweep and render all saved test cells."""
import argparse,json,sys
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.battery_zeroshot import sha256
from src.c2_rul_context_nominal import load,make_cases,assemble,LABEL_FILE
from src.c2_rul_learning_curve import constants,fit_ridge,predict_ridge
from scripts.run_c2_v4_suite import read,write,now

def audit(out):
    p=read(out/'protocol.json');finish=read(out/'execution_finished.json');lock=read(out/'selection_lock.json')
    assert finish['protocol_sha256']==lock['protocol_sha256']==sha256(out/'protocol.json')
    assert finish['predictions_sha256']==sha256(out/'predictions.csv')
    assert finish['backbone_frozen_verified'] and finish['selection_unchanged'] and finish['native_reload_test_points']>0
    assert all(sha256(ROOT/k)==v for k,v in p['source_sha256'].items())
    assert sha256(ROOT/LABEL_FILE)==p['label_sha256']
    cells,hashes,eol=load(ROOT);assert hashes==p['data_artifact_sha256'];assert make_cases(cells)==p['cases']
    d=pd.read_csv(out/'predictions.csv');labels=pd.read_csv(out/'label_budget.csv')
    expected=0;label_count=0
    for c in p['cases']:
        directory=out/'checkpoints'/c['case_id'];s=read(directory/'selected.json')
        assert s['protocol_sha256']==sha256(out/'protocol.json') and s['weights_sha256']==sha256(directory/'weights.pt')
        assert lock['selected_sha256'][(directory/'selected.json').relative_to(out).as_posix()]==sha256(directory/'selected.json')
        assert s['reload_verified'] and s['trainable_parameters']==51015 and s['train_labels']==32
        weights=torch.load(directory/'weights.pt',map_location='cpu',weights_only=True)
        assert all(torch.isfinite(v).all() for v in weights.values())
        arrays={}
        for split,ids,budget in [('train',c['train_ids'],'16'),('validation',c['validation_ids'],'all')]:
            a=assemble(cells,c['dataset'],ids,budget,length=c['length'],start=c['start']);arrays[split]=a
            z=labels[(labels.case_id==c['case_id'])&(labels.split==split)]
            np.testing.assert_array_equal(z.battery_id,a['battery']);np.testing.assert_array_equal(z.origin_cycle,a['cycle']);np.testing.assert_array_equal(z.target,a['target']);label_count+=len(z)
        const=constants(arrays['train']['target'],arrays['train']['battery'])
        ridge=fit_ridge(arrays['train'],arrays['validation'])
        assert ridge['alpha']==s['ridge']['alpha'];np.testing.assert_allclose(ridge['coef'],s['ridge']['coef'],rtol=1e-7,atol=1e-7)
        targets=[(c['dataset'],c['test_ids'],'within_dataset')]+[(g,ids,'transfer' if g!=c['dataset'] else 'censored') for g,ids in c['extra_tests'].items()]
        for g,ids,scope in targets:
            a=assemble(cells,g,ids,length=c['length'],start=c['start'],stride=16)
            z=d[(d.case_id==c['case_id'])&(d.dataset==g)&(d.scope==scope)]
            assert set(z.method)=={'head','ridge','source_mean','source_median'}
            for method,q in z.groupby('method',sort=False):
                np.testing.assert_array_equal(q.battery_id,a['battery']);np.testing.assert_array_equal(q.origin_cycle,a['cycle'])
                np.testing.assert_allclose(q.true_rul,a['target'],equal_nan=True)
                assert np.isfinite(q.predicted_rul).all() and (q.predicted_rul>=0).all()
                if method in const:np.testing.assert_allclose(q.predicted_rul,const[method])
                if method=='ridge':np.testing.assert_allclose(q.predicted_rul,predict_ridge(ridge,a['context']),rtol=1e-7,atol=1e-6)
                expected+=len(q)
    assert expected==len(d) and label_count==len(labels)
    for _,z in d[d.method=='head'].groupby(['dataset','source_dataset','scope']):
        ref=None
        for _,q in z.groupby(['length','seed']):
            arr=q[['battery_id','origin_cycle','true_rul']].sort_values(['battery_id','origin_cycle']).reset_index(drop=True)
            if ref is None:ref=arr
            else:pd.testing.assert_frame_equal(arr,ref)
    recovered=d[(d.scope=='transfer')&d.dataset.isin(['mit_batch1','mit_batch3'])&d.event_observed]
    assert len(recovered[['dataset','battery_id']].drop_duplicates())==81
    cens=d[(d.dataset=='mit_batch1')&~d.event_observed]
    assert set(cens.battery_id)=={'b1c8','b1c10','b1c12','b1c13','b1c22'}
    write(out/'verification.json',dict(status='passed',verified_at=now(),models=len(p['cases']),predictions=len(d),
        label_rows=len(labels),matched_origins=True,source_data_hashes=True,mit_restored_exact_labels=81,
        mit_still_censored=5,native_reload_test_points=finish['native_reload_test_points']))
    return d

def render(out,d):
    rows=[]
    keys=['dataset','source_dataset','scope','length','seed','method','battery_id']
    for key,z in d[d.event_observed].groupby(keys):
        err=z.predicted_rul-z.true_rul;den=((z.true_rul-z.true_rul.mean())**2).sum()
        rows.append(dict(zip(keys,key),mae=float(abs(err).mean()),rmse=float(np.sqrt((err**2).mean())),r2=float(1-(err**2).sum()/den) if den>0 else np.nan,n=len(z)))
    m=pd.DataFrame(rows);m.to_csv(out/'cell_metrics.csv',index=False)
    seed=m.groupby(keys[:-1])[['mae','rmse','r2']].mean().reset_index();seed.to_csv(out/'seed_metrics.csv',index=False)
    summary=seed.groupby(['dataset','source_dataset','scope','length','method'])[['mae','rmse','r2']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index();summary.to_csv(out/'summary.csv',index=False)
    h=summary[(summary.method=='head')&(summary.scope=='within_dataset')]
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,(g,z) in zip(axes.ravel(),h.groupby('dataset',sort=False)):
        ax.errorbar(z.length,z.mae_mean,yerr=z.mae_std,marker='o',capsize=4)
        ax.set(title=g,xlabel='Context length',ylabel='Cell-macro MAE (record positions)');ax.set_xticks(z.length);ax.grid(alpha=.2)
    fig.suptitle('Direct RUL | Head-only | matched origins | mean +/- SD across 3 seeds\nMIT: official nominal80 EOL; NASA uses L8/16/32 only')
    fig.tight_layout();fig.savefig(out/'context_mae.png',dpi=160);plt.close(fig)
    within=seed[(seed.method=='head')&(seed.scope=='within_dataset')&(seed.dataset!='nasa_aging')]
    macro=within.groupby(['length','seed'])[['mae','rmse','r2']].mean().reset_index()
    macro.to_csv(out/'five_dataset_macro.csv',index=False)
    colors={8:'#0072b2',16:'#d55e00',32:'#009e73',64:'#cc79a7',128:'#e69f00'}
    cards=[];count=0
    for (g,source,scope,b),z in d[d.method=='head'].groupby(['dataset','source_dataset','scope','battery_id']):
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        ref=z[(z.length==z.length.min())&(z.seed==42)].sort_values('origin_cycle');event=bool(ref.event_observed.iloc[0])
        for ax,s in zip(axes.ravel(),[42,43,44,None]):
            ax.plot(ref.origin_cycle,ref.true_rul if event else ref.lower_bound,'k-' if event else 'k--',label='True RUL' if event else 'Follow-up lower bound (NOT true RUL)')
            for L,q in z.groupby('length'):
                piv=q.pivot(index='origin_cycle',columns='seed',values='predicted_rul')
                ax.plot(piv.index,piv.mean(axis=1) if s is None else piv[s],color=colors[L],label=f'L={L}')
                if s is None:ax.fill_between(piv.index,piv.min(axis=1),piv.max(axis=1),color=colors[L],alpha=.09)
            ax.set(title=f'Seed {s}' if s is not None else 'Mean + min-max across seeds (not CI)',xlabel='Current cycle / record position',ylabel='Remaining cycle / record positions',ylim=(0,None));ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'{g} | {b} | {scope} | source={source}\n'+('MIT official nominal80 EOL' if g.startswith('mit') else 'Initial-reference80 EOL'))
        fig.tight_layout(rect=(0,0,1,.96))
        rel=Path('cells')/f'{g}__{source}__{scope}__{b}.png';(out/rel).parent.mkdir(exist_ok=True)
        fig.savefig(out/rel,dpi=130);plt.close(fig);count+=1
        cards.append(f'<article data-group="{g}" data-scope="{scope}" data-cell="{b.lower()}"><h2>{g} · {b} · {scope}</h2><p>Source: {source}</p><a href="{rel.as_posix()}" target="_blank"><img loading="lazy" src="{rel.as_posix()}"></a></article>')
    groups=''.join(f'<option>{g}</option>' for g in sorted(d.dataset.unique()))
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>RUL context · MIT 정격80</title><style>body{font:16px Arial;margin:24px;background:#f5f6f9}header,main{max-width:1300px;margin:auto}article{background:white;padding:18px;margin:20px 0}img{width:100%}select,input{padding:10px;margin:5px}article[hidden]{display:none}.bar{position:sticky;top:0;background:#fff;padding:10px}</style><header><h1>RUL 직접 예측 — 입력 길이 비교</h1><p>Head-only · 학습 셀당 16개 정답 · seed 42/43/44. MIT는 정격용량 1.1Ah의 80%에 해당하는 공식 수명 라벨을 사용했습니다. 기존 검열 81셀은 정답이 복원됐으며, 실제 미도달 5셀은 검열로 유지합니다.</p><p>NASA는 L8/16/32, t≥32. 나머지는 L8/16/32/64/128, t≥128의 동일 시점 비교입니다. 긴 입력 비교는 초기 수명 예측 성능을 뜻하지 않습니다. within_dataset은 배치 내 시험, transfer는 다른 배치에서 학습한 모델입니다. 점선은 정답이 아닌 추적기간 하한입니다.</p><a href="context_mae.png" target="_blank"><img src="context_mae.png"></a><p><a href="summary.csv">성능표 CSV</a> · <a href="report.md">결과 설명</a></p></header><div class="bar"><label>데이터셋 <select id="group"><option value="all">전체</option>GROUPS</select></label><label>평가 <select id="scope"><option value="all">전체</option><option>within_dataset</option><option>transfer</option><option>censored</option></select></label><label>셀 검색 <input id="cell"></label><b id="count"></b></div><main>CARDS</main><script>const a=[...document.querySelectorAll('article')],g=document.querySelector('#group'),s=document.querySelector('#scope'),b=document.querySelector('#cell');function f(){let n=0;for(const x of a){x.hidden=!((g.value==='all'||x.dataset.group===g.value)&&(s.value==='all'||x.dataset.scope===s.value)&&x.dataset.cell.includes(b.value.toLowerCase()));if(!x.hidden)n++;}document.querySelector('#count').textContent=n+'개';}for(const x of [g,s,b])x.addEventListener('input',f);f();</script></html>'''
    (out/'index.html').write_text(html.replace('GROUPS',groups).replace('CARDS','\n'.join(cards)),encoding='utf-8')
    text=['# Direct RUL context sweep','', 'MIT: nominal 1.1Ah x 80%, audited official cycle_life. Other groups retain initial-reference80.',
          '201 heads, frozen C2 backbone, 2 source cells x 16 labels, 3 seeds. Fixed labels/origins across context lengths.',
          'NASA only L8/16/32 due to EOL61 cell. All other groups start at128, so late-survivor evaluation.',
          '81 MIT exact labels restored via official labels, not last-record+1; 5 genuinely censored cells stay censored.',
          'Within-dataset and transfer are separate. Existing historical results unchanged. This is exploratory reused data.',
          '', '| Dataset | L | Head cell-macro MAE | Seed SD |','|---|---:|---:|---:|']
    for r in h.itertuples():text.append(f'| {r.dataset} | {r.length} | {r.mae_mean:.3f} | {r.mae_std:.3f} |')
    text+=['',f'All-cell figures: {count}. Three actual seeds and their mean/range; range is not a confidence interval.']
    (out/'report.md').write_text('\n'.join(text),encoding='utf-8')
    protocol=read(out/'protocol.json')
    source_paths=set(protocol['source_sha256'])|{LABEL_FILE,'scripts/report_c2_rul_context_nominal.py','scripts/export_c2_rul_context_nominal.py','tests/test_c2_rul_context_nominal.py'}
    with ZipFile(out/'source_snapshot.zip','w',ZIP_DEFLATED) as z:
        for rel in sorted(source_paths):z.write(ROOT/rel,rel)
    write(out/'complete.json',dict(status='complete',finished_at=now(),models=201,cell_figures=count))
    files={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and '.runtime' not in p.parts}
    write(out/'artifact_manifest.json',files)
    with ZipFile(out.with_suffix('.zip'),'w',ZIP_DEFLATED,compresslevel=3) as z:
        for p in out.rglob('*'):
            if p.is_file() and '.runtime' not in p.parts:z.write(p,p.relative_to(out).as_posix())
    print(h[['dataset','length','mae_mean','mae_std']].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    d=audit(out);render(out,d)
