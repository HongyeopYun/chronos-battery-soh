"""Completion audit and matched Head/Full/LoRA report; no model training."""
import argparse,sys
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
from src.c2_rul_context_nominal import load,assemble,LABEL_FILE
from scripts.run_c2_v4_suite import read,write,now
REFERENCE=ROOT/'results/c2-battery/rul-context-nominal80-v1'

def report(out):
    out=Path(out);p=read(out/'protocol.json');finish=read(out/'execution_finished.json');lock=read(out/'selection_lock.json')
    assert finish['protocol_sha256']==lock['protocol_sha256']==sha256(out/'protocol.json')
    assert finish['predictions_sha256']==sha256(out/'predictions.csv')
    assert finish['selection_unchanged'] and len(lock['selected_sha256'])==90
    assert read(out/'head_device_parity.json')['status']=='passed'
    assert all(sha256(ROOT/k)==v for k,v in p['source_sha256'].items())
    assert sha256(REFERENCE/'protocol.json')==p['reference_protocol_sha256']
    assert sha256(REFERENCE/'predictions.csv')==p['reference_predictions_sha256']
    cells,hashes,_=load(ROOT);assert hashes==p['data_artifact_sha256'];assert sha256(ROOT/LABEL_FILE)==p['label_sha256']
    d=pd.read_csv(out/'predictions.csv');expected=0
    labels=pd.read_csv(out/'label_budget.csv');nlabels=0
    for c in p['cases']:
        directory=out/'checkpoints'/c['case_id'];s=read(directory/'selected.json')
        assert lock['selected_sha256'][(directory/'selected.json').relative_to(out).as_posix()]==sha256(directory/'selected.json')
        assert s['weights_sha256']==sha256(directory/'weights.pt')
        assert s['protocol_sha256']==sha256(out/'protocol.json') and s['reload_verified'] and s['frozen_verified']
        assert set(s['gradient_audit'])=={str(v) for v in p['backbone_lrs'][c['mode']]}
        assert all(a['backbone_nonzero'] and a['head_nonzero'] for a in s['gradient_audit'].values())
        state=torch.load(directory/'weights.pt',map_location='cpu',weights_only=True)
        names=set(next(iter(s['gradient_audit'].values()))['trainable_names']);assert set(state)==names
        if c['mode']=='lora':assert all('lora_' in n for n in names if n.startswith('backbone.'))
        assert all(torch.isfinite(v).all() for v in state.values());del state
        for split,ids,budget in [('train',c['train_ids'],'16'),('validation',c['validation_ids'],'all')]:
            a=assemble(cells,c['dataset'],ids,budget,length=c['length'],start=c['start'])
            q=labels[(labels.case_id==c['case_id'])&(labels.split==split)]
            np.testing.assert_array_equal(q.battery_id,a['battery']);np.testing.assert_array_equal(q.origin_cycle,a['cycle']);np.testing.assert_allclose(q.target,a['target']);nlabels+=len(q)
        targets=[(c['dataset'],c['test_ids'],'within_dataset')]+[(g,ids,'transfer' if g!=c['dataset'] else 'censored') for g,ids in c['extra_tests'].items()]
        for g,ids,scope in targets:
            a=assemble(cells,g,ids,length=c['length'],start=c['start'],stride=16)
            q=d[(d.case_id==c['case_id'])&(d.dataset==g)&(d.scope==scope)]
            assert set(q.method)=={c['mode']}
            np.testing.assert_array_equal(q.battery_id,a['battery']);np.testing.assert_array_equal(q.origin_cycle,a['cycle']);np.testing.assert_allclose(q.true_rul,a['target'],equal_nan=True)
            assert np.isfinite(q.predicted_rul).all() and (q.predicted_rul>=0).all();expected+=len(q)
    assert expected==len(d) and nlabels==len(labels)
    h=pd.read_csv(REFERENCE/'predictions.csv');h=h[(h.method=='head')&h.source_dataset.isin(['mit_batch1','mit_batch2','mit_batch3'])]
    combined=pd.concat([h,d],ignore_index=True)
    for _,z in combined.groupby(['dataset','source_dataset','scope']):
        ref=None
        for _,q in z.groupby(['length','seed','method']):
            a=q[['battery_id','origin_cycle','true_rul']].sort_values(['battery_id','origin_cycle']).reset_index(drop=True)
            if ref is None:ref=a
            else:pd.testing.assert_frame_equal(ref,a)
    combined.to_csv(out/'head_full_lora_predictions.csv',index=False)
    rows=[];keys=['dataset','source_dataset','scope','length','seed','method','battery_id']
    for key,z in combined[combined.event_observed].groupby(keys):
        e=z.predicted_rul-z.true_rul;den=((z.true_rul-z.true_rul.mean())**2).sum()
        rows.append(dict(zip(keys,key),mae=float(abs(e).mean()),rmse=float(np.sqrt((e**2).mean())),r2=float(1-(e**2).sum()/den) if den else np.nan))
    cm=pd.DataFrame(rows);cm.to_csv(out/'cell_metrics.csv',index=False)
    sm=cm.groupby(keys[:-1])[['mae','rmse','r2']].mean().reset_index();sm.to_csv(out/'seed_metrics.csv',index=False)
    summary=sm.groupby(['dataset','source_dataset','scope','length','method'])[['mae','rmse','r2']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index();summary.to_csv(out/'summary.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,g in zip(axes,['mit_batch1','mit_batch2','mit_batch3']):
        z=summary[(summary.dataset==g)&(summary.scope=='within_dataset')]
        for method,color in [('head','#9543ba'),('full','#df831b'),('lora','#009f73')]:
            q=z[z.method==method];ax.errorbar(q.length,q.mae_mean,yerr=q.mae_std,label=method,color=color,marker='o',capsize=3)
        ax.set(title=g,xlabel='Context length',ylabel='Cell-macro MAE (cycles)');ax.set_xticks([8,16,32,64,128]);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('MIT nominal80 direct RUL | 3 seeds | same cells, labels and evaluation origins\nHead: CUDA reference; Full/LoRA: laptop XPU, FP32')
    fig.tight_layout();fig.savefig(out/'method_context_mae.png',dpi=160);plt.close(fig)
    within=summary[summary.scope=='within_dataset'][['dataset','length','method','mae_mean','mae_std','r2_mean']]
    html='<!doctype html><html lang="ko"><meta charset="utf-8"><title>MIT RUL Full · LoRA XPU 결과</title><style>body{font:16px Arial;max-width:1300px;margin:30px auto}img{width:100%}td,th{padding:8px;text-align:right}table{border-collapse:collapse;width:100%}</style><h1>MIT 정격80% RUL — Head·Full·LoRA</h1><p>동일한 학습 2셀 × 정답16개, 별도 검증1셀, 입력8·16·32·64·128, seed42·43·44. 모든 모델 선택 후 시험 평가. t≥128이며 초기수명 성능이 아닙니다. Head는 기존 CUDA 결과, Full·LoRA는 노트북 XPU 결과입니다.</p><img src="method_context_mae.png"><p><a href="summary.csv">전체 성능표(전이 포함)</a> · <a href="head_full_lora_predictions.csv">모든 예측값</a> · <a href="verification.json">검증 기록</a></p>'+within.to_html(index=False,float_format=lambda v:f'{v:.3f}')+'</html>'
    (out/'index.html').write_text(html,encoding='utf-8')
    write(out/'verification.json',dict(status='passed',verified_at=now(),models=90,cell_disjoint=True,matched_head_origins=True,
        frozen_scope_verified=True,nonzero_backbone_gradients=True,checkpoint_reload_verified=True,source_data_hashes=True,
        raw_prediction_rows=len(d),combined_prediction_rows=len(combined)))
    paths=set(p['source_sha256'])|{LABEL_FILE}
    with ZipFile(out/'source_snapshot.zip','w',ZIP_DEFLATED) as z:
        for rel in paths:z.write(ROOT/rel,rel)
    files={x.relative_to(out).as_posix():sha256(x) for x in out.rglob('*') if x.is_file() and '.runtime' not in x.parts and x.name not in ['artifact_manifest.json','complete.json','status.json']}
    write(out/'artifact_manifest.json',files)
    write(out/'complete.json',dict(status='complete',finished_at=now(),models=90,verification_sha256=sha256(out/'verification.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();report(a.out)
