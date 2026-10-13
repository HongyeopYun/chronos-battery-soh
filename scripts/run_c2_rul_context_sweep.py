"""Matched NASA t=64, H256 RUL diagnostic sweep: L16/32/64 on desktop CUDA."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts import run_c2_rul_head as engine
from scripts.run_c2_rul_pilot import sha,write_json,now,QUANTILES,REVISION
from scripts.run_c2_context2 import runtime_info
from src.c2_rul import make_tasks,score_trajectory
from src.c2_rul_head import source_trajectories,cache_features,validate_cache_equivalence
from src.c2_one_step import restore_head
from src.c2_partial_tuning import configure_subset
from src.soh_data import load_bundle

OUT=ROOT/'results/c2-battery/rul-nasa-context-sweep-cuda-v1'
BUNDLE=ROOT/'data/processed/reviewed_native_v3/nasa_aging'
PRIOR=ROOT/'results/c2-battery/rul-nasa-l64-h256-head-v1'
LENGTHS=(16,32,64)
SOURCES=sorted(set(engine.SOURCES)|{'scripts/run_c2_rul_context_sweep.py',
    'scripts/report_c2_rul_context_sweep.py','scripts/run_c2_context2.py',
    'src/c2_context_comparison.py','scripts/run_c2_one_step_context.py'})

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def matched_inputs(data,metadata,length):
    if length not in LENGTHS:raise ValueError('Unsupported context')
    audit,master=make_tasks(data,metadata,context_length=64)
    tasks=[]
    for task in master:
        if task['origin_cycle']!=64:continue
        t=dict(task);t['context']=task['context'][-length:].copy()
        t['context_start']=65-length
        t['context_observed']=int(np.isfinite(t['context']).sum())
        assert t['context_observed']>0
        tasks.append(t)
    source=source_trajectories(data,context_length=64,horizon=256)
    for record in source.values():record['context']=record['context'][-length:].copy()
    return audit,tasks,source

def prepare():
    assert read(PRIOR/'status.json')['status']=='complete'
    old=read(PRIOR/'protocol.json')
    for name,digest in old['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in old['data_sha256'].items():assert sha(BUNDLE/name)==digest,name
    data,meta,_=load_bundle(BUNDLE)
    plan=dict(schema_version=1,contexts=list(LENGTHS),primary_origin=64,prediction_length=256,
        model_id='amazon/chronos-2',revision=REVISION,seed=42,learning_rates=[1e-6,1e-5],
        steps_per_lr=100,validation_every=20,folds=old['folds'],
        data_sha256=old['data_sha256'],source_sha256={s:sha(ROOT/s) for s in SOURCES},
        prior_protocol_sha256=sha(PRIOR/'protocol.json'),
        prior_forecast_sha256=sha(PRIOR/'forecast_quantiles.npz'),
        design='All source, validation, and test origins fixed at t=64. Crop only context; identical future labels and folds across lengths. Fresh pretrained head for each length/fold/LR. Lock all 12 heads before test inference.',
        selection=old['selection'].replace('four heads','12 heads'),loss=old['loss'],
        eol_definition=old['eol_definition'],censoring=old['censoring'],
        diagnostics='Same H256 trajectories scored in prefixes 16/32/64/128/256 and disjoint bands 1-16/17-32/33-64/65-128/129-256, eligible observations only through EOL or censoring. Missing future labels remain unscored. Pointwise quantiles are not RUL intervals.',
        limitations='Exploratory NASA four-cell study, three observed EOLs, single seed. No optimal input selected using test RUL. No smoothing, monotonic projection, extra extrapolation, or future measured input.',
        hardware='All three contexts freshly inferred and fitted on identical desktop CUDA; old XPU L64 is a diagnostic reference only.')
    prepared={};labels=[];reference=None
    for length in LENGTHS:
        audit,tasks,source=matched_inputs(data,meta,length)
        assert len(tasks)==4 and all(t['origin_cycle']==64 for t in tasks)
        ids=sorted(source)
        if reference is None:reference=source
        for battery in ids:np.testing.assert_array_equal(source[battery]['target'],reference[battery]['target'])
        for fold in plan['folds']:
            engine.check_physical_split(fold,{b:b for b in ids})
            for split,key in [('train','train_ids'),('validation','validation_ids')]:
                for b in fold[key]:
                    cycles=(np.flatnonzero(np.isfinite(source[b]['target']))+65).tolist()
                    labels.append(dict(context_length=length,fold_id=fold['fold_id'],split=split,battery_id=b,
                        origin_cycle=64,context_start=65-length,context_end=64,label_cycles=cycles,n_labels=len(cycles)))
        prepared[length]=(tasks,source)
    old_labels={(x['fold_id'],x['split'],x['battery_id']):x['label_cycles'] for x in old['label_budget']}
    for row in labels:assert row['label_cycles']==old_labels[(row['fold_id'],row['split'],row['battery_id'])]
    plan['label_budget']=labels
    plan['tasks']={str(l):[{k:v for k,v in t.items() if k!='context'} for t in ts] for l,(ts,_) in prepared.items()}
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==plan,'Locked protocol changed'
    write_json(OUT/'protocol.json',plan)
    audit.to_csv(OUT/'eol_audit.csv',index=False)
    pd.DataFrame(labels).to_csv(OUT/'label_budget.csv',index=False)
    print('PREPARED L16/32/64; fixed t64; H256; 12 heads; identical source labels',flush=True)
    return plan,data,audit,prepared

def predict(pipeline,contexts,length):
    pipeline.model.eval()
    with engine.torch.inference_mode():
        values,_=pipeline.predict_quantiles(inputs=[np.asarray(x,dtype=np.float32) for x in contexts],
            prediction_length=256,context_length=length,quantile_levels=QUANTILES,batch_size=4,cross_learning=False)
    q=np.stack([v.cpu().numpy()[0] for v in values])
    assert q.shape==(len(contexts),256,9) and np.isfinite(q).all()
    return q

def verify(plan,prepared,check_weights=True):
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['data_sha256'].items():assert sha(BUNDLE/name)==digest,name
    assert sha(PRIOR/'protocol.json')==plan['prior_protocol_sha256']
    assert sha(PRIOR/'forecast_quantiles.npz')==plan['prior_forecast_sha256']
    rows=pd.read_csv(OUT/'rul_predictions.csv')
    locked=read(OUT/'heads_locked.json');assert len(locked['selected'])==12
    with np.load(OUT/'forecast_quantiles.npz') as saved:
        for length,(tasks,source) in prepared.items():
            np.testing.assert_array_equal(saved[f'L{length}_context'],np.stack([t['context'] for t in tasks]))
            for method,key in [('zero-shot','zero'),('head-only','head')]:
                q=saved[f'L{length}_{key}'];assert q.shape==(4,256,9) and np.isfinite(q).all()
                for task,value in zip(tasks,q):
                    score=score_trajectory(task,value[:,4]);r=rows[rows.context_length.eq(length)&rows.method.eq(method)&rows.battery_id.eq(task['battery_id'])].iloc[0]
                    for k in ['predicted_rul','error_cycles','absolute_error_lower_bound_cycles']:
                        np.testing.assert_allclose(r[k],np.nan if score[k] is None else score[k],rtol=1e-10,atol=1e-12,equal_nan=True)
                    assert bool(r.contradicts_censored_followup)==score['contradicts_censored_followup']
            for fold in plan['folds']:
                directory=OUT/f'L{length}'/fold['fold_id'];s=read(directory/'selected.json')
                assert s['frozen_backbone_verified'] and s['selected_head_reload_verified'] and s['masked_loss_native_parity']
                history=pd.read_csv(directory/'validation_history.csv')
                best=history[history.eligible].sort_values(['validation_loss','step','learning_rate']).iloc[0]
                assert s['step']==int(best.step) and s['learning_rate']==best.learning_rate
                if check_weights:assert sha(ROOT/s['weights_file'])==s['weights_sha256']
                if s['step']==0:
                    i=[t['battery_id'] for t in tasks].index(fold['test_ids'][0])
                    np.testing.assert_allclose(saved[f'L{length}_head'][i],saved[f'L{length}_zero'][i],rtol=2e-5,atol=2e-6)
    assert len(rows)==24
    return dict(status='passed',verified_at=now(),lengths=list(LENGTHS),test_origins_per_condition=4,
        conditions=6,heads=12,identical_source_labels=True,all_origins_t64=True,
        raw_rul_metrics_recomputed=True,checkpoint_selection_recomputed=True,
        selected_head_reload_verified=True,checkpoint_files_checked=check_weights,
        source_data_hashes_verified=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--model-cache',type=Path,default=ROOT/'models/huggingface');args=parser.parse_args()
    if (OUT/'status.json').exists() and read(OUT/'status.json')['status']=='complete':raise ValueError('Completed run immutable')
    plan,data,audit,prepared=prepare()
    if args.prepare_only:return
    started=now();state=dict(status='loading',started_at=started,protocol_sha256=sha(OUT/'protocol.json'))
    write_json(OUT/'status.json',state)
    try:
        runtime=runtime_info('cuda:0')
        assert runtime==read(ROOT/'results/c2-battery/one-step-2-reviewed-v3/runtime.json'),'Desktop runtime changed'
        write_json(OUT/'runtime.json',runtime)
        torch=engine.torch;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(42);np.random.seed(42)
        pipeline=engine.Chronos2Pipeline.from_pretrained(plan['model_id'],revision=REVISION,device_map='cuda:0',dtype=torch.float32,cache_dir=str(args.model_cache.resolve()),local_files_only=True)
        model=pipeline.model;assert model.device.type=='cuda';model.eval()
        names=configure_subset(model,'head',training=False)
        initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names}
        frozen={n:p._version for n,p in model.named_parameters() if n not in names}
        selections={};parity={}
        for length,(_,source) in prepared.items():
            restore_head(model,initial);ids=sorted(source)
            cache=cache_features(model,np.stack([source[b]['context'] for b in ids]),np.stack([source[b]['target'] for b in ids]))
            parity[str(length)]=validate_cache_equivalence(model,cache)
            engine.OUT=OUT/f'L{length}';engine.OUT.mkdir(exist_ok=True)
            engine.WEIGHTS=ROOT/f'models/c2-rul-context-sweep-cuda-v1/L{length}'
            selections[length]=[]
            for fold in plan['folds']:
                state.update(status='training',context_length=length,fold=fold['fold_id']);write_json(OUT/'status.json',state)
                selections[length].append(engine.train_fold(model,initial,cache,ids,fold,plan))
            del cache
        locked=now();write_json(OUT/'heads_locked.json',dict(locked_at=locked,selected=[dict(context_length=l,**s) for l,ss in selections.items() for s in ss]))
        write_json(OUT/'cache_parity.json',parity)
        state.update(status='test_inference',heads_locked_at=locked,completed_heads=12);write_json(OUT/'status.json',state)
        arrays={}
        for length,(tasks,_) in prepared.items():
            x=np.stack([t['context'] for t in tasks]);restore_head(model,initial)
            zero=predict(pipeline,x,length);head=np.empty_like(zero)
            for fold,s in zip(plan['folds'],selections[length]):
                restore_head(model,torch.load(ROOT/s['weights_file'],map_location='cpu',weights_only=True))
                i=[t['battery_id'] for t in tasks].index(fold['test_ids'][0])
                head[i]=predict(pipeline,x[i:i+1],length)[0]
            arrays.update({f'L{length}_context':x,f'L{length}_zero':zero,f'L{length}_head':head})
            print('TEST COMPLETE',length,flush=True)
        np.savez_compressed(OUT/'forecast_quantiles.npz',**arrays)
        from scripts.report_c2_rul_context_sweep import build_report
        build_report(OUT,plan,data,prepared,arrays)
        restore_head(model,initial);params=dict(model.named_parameters())
        assert all(params[n]._version==v and params[n].grad is None for n,v in frozen.items())
        verification=verify(plan,prepared)
        verification['artifacts_sha256']={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['verification.json','status.json']}
        write_json(OUT/'verification.json',verification)
        state.update(status='complete',finished_at=now());write_json(OUT/'status.json',state)
        print('COMPLETE',OUT,flush=True)
    except Exception as e:
        state.update(status='failed',error=repr(e));write_json(OUT/'status.json',state);raise

if __name__=='__main__':main()
