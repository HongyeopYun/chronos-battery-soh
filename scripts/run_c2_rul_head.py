"""Four held-out NASA folds: native C2 output-head adaptation, L64 H256."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from src.soh_data import load_bundle
from src.c2_rul import make_tasks
from src.c2_rul_head import source_trajectories,cache_features,subset_cache,cached_forward,validation_scores,validate_cache_equivalence
from src.c2_one_step import restore_head
from src.c2_partial_tuning import configure_subset,choose_checkpoint,eligible_checkpoint
from src.c2_head_protocol import check_physical_split
from scripts.run_c2_rul_pilot import sha,write_json,now,REVISION,QUANTILES

PRIOR=ROOT/'results/c2-battery/rul-nasa-l64-h256-zero-v1'
OUT=ROOT/'results/c2-battery/rul-nasa-l64-h256-head-v1'
BUNDLE=ROOT/'data/processed/reviewed_native_v3/nasa_aging'
WEIGHTS=ROOT/'models/c2-rul-nasa-l64-h256-head-v1'
FOLDS=ROOT/'results/c2-battery/one-step-64-reviewed-v3/protocol.json'
SOURCES=['scripts/run_c2_rul_head.py','scripts/report_c2_rul_head.py','src/c2_rul_head.py',
         'scripts/run_c2_rul_pilot.py','src/c2_rul.py','src/c2_one_step.py','src/c2_partial_tuning.py',
         'src/c2_head_protocol.py','src/c2_fewshot_trainer.py','src/battery_fewshot_tuning.py','src/soh_data.py']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def check_reference():
    assert read(PRIOR/'status.json')['status']=='complete'
    reference=read(PRIOR/'protocol.json')
    for name,digest in reference['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in reference['data_sha256'].items():assert sha(BUNDLE/name)==digest,name
    for name,digest in read(PRIOR/'verification.json')['artifacts_sha256'].items():assert sha(PRIOR/name)==digest,name
    return reference


def prepare():
    prior=check_reference()
    data,meta,_=load_bundle(BUNDLE)
    audit,tasks=make_tasks(data,meta)
    serial=[{k:v for k,v in task.items() if k!='context'} for task in tasks]
    assert serial==prior['tasks']
    source=source_trajectories(data)
    source_ids=sorted(source)
    folds=[{k:f[k] for k in ['fold_id','train_ids','validation_ids','test_ids']}
           for f in read(FOLDS)['groups']['nasa_aging']['folds']]
    labels=[]
    for fold in folds:
        check_physical_split(fold,{b:b for b in source_ids})
        for role,key in [('train','train_ids'),('validation','validation_ids')]:
            for battery in fold[key]:
                target=source[battery]['target']
                cycles=(np.flatnonzero(np.isfinite(target))+65).tolist()
                labels.append(dict(fold_id=fold['fold_id'],split=role,battery_id=battery,origin_cycle=64,
                                   context_start=1,context_end=64,trajectory_count=1,
                                   unique_soh_labels=len(cycles),first_label_cycle=cycles[0],last_label_cycle=cycles[-1],
                                   masked_future_positions=256-len(cycles),label_cycles=cycles))
    plan=dict(schema_version=1,scope='NASA four-fold matched zero-shot versus two-trajectory head-only pilot',
              model_id='amazon/chronos-2',revision=REVISION,context_length=64,prediction_length=256,
              primary_origin=64,folds=folds,reference_protocol_sha256=sha(PRIOR/'protocol.json'),
              reference_forecast_sha256=sha(PRIOR/'forecast_quantiles.npz'),reference_fold_protocol_sha256=sha(FOLDS),
              seed=42,learning_rates=[1e-6,1e-5],steps_per_lr=100,validation_every=20,
              head='output_patch_embedding only; original pretrained initialization separately for each fold and LR',
              few_shot='Exactly two source battery trajectories (one t=64 origin each), one disjoint validation trajectory. 172 or 208 unique training SOH labels, not two point labels. Context is also observed information.',
              labels='All eligible actually observed source SOH after t=64, including recorded post-EOL capacity; no fictitious future targets. Remaining 256-step horizon is NaN-masked. No RUL labels or RUL loss used for fitting or selection.',
              loss='Native 21-quantile pinball loss in context-normalized/arcsinh space, mean over finite targets per cell, sum over quantiles, mean over source cells. Verified against native forward loss scaled per cell by H/observed_count.',
              optimizer='Full batch of the two source trajectories; AdamW eps=1e-8, weight_decay=0, gradient clipping 1, linear LR decay; frozen encoder eval, output head train',
              cache='Cache frozen context-only encoder features for efficiency. No future SOH is supplied to encode. Verify cached outputs and loss against native forward before/after fitting.',
              selection='Validation SOH loss only; step zero allowed. Reject candidates whose raw decile crossing or q10>q90 fraction over all 256 validation forecast positions exceeds step zero. Lock all four heads before test inference. No test-driven hyperparameter search.',
              evaluation=prior['scoring'],censoring=prior['censoring'],eol_definition=prior['eol_definition'],
              future_conditions=prior['future_conditions'],uncertainty=prior['uncertainty'],
              limitations='Single seed, three cells with observed EOL, repeated overlapping test origins. Beyond 68-104 future steps there are no source labels; H256 is not fully supervised. Previous test results motivated this exploratory study, so it is not an untouched external benchmark.',
              data_sha256=prior['data_sha256'],source_sha256={p:sha(ROOT/p) for p in SOURCES},
              tasks=serial,label_budget=labels)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists() and read(OUT/'protocol.json')!=plan:
        raise ValueError('Frozen protocol changed; preserve outputs and use a new run directory')
    write_json(OUT/'protocol.json',plan)
    pd.DataFrame([{k:v for k,v in row.items() if k!='label_cycles'} for row in labels]).to_csv(OUT/'label_budget.csv',index=False)
    audit.to_csv(OUT/'eol_audit.csv',index=False)
    with np.load(PRIOR/'inputs.npz') as archive:
        inputs=np.stack([task['context'] for task in tasks])
        np.testing.assert_array_equal(inputs,archive['context'])
    print('PREPARED four disjoint folds, 17 identical test origins, 2 training trajectories per fold',flush=True)
    return plan,data,audit,tasks,source,inputs


def train_fold(model,initial,all_cache,source_ids,fold,plan):
    folder=OUT/fold['fold_id'];folder.mkdir(exist_ok=True)
    selected_path=folder/'selected.json'
    if selected_path.exists():
        selected=read(selected_path)
        assert sha(ROOT/selected['weights_file'])==selected['weights_sha256']
        return selected
    train=subset_cache(all_cache,[source_ids.index(b) for b in fold['train_ids']])
    val=subset_cache(all_cache,[source_ids.index(b) for b in fold['validation_ids']])
    restore_head(model,initial)
    names=configure_subset(model,'head')
    params=dict(model.named_parameters())
    frozen={n:p._version for n,p in params.items() if n not in names}
    baseline=dict(step=0,learning_rate=0.,eligible=True,**validation_scores(model,val))
    history=[baseline];winner=baseline;winner_weights={n:v.clone() for n,v in initial.items()}
    started=time.perf_counter()
    for lr in plan['learning_rates']:
        restore_head(model,initial);torch.manual_seed(plan['seed']);np.random.seed(plan['seed'])
        configure_subset(model,'head')
        optimizer=torch.optim.AdamW([params[n] for n in names],lr=lr,weight_decay=0.,eps=1e-8)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:max(0.,1-step/plan['steps_per_lr']))
        for step in range(1,plan['steps_per_lr']+1):
            optimizer.zero_grad(set_to_none=True)
            loss,_=cached_forward(model,train)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([params[n] for n in names],1.)
            optimizer.step();scheduler.step()
            if step%plan['validation_every']==0:
                candidate=dict(step=step,learning_rate=lr,**validation_scores(model,val))
                candidate['eligible']=eligible_checkpoint(candidate,baseline);history.append(candidate)
                if choose_checkpoint(history,baseline) is candidate:
                    winner=candidate;winner_weights={n:params[n].detach().cpu().clone() for n in names}
                pd.DataFrame(history).to_csv(folder/'validation_history.csv',index=False)
                print(f"{fold['fold_id']} lr={lr:g} step={step} val={candidate['validation_loss']:.6f} eligible={candidate['eligible']}",flush=True)
                configure_subset(model,'head')
        del optimizer,scheduler
    assert winner==choose_checkpoint(history,baseline)
    assert all(params[n]._version==v and params[n].grad is None for n,v in frozen.items())
    WEIGHTS.mkdir(parents=True,exist_ok=True)
    weights=WEIGHTS/(fold['fold_id']+'.pt');torch.save(winner_weights,weights)
    restore_head(model,torch.load(weights,map_location='cpu',weights_only=True))
    reloaded=validation_scores(model,val)
    for key,value in reloaded.items():np.testing.assert_allclose(value,winner[key],rtol=1e-6,atol=1e-7)
    parity=validate_cache_equivalence(model,val)
    result=dict(**winner,baseline=baseline,fold_id=fold['fold_id'],train_ids=fold['train_ids'],
                validation_ids=fold['validation_ids'],test_ids=fold['test_ids'],selected_at=now(),
                trainable_parameters=sum(params[n].numel() for n in names),
                total_parameters=sum(p.numel() for p in params.values()),trainable_names=names,
                frozen_backbone_verified=True,selected_head_reload_verified=True,**parity,
                weights_file=str(weights.relative_to(ROOT)).replace('\\','/'),weights_sha256=sha(weights),
                elapsed_seconds=time.perf_counter()-started)
    write_json(selected_path,result)
    restore_head(model,initial)
    print('SELECTED',fold['fold_id'],'step',winner['step'],'lr',winner['learning_rate'],flush=True)
    return result


def predict(pipeline,contexts):
    pipeline.model.eval();values=[]
    for start in range(0,len(contexts),4):
        with torch.inference_mode():
            q,_=pipeline.predict_quantiles(inputs=[np.asarray(x,dtype=np.float32) for x in contexts[start:start+4]],
                prediction_length=256,context_length=64,quantile_levels=QUANTILES,batch_size=4,cross_learning=False)
        values.extend(v.cpu().numpy()[0] for v in q)
    return np.stack(values)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--device',choices=['xpu','cuda','cpu'],default='xpu');args=parser.parse_args()
    plan,data,audit,tasks,source,inputs=prepare()
    if args.prepare_only:return
    if (OUT/'status.json').exists() and read(OUT/'status.json')['status']=='complete':
        raise ValueError('Completed run must not be overwritten')
    started=now();write_json(OUT/'status.json',dict(status='loading',started_at=started,protocol_sha256=sha(OUT/'protocol.json')))
    torch.manual_seed(42);np.random.seed(42)
    pipeline=Chronos2Pipeline.from_pretrained('amazon/chronos-2',revision=REVISION,device_map=args.device,
        dtype=torch.float32,cache_dir=str(ROOT/'models/huggingface'),local_files_only=True)
    model=pipeline.model;model.eval();names=configure_subset(model,'head',training=False)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names}
    frozen={n:p._version for n,p in model.named_parameters() if n not in names}
    source_ids=sorted(source)
    cache=cache_features(model,np.stack([source[b]['context'] for b in source_ids]),np.stack([source[b]['target'] for b in source_ids]))
    cache_parity=validate_cache_equivalence(model,cache)
    runtime=dict(started_at=started,host=platform.node(),python=platform.python_version(),device=args.device,
                 device_name=torch.xpu.get_device_name(0) if args.device=='xpu' else torch.cuda.get_device_name(0) if args.device=='cuda' else platform.processor(),
                 packages={p:version(p) for p in ['torch','chronos-forecasting','transformers','numpy','pandas']},
                 native_prediction_length=pipeline.model_prediction_length,cache_parity=cache_parity)
    write_json(OUT/'runtime.json',runtime)
    print('MODEL AND CACHED-FORWARD PARITY VERIFIED',flush=True)
    selected=[]
    for i,fold in enumerate(plan['folds']):
        write_json(OUT/'status.json',dict(status='training',fold=fold['fold_id'],completed_heads=i,total_heads=4,started_at=started))
        selected.append(train_fold(model,initial,cache,source_ids,fold,plan))
    locked=now();write_json(OUT/'heads_locked.json',dict(locked_at=locked,selected=[dict(fold_id=r['fold_id'],weights_sha256=r['weights_sha256'],step=r['step'],learning_rate=r['learning_rate']) for r in selected]))
    pd.DataFrame([{k:v for k,v in r.items() if isinstance(v,(str,int,float,bool))} for r in selected]).to_csv(OUT/'selected_checkpoints.csv',index=False)
    write_json(OUT/'status.json',dict(status='test_inference',heads_locked_at=locked,started_at=started))
    restore_head(model,initial)
    zero=predict(pipeline,inputs)
    with np.load(PRIOR/'forecast_quantiles.npz') as saved:reference=saved['quantiles']
    np.testing.assert_allclose(zero,reference,rtol=2e-5,atol=2e-6)
    # Preserve the published reference values exactly; the fresh inference is a reproducibility check.
    head=np.empty_like(reference)
    for fold,selection in zip(plan['folds'],selected):
        weights=ROOT/selection['weights_file'];assert sha(weights)==selection['weights_sha256']
        restore_head(model,torch.load(weights,map_location='cpu',weights_only=True))
        indices=[i for i,t in enumerate(tasks) if t['battery_id'] in fold['test_ids']]
        forecast=predict(pipeline,inputs[indices])
        if selection['step']==0:np.testing.assert_allclose(forecast,reference[indices],rtol=2e-5,atol=2e-6)
        head[indices]=forecast
    assert np.isfinite(head).all()
    np.savez_compressed(OUT/'forecast_quantiles.npz',zero_shot=reference,head_only=head,context=inputs)
    from scripts.report_c2_rul_head import build_report
    build_report(OUT,plan,data,audit,tasks,reference,head,selected)
    restore_head(model,initial)
    params=dict(model.named_parameters())
    assert all(params[n]._version==v and params[n].grad is None for n,v in frozen.items())
    check_reference()
    for path,digest in plan['source_sha256'].items():assert sha(ROOT/path)==digest,path
    runtime.update(finished_at=now(),heads_locked_at=locked,reference_reproduction_max_abs=float(abs(zero-reference).max()))
    write_json(OUT/'runtime.json',runtime)
    verify_outputs(plan,tasks,reference,head,selected)
    write_json(OUT/'status.json',dict(status='complete',started_at=started,finished_at=now(),protocol_sha256=sha(OUT/'protocol.json')))
    print('COMPLETE',str(OUT),flush=True)


def verify_outputs(plan,tasks,zero,head,selected):
    scores=pd.read_csv(OUT/'rul_predictions.csv')
    for method,values in [('zero-shot',zero),('head-only',head)]:
        for task,value in zip(tasks,values):
            row=scores[scores.method.eq(method)&scores.battery_id.eq(task['battery_id'])&scores.origin_cycle.eq(task['origin_cycle'])].iloc[0]
            hit=np.flatnonzero(value[:,4]<=task['threshold_soh'])
            if len(hit):assert row.predicted_rul==int(hit[0])+1
            else:assert pd.isna(row.predicted_rul)
    for selection in selected:
        history=pd.read_csv(OUT/selection['fold_id']/'validation_history.csv')
        best=history[history.eligible].sort_values(['validation_loss','step','learning_rate']).iloc[0]
        assert int(best.step)==selection['step'] and best.learning_rate==selection['learning_rate']
    artifacts={str(p.relative_to(OUT)).replace('\\','/'):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['verification.json','status.json']}
    write_json(OUT/'verification.json',dict(status='passed',verified_at=now(),test_origins=17,methods=2,
        original_zero_artifacts_unchanged=True,identical_inputs_targets_eol_and_origins=True,
        all_four_heads_locked_before_test=True,all_saved_weights_reload_verified=True,
        frozen_backbone_verified=True,masked_loss_native_parity=True,
        saved_rul_hits_recomputed=True,validation_only_selection_recomputed=True,artifacts_sha256=artifacts))


if __name__=='__main__':main()
