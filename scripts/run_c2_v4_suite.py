"""Frozen-protocol CUDA suite: rolling SOH, trajectory RUL, direct RUL adaptation."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from src.c2_v4_experiments import (LENGTHS, METHODS, DirectRUL, build_design, load_cells,
    rul_windows, soh_windows, source_arrays)
from src.c2_one_step import point_metrics, restore_head
from src.c2_rul import score_trajectory, summarize_rul
from src.battery_zeroshot import sha256

REVISION='29ec3766d36d6f73f0696f85560a422f50e8498c'
HORIZON=1024


def now(): return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def status(out, phase, **kw):
    write(out/'status.json',dict(status=phase,updated_at=now(),**kw))
    print(now(),phase,json.dumps(kw,ensure_ascii=True),flush=True)


def load_model(cache_root):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required; no CPU fallback')
    errors=[]
    candidates=[cache_root,ROOT/'models/huggingface',ROOT.parents[1]/'models/huggingface',
                ROOT.parents[1]/'c2-context16-desktop/models/huggingface',None]
    for cache in dict.fromkeys(str(p) if p else None for p in candidates):
        try:
            pipe=Chronos2Pipeline.from_pretrained('amazon/chronos-2',revision=REVISION,
                device_map='cuda',dtype=torch.float32,cache_dir=cache,local_files_only=True)
            pipe.model.eval()
            return pipe
        except OSError as e: errors.append(str(e).splitlines()[0])
    raise RuntimeError('Pinned model not in local caches: '+repr(errors))


def snapshot_sources():
    paths=[Path(__file__),*(ROOT/'src').glob('*.py')]
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in paths}


def prepare(out):
    cells, fingerprints=load_cells(ROOT/'data/processed/reviewed_v4/forecast_safe')
    design,budget,coverage=build_design(cells)
    protocol=dict(schema_version=1,revision=REVISION,seed=42,device='cuda',dtype='float32',
        data_profile='reviewed_v4/forecast_safe',data_artifact_sha256=fingerprints,source_sha256=snapshot_sources(),
        soh_lengths=list(LENGTHS),soh_prediction_length=1,soh_methods=['LastValue','zero-shot','head-only'],
        soh_matched_targets='Original t>=65, eligible target, at least one valid observation in t-2:t-1; identical times for all L.',
        rul_context=32,rul_horizon=HORIZON,rul_origin_stride=16,rul_threshold=.8,
        rul_definition='First eligible observed Q/Q_ref <= .8; absorbing, no smoothing/interpolation. Record end is censoring, never EOL.',
        rul_direct_methods=list(METHODS),rul_features='Native REG token plus input loc, scale and last eligible SOH. No cycle, dataset ID, Qref, future covariates or EOL input.',
        rul_head='LayerNorm(771), Linear(771,64), GELU, Linear(64,1), softplus; train-label median scaling; initial constant train-label mean.',
        rul_labels='Four evenly spaced eligible pre-EOL source origins per completed cell; eight train/four validation labels. Source selection uses completed source histories only. Evaluation origins fixed every 16 positions.',
        soh_steps=100,soh_lrs=[1e-6,1e-5],rul_steps=200,rul_lrs={'head':[1e-3,3e-3],'full':[1e-6,1e-5],'lora':[1e-4,3e-4]},
        rul_head_lr_with_backbone=1e-3,batch_size=4,validation_every=20,
        rul_loss='Huber on RUL divided by median training-label RUL; validation MAE, step zero allowed; no refit.',
        lora=dict(rank=8,alpha=16,dropout=0.,target_modules=['self_attention.q','self_attention.k','self_attention.v','self_attention.o']),
        dropout='eval mode in all optimization forwards: no dropout, gradients enabled for selected scope.',
        all_selections_before_test=True,split_design=design,
        limitations=['Single seed and fixed small labeled budgets; not a hyperparameter optimum.',
            'CALCE axis is retained record count; missing physical cycles between files unknown.',
            'Within-dataset and transferred direct models reported separately.',
            'Censored labels never used as exact RUL; only observed-followup consistency evaluated.',
            'Conditioning exact-RUL training on observed EOL introduces subset selection; does not establish censored-population accuracy.'])
    if (out/'protocol.json').exists():
        if read(out/'protocol.json')!=protocol: raise ValueError('Frozen protocol/source changed; choose a new run directory')
    else:
        out.mkdir(parents=True,exist_ok=True); write(out/'protocol.json',protocol)
        budget.to_csv(out/'label_budget.csv',index=False); coverage.to_csv(out/'coverage.csv',index=False)
        pd.DataFrame([dict(dataset=g,battery_id=b,eol_cycle=c['eol'],event_observed=c['eol'] is not None,
            last_eligible_cycle=c['last'],eligible_observations=int(c['keep'].sum())) for (g,b),c in cells.items()]).to_csv(out/'eol_audit.csv',index=False)
    return cells,protocol,budget


def tensor(x): return torch.as_tensor(x,device='cuda',dtype=torch.float32)


@torch.no_grad()
def soh_cache(model,x,y):
    x=tensor(x); y=tensor(y)[:,None]
    encoded,locscale,_,_=model.encode(context=x)
    target,_=model.instance_norm(y,locscale)
    return encoded[0][:,-1:,:].detach(),locscale,target.detach()


def cache_output(model,cache,indices=None):
    h,ls,y=cache
    if indices is not None: h=h[indices]; ls=tuple(a[indices] for a in ls); y=y[indices]
    q=model.output_patch_embedding(h).reshape(len(h),model.num_quantiles,16)
    levels=model.quantiles[None,:,None]
    err=y[:,None,:]-q[:,:,:1]
    loss=(2*torch.abs(err*((err<=0).to(q.dtype)-levels))).sum(dim=1).mean()
    unscaled=model.instance_norm.inverse(q.reshape(len(h),-1),ls).reshape_as(q)
    return loss,unscaled


@torch.no_grad()
def soh_validation(model,cache):
    loss,q=cache_output(model,cache)
    idx=[int(torch.argmin(abs(model.quantiles-v/10))) for v in range(1,10)]
    q=q[:,idx,0]
    return dict(loss=float(loss),crossing=float((q[:,1:]<q[:,:-1]).any(dim=1).float().mean()),
                invalid=float((q[:,0]>q[:,-1]).float().mean()))


def selected_ok(directory,protocol_hash):
    if not (directory/'selected.json').exists(): return False
    s=read(directory/'selected.json')
    if s['protocol_sha256']!=protocol_hash or sha256(directory/'weights.pt')!=s['weights_sha256']:
        raise ValueError('Checkpoint provenance mismatch')
    return True


def train_soh(model,initial,cells,budget,out,protocol):
    ph=sha256(out/'protocol.json')
    names=set(initial); params=dict(model.named_parameters())
    for n,p in params.items(): p.requires_grad_(n in names); p.grad=None
    for group,folds in protocol['split_design']['soh'].items():
        for length in LENGTHS:
            for fold in folds:
                directory=out/'checkpoints/soh'/group/f'L{length}'/fold['fold_id']
                if selected_ok(directory,ph): continue
                status(out,'training_soh',dataset=group,length=length,fold=fold['fold_id'])
                restore_head(model,initial)
                train,val=[source_arrays(cells,budget,'soh',group,fold['fold_id'],s,length) for s in ('train','validation')]
                tc,vc=soh_cache(model,*train),soh_cache(model,*val)
                with torch.no_grad():
                    native=model(context=tensor(val[0]),future_target=tensor(val[1])[:,None])
                    torch.testing.assert_close(cache_output(model,vc)[1],native.quantile_preds,rtol=2e-5,atol=1e-5)
                    torch.testing.assert_close(cache_output(model,vc)[0],native.loss*16,rtol=2e-5,atol=2e-6)
                frozen={n:p._version for n,p in params.items() if n not in names}
                baseline=soh_validation(model,vc)
                winner=dict(step=0,lr=0.,**baseline); weights={n:p.detach().cpu().clone() for n,p in params.items() if n in names}
                history=[winner]
                started=time.monotonic()
                for lr in protocol['soh_lrs']:
                    restore_head(model,initial)
                    opt=torch.optim.AdamW([params[n] for n in sorted(names)],lr=lr,weight_decay=0.)
                    rng=np.random.default_rng(42)
                    for step in range(1,protocol['soh_steps']+1):
                        indices=rng.integers(0,len(train[1]),size=4)
                        opt.zero_grad(set_to_none=True)
                        loss,_=cache_output(model,tc,indices)
                        if not torch.isfinite(loss): raise ValueError('Nonfinite SOH loss')
                        loss.backward(); torch.nn.utils.clip_grad_norm_([params[n] for n in names],1.); opt.step()
                        for pg in opt.param_groups: pg['lr']=lr*(1-step/protocol['soh_steps'])
                        if step%20==0:
                            row=dict(step=step,lr=lr,**soh_validation(model,vc)); history.append(row)
                            if row['loss']<winner['loss'] and row['crossing']<=baseline['crossing'] and row['invalid']<=baseline['invalid']:
                                winner=row; weights={n:params[n].detach().cpu().clone() for n in names}
                    del opt
                assert all(params[n]._version==v and params[n].grad is None for n,v in frozen.items())
                directory.mkdir(parents=True,exist_ok=True); torch.save(weights,directory/'weights.pt')
                restore_head(model,torch.load(directory/'weights.pt',map_location='cpu',weights_only=True))
                check=soh_validation(model,vc)
                if not np.isclose(check['loss'],winner['loss'],rtol=1e-5): raise ValueError('SOH checkpoint reload failed')
                pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
                write(directory/'selected.json',dict(**winner,protocol_sha256=ph,weights_sha256=sha256(directory/'weights.pt'),
                    train_labels=len(train[1]),validation_labels=len(val[1]),trainable_parameters=sum(params[n].numel() for n in names),
                    frozen_verified=True,reload_verified=True,elapsed_seconds=time.monotonic()-started))
    restore_head(model,initial)
    for p in model.parameters(): p.requires_grad_(False); p.grad=None


def make_direct(pristine,mode,ys):
    return DirectRUL(pristine,mode,max(1.,float(np.median(ys))),float(np.mean(ys)))


@torch.no_grad()
def direct_predict(model,x,batch=128):
    model.eval(); result=[]
    for start in range(0,len(x),batch): result.append(model(tensor(x[start:start+batch])).cpu().numpy())
    result=np.concatenate(result) if result else np.empty(0)
    if not np.isfinite(result).all() or (result<0).any(): raise ValueError('Invalid direct RUL output')
    return result


def train_direct(pristine,cells,budget,out,protocol):
    ph=sha256(out/'protocol.json')
    for group,folds in protocol['split_design']['rul'].items():
        for fold in folds:
            train,val=[source_arrays(cells,budget,'rul',group,fold['fold_id'],s,32) for s in ('train','validation')]
            for mode in METHODS:
                directory=out/'checkpoints/rul'/group/fold['fold_id']/mode
                if selected_ok(directory,ph): continue
                status(out,'training_direct_rul',dataset=group,fold=fold['fold_id'],method=mode)
                started=time.monotonic(); history=[]; best=None; winner_state=None; gradient_audit={}
                for lr in protocol['rul_lrs'][mode]:
                    model=make_direct(pristine,mode,train[1]); model.eval()
                    frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
                    params={n:p for n,p in model.named_parameters() if p.requires_grad}
                    head=[p for n,p in params.items() if n.startswith('head.')]
                    backbone=[p for n,p in params.items() if n.startswith('backbone.')]
                    groups=[dict(params=head,lr=lr if mode=='head' else 1e-3)]
                    if backbone: groups.append(dict(params=backbone,lr=lr))
                    opt=torch.optim.AdamW(groups,weight_decay=0.)
                    initial_lrs=[pg['lr'] for pg in opt.param_groups]
                    x,y=tensor(train[0]),tensor(train[1])
                    with torch.no_grad():
                        features=model.features(x).detach() if mode=='head' else None
                        vfeatures=model.features(tensor(val[0])).detach() if mode=='head' else None
                    def validation():
                        with torch.no_grad():
                            pred=model.from_features(vfeatures).cpu().numpy() if vfeatures is not None else direct_predict(model,val[0])
                        return float(np.abs(pred-val[1]).mean())
                    initial=dict(step=0,lr=lr,validation_mae=validation()); history.append(initial)
                    if best is None: best=initial; winner_state=model.trainable_state()
                    rng=np.random.default_rng(42)
                    for step in range(1,protocol['rul_steps']+1):
                        idx=rng.integers(0,len(y),size=4)
                        opt.zero_grad(set_to_none=True)
                        pred=model.from_features(features[idx]) if features is not None else model(x[idx])
                        loss=torch.nn.functional.huber_loss(pred/model.scale,y[idx]/model.scale,delta=1.)
                        if not torch.isfinite(loss): raise ValueError('Nonfinite direct RUL loss')
                        loss.backward()
                        if step==5:
                            gradient_audit[str(lr)]={
                                'head_nonzero_gradient':any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in head),
                                'backbone_nonzero_gradient':any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in backbone),
                                'trainable_parameters':sum(p.numel() for p in params.values()),
                                'trainable_names':list(params)}
                            assert gradient_audit[str(lr)]['head_nonzero_gradient']
                            assert gradient_audit[str(lr)]['backbone_nonzero_gradient']==(mode!='head')
                        torch.nn.utils.clip_grad_norm_(list(params.values()),1.); opt.step()
                        for pg,start_lr in zip(opt.param_groups,initial_lrs): pg['lr']=start_lr*(1-step/protocol['rul_steps'])
                        if step%20==0:
                            row=dict(step=step,lr=lr,validation_mae=validation(),training_loss=float(loss.detach())); history.append(row)
                            if row['validation_mae']<best['validation_mae']:
                                best=row; winner_state=model.trainable_state()
                            print(f'RUL {group} {fold["fold_id"]} {mode} lr={lr} step={step} val={row["validation_mae"]:.3f}',flush=True)
                    parameters=dict(model.named_parameters())
                    assert all(parameters[n]._version==v and parameters[n].grad is None for n,v in frozen.items())
                    del opt,model,params,head,backbone,parameters,groups,x,y,features,vfeatures
                    gc.collect(); torch.cuda.empty_cache()
                directory.mkdir(parents=True,exist_ok=True); torch.save(winner_state,directory/'weights.pt')
                reloaded=make_direct(pristine,mode,train[1])
                reloaded.load_trainable(torch.load(directory/'weights.pt',map_location='cpu',weights_only=True))
                checked=float(np.abs(direct_predict(reloaded,val[0])-val[1]).mean())
                if not np.isclose(checked,best['validation_mae'],rtol=1e-5,atol=1e-4): raise ValueError('RUL reload parity failed')
                pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
                write(directory/'selected.json',dict(**best,method=mode,protocol_sha256=ph,weights_sha256=sha256(directory/'weights.pt'),
                    scale=reloaded.scale,mean_target=float(np.mean(train[1])),train_labels=len(train[1]),validation_labels=len(val[1]),
                    reload_verified=True,frozen_verified=True,gradient_audit=gradient_audit,elapsed_seconds=time.monotonic()-started))
                del reloaded,winner_state; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def native_predictions(model,x,horizon=1,batch=256):
    if horizon>model.chronos_config.max_output_patches*16: raise ValueError('Native forecast horizon exceeded')
    idx=[int(torch.argmin(abs(model.quantiles-q))) for q in (.1,.5,.9)]
    result=[]
    for start in range(0,len(x),batch):
        q=model(context=tensor(x[start:start+batch]),num_output_patches=(horizon+15)//16).quantile_preds
        result.append(q[:,idx,:horizon].cpu().numpy())
    return np.concatenate(result) if result else np.empty((0,3,horizon))


def score_soh(model,initial,cells,out,protocol):
    metrics=[]
    for group,folds in protocol['split_design']['soh'].items():
        for length in LENGTHS:
            for fold in folds:
                checkpoint=out/'checkpoints/soh'/group/f'L{length}'/fold['fold_id']
                selected=read(checkpoint/'selected.json')
                for battery in fold['test_ids']:
                    path=out/'soh_predictions'/group/f'L{length}_{battery}.npz'
                    w=soh_windows(cells[(group,battery)]); x=w['context'][:,-length:]
                    if not path.exists():
                        status(out,'predicting_soh',dataset=group,length=length,battery=battery)
                        restore_head(model,initial); zero=native_predictions(model,x)[:,:,0]
                        if selected['step']==0: head=zero.copy()
                        else:
                            restore_head(model,torch.load(checkpoint/'weights.pt',map_location='cpu',weights_only=True))
                            head=native_predictions(model,x)[:,:,0]
                        last=np.max(np.where(np.isfinite(x),np.arange(length),-1),axis=1)
                        path.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(path,cycle=w['cycle'],target=w['target'],last_value=x[np.arange(len(x)),last],
                            zero=zero,head=head,context_observed=np.isfinite(x).sum(axis=1),protocol_sha256=sha256(out/'protocol.json'))
                    with np.load(path) as z:
                        if str(z['protocol_sha256'])!=sha256(out/'protocol.json'): raise ValueError('Prediction provenance mismatch')
                        np.testing.assert_array_equal(z['cycle'],w['cycle']); np.testing.assert_array_equal(z['target'],w['target'])
                        lv_mse=float(np.mean((z['last_value']-z['target'])**2))
                        for method,pred in [('LastValue',z['last_value']),('zero-shot',z['zero'][:,1]),('head-only',z['head'][:,1])]:
                            mse=float(np.mean((pred-z['target'])**2))
                            metrics.append(dict(dataset=group,battery_id=battery,context_length=length,method=method,n_targets=len(pred),
                                mse=mse,mse_reduction_pct=100*(1-mse/lv_mse) if lv_mse else np.nan,**point_metrics(z['target'],pred)))
    restore_head(model,initial)
    frame=pd.DataFrame(metrics); frame.to_csv(out/'soh_battery_metrics.csv',index=False)
    frame.groupby(['dataset','context_length','method'],as_index=False).agg(n_batteries=('battery_id','size'),n_targets=('n_targets','sum'),
        mae=('mae','mean'),rmse=('rmse','mean'),temporal_r2=('temporal_r2','mean'),mse_reduction_pct=('mse_reduction_pct','mean')).to_csv(out/'soh_summary.csv',index=False)


def rul_task(cell,t):
    event=cell['eol'] is not None
    return dict(battery_id=cell['battery_id'],origin_cycle=int(t),threshold_soh=.8,event_observed=event,
        true_eol_cycle=cell['eol'],true_rul=cell['eol']-int(t) if event else None,
        observed_rul_lower_bound=cell['last']-int(t) if not event else None)


def score_rul(pristine,cells,budget,out,protocol):
    zero_rows=[]; direct_rows=[]
    for (group,battery),cell in cells.items():
        w=rul_windows(cell)
        if not len(w['cycle']): continue
        path=out/'rul_trajectories'/group/f'{battery}.npz'
        if not path.exists():
            status(out,'predicting_zero_shot_rul',dataset=group,battery=battery,origins=len(w['cycle']))
            q=native_predictions(pristine,w['context'],horizon=HORIZON,batch=16)
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,cycle=w['cycle'],quantiles=q,protocol_sha256=sha256(out/'protocol.json'))
        with np.load(path) as z:
            assert str(z['protocol_sha256'])==sha256(out/'protocol.json')
            np.testing.assert_array_equal(w['cycle'],z['cycle'])
            for i,t in enumerate(w['cycle']):
                zero_rows.append(dict(dataset=group,method='trajectory-zero-shot',**score_trajectory(rul_task(cell,t),z['quantiles'][i,1])))
    zero=pd.DataFrame(zero_rows); zero.to_csv(out/'rul_zero_predictions.csv',index=False)
    for group,folds in protocol['split_design']['rul'].items():
        for fold in folds:
            train=source_arrays(cells,budget,'rul',group,fold['fold_id'],'train',32)
            for mode in METHODS:
                path=out/'rul_direct_predictions'/group/f'{fold["fold_id"]}_{mode}.csv'
                if path.exists(): direct_rows.extend(pd.read_csv(path).to_dict('records')); continue
                status(out,'predicting_direct_rul',source_dataset=group,fold=fold['fold_id'],method=mode)
                directory=out/'checkpoints/rul'/group/fold['fold_id']/mode
                model=make_direct(pristine,mode,train[1])
                model.load_trainable(torch.load(directory/'weights.pt',map_location='cpu',weights_only=True))
                targets={group:fold['test_ids'],**fold.get('transfer_test',{})}; rows=[]
                for target_group,batteries in targets.items():
                    for battery in batteries:
                        cell=cells[(target_group,battery)]; w=rul_windows(cell)
                        if not len(w['cycle']): continue
                        pred=direct_predict(model,w['context'])
                        for t,p in zip(w['cycle'],pred):
                            task=rul_task(cell,t)
                            error=float(p-task['true_rul']) if task['event_observed'] else None
                            rows.append(dict(dataset=target_group,source_dataset=group,fold=fold['fold_id'],method=mode,
                                scope='within_dataset' if group==target_group else 'transfer',**task,predicted_rul=float(p),
                                error_cycles=error,absolute_error_cycles=abs(error) if error is not None else None,
                                contradicts_censored_followup=bool(not task['event_observed'] and p<=task['observed_rul_lower_bound'])))
                path.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(path,index=False)
                direct_rows.extend(rows)
                del model; gc.collect(); torch.cuda.empty_cache()
    direct=pd.DataFrame(direct_rows); direct.to_csv(out/'rul_direct_predictions.csv',index=False)
    return zero,direct


def summarize_rul_tables(out,zero,direct):
    rows=[]
    # Match the trajectory comparator to precisely the direct models' held-out
    # cells and fixed origins, excluding their source/validation cells.
    keys=direct[direct.method=='head'][['dataset','battery_id','origin_cycle','scope']]
    if keys.duplicated(['dataset','battery_id','origin_cycle']).any(): raise ValueError('Duplicated test origin')
    matched=zero.merge(keys,on=['dataset','battery_id','origin_cycle'],validate='one_to_one')
    matched.to_csv(out/'rul_zero_matched_predictions.csv',index=False)
    for (g,scope),table in matched.groupby(['dataset','scope']):
        rows.append(dict(dataset=g,scope=scope,method='trajectory-zero-shot',**summarize_rul(table)))
    for (g,scope,method),table in direct.groupby(['dataset','scope','method']):
        observed=table[table.event_observed.eq(True)]; error=observed.error_cycles.to_numpy(float)
        cell_r2=[]
        for _,c in observed.groupby('battery_id'):
            cell_r2.append(point_metrics(c.true_rul,c.predicted_rul)['temporal_r2'])
        rows.append(dict(dataset=g,scope=scope,method=method,n_origins=len(table),n_event_origins=len(observed),
            n_event_batteries=observed.battery_id.nunique(),n_finite_event_predictions=len(observed),n_no_crossing_event_predictions=0,
            conditional_mae_cycles=float(np.abs(error).mean()) if len(error) else None,
            conditional_rmse_cycles=float(np.sqrt((error**2).mean())) if len(error) else None,
            conditional_bias_cycles=float(error.mean()) if len(error) else None,
            mae_lower_bound_all_event_origins=float(np.abs(error).mean()) if len(error) else None,
            censored_origins=int((~table.event_observed).sum()),censored_contradictions=int(table.contradicts_censored_followup.sum()),
            all_event_predictions_finite=True,mean_cell_r2=float(np.nanmean(cell_r2)) if any(np.isfinite(cell_r2)) else None))
    pd.DataFrame(rows).to_csv(out/'rul_summary.csv',index=False)


def smoke(model,out):
    x=np.asarray([[1.,.999,.997,.995]*8,[1.,.99,np.nan,.97]*8],np.float32)
    y=np.asarray([50.,100.],np.float32)
    result={}
    with torch.no_grad():
        cache=soh_cache(model,x,y/100)
        native=model(context=tensor(x),future_target=tensor(y/100)[:,None])
        torch.testing.assert_close(cache_output(model,cache)[1],native.quantile_preds,rtol=2e-5,atol=1e-5)
        torch.testing.assert_close(cache_output(model,cache)[0],native.loss*16,rtol=2e-5,atol=2e-6)
        assert native_predictions(model,x,HORIZON,batch=2).shape==(2,3,HORIZON)
    for mode in METHODS:
        m=make_direct(model,mode,y); params=[p for p in m.parameters() if p.requires_grad]
        opt=torch.optim.Adam(params,lr=.001)
        for _ in range(3):
            opt.zero_grad(set_to_none=True); p=m(tensor(x)); ((p-tensor(y))/m.scale).square().mean().backward(); opt.step()
        grad=[n for n,p in m.named_parameters() if p.grad is not None and p.grad.abs().sum()>0]
        assert any(n.startswith('head.') for n in grad)
        assert any(n.startswith('backbone.') for n in grad)==(mode!='head')
        saved=m.trainable_state(); before=direct_predict(m,x)
        m2=make_direct(model,mode,y);m2.load_trainable(saved)
        np.testing.assert_allclose(before,direct_predict(m2,x),rtol=1e-5,atol=1e-4)
        # Independent group IDs: no sharing between unrelated input cells.
        np.testing.assert_allclose(before[:1],direct_predict(m2,x[:1]),rtol=2e-4,atol=.01)
        result[mode]=dict(parameters=sum(p.numel() for p in params),reload=True,gradient_scope=True)
        del m,m2,opt,params,saved;gc.collect();torch.cuda.empty_cache()
    write(out/'smoke.json',dict(device=torch.cuda.get_device_name(),methods=result,status='passed'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cache-root',type=Path);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--smoke-only',action='store_true');args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists(): raise ValueError('Completed run is immutable')
    cells,protocol,budget=prepare(out)
    if args.prepare_only:
        status(out,'prepared',cells=len(cells),soh_heads=sum(map(len,protocol['split_design']['soh'].values()))*6,
               rul_models=sum(map(len,protocol['split_design']['rul'].values()))*3);return
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_model(args.cache_root).model
    environment=dict(started_at=now(),host=platform.node(),python=sys.version,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),packages={p:importlib.metadata.version(p) for p in ('torch','chronos-forecasting','transformers','peft','numpy','pandas')})
    write(out/'environment.json',environment)
    if args.smoke_only: smoke(model,out);return
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith('output_patch_embedding.')}
    train_soh(model,initial,cells,budget,out,protocol)
    train_direct(model,cells,budget,out,protocol)
    selections={str(p.relative_to(out)):sha256(p) for p in (out/'checkpoints').rglob('selected.json')}
    assert len(selections)==117
    write(out/'selection_lock.json',dict(locked_at=now(),selected_sha256=selections,protocol_sha256=sha256(out/'protocol.json')))
    status(out,'all_models_locked',selected_models=len(selections))
    score_soh(model,initial,cells,out,protocol)
    zero,direct=score_rul(model,cells,budget,out,protocol)
    summarize_rul_tables(out,zero,direct)
    for p,h in selections.items(): assert sha256(out/p)==h
    write(out/'execution_finished.json',dict(status='predictions_finished',finished_at=now(),selection_unchanged=True,
        max_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated(),protocol_sha256=sha256(out/'protocol.json')))
    status(out,'predictions_finished',next='independent verification and reporting')


if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc()
        # The existing protocol/checkpoints remain available for an exact resume.
        sys.exit(1)
