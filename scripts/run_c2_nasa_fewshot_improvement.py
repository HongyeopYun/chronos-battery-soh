"""NASA few-shot improvement; unchanged architecture; validation-only selection."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from src.battery_zeroshot import sha256
from src.c2_rul_context_nominal import load,assemble,LABEL_FILE
from src.c2_rul_learning_curve import cell_weights,constants
from src.c2_v4_experiments import raw_features
from src.c2_rul_improvement import fit_scaler,transform,new_head,output,loss_value,configuration_grid

REFERENCE=ROOT/'results/c2-battery/rul-context-nominal80-v1'
REVISION='29ec3766d36d6f73f0696f85560a422f50e8498c'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def status(out,phase,**kwargs):
    value=dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kwargs)
    write(out/'status.json',value);print(json.dumps(value),flush=True)


def prepare(out):
    cells,hashes,_=load(ROOT);ref=read(REFERENCE/'protocol.json')
    assert hashes==ref['data_artifact_sha256']
    cases=[c for c in ref['cases'] if c['dataset']=='nasa_aging' and c['seed']==42]
    assert len(cases)==12
    paths=[Path(__file__),ROOT/'src/c2_rul_improvement.py',ROOT/'src/c2_rul_context_nominal.py',
        ROOT/'src/c2_rul_learning_curve.py',ROOT/'src/c2_v4_experiments.py']
    protocol=dict(task='NASA few-shot direct RUL improvement without architecture changes',revision=REVISION,
        cases=cases,grid=configuration_grid(),lengths=[8,16,32],seeds=[42,43,44],lrs=[.001,.003],
        steps=1000,batch_size=16,eval_every=50,budget_per_source_cell=16,n_source_cells=2,
        architecture='Unchanged frozen Chronos2 REG(768)+loc/scale/lastSOH(3); LayerNorm771,Linear64,GELU,Linear1,softplus.',
        user_constraint='Ask before changing model architecture/internal structure. No new adapters/layers/inputs. Feature scaling is fitted outside the unchanged head using source training examples only.',
        scope='Few-shot only: 2 source cells x 16 labels. No all-label fitting. NASA prioritized; same initial-reference80 EOL as historical NASA results.',
        selection='For each outer fold, seed42 validation MAE selects configuration/L/LR, including the old model as fallback. Fixed-L32 and flexible-L8/16/32 selections separately. Seeds43/44 reuse chosen config/L/LR, select checkpoint only on that fold validation cell. All selections locked before test prediction.',
        evaluation='Original stride16 origins for direct baseline comparability plus dense stride1 sensitivity on identical origins per method. t>=32 and t<EOL. Report each cell and early/middle/late separately. Test reused historically: exploratory, not pristine external validation.',
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in paths},
        data_artifact_sha256=hashes,reference_protocol_sha256=sha256(REFERENCE/'protocol.json'),
        label_sha256=sha256(ROOT/LABEL_FILE))
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol,'Frozen run changed; new directory required'
    else:write(out/'protocol.json',protocol)
    rows=[]
    for c in cases:
        roles=[set(c[k]) for k in ('train_ids','validation_ids','test_ids')]
        assert not any(roles[i]&roles[j] for i,j in ((0,1),(0,2),(1,2)))
        for split,ids,budget in [('train',c['train_ids'],'16'),('validation',c['validation_ids'],'all')]:
            a=arrays(cells,c,ids,budget)
            rows.extend(dict(fold=c['fold'],length=c['length'],split=split,battery=b,origin=int(t),rul=float(y))
                for b,t,y in zip(a['battery'],a['cycle'],a['target']))
            if split=='train':assert len(a['target'])==32
    pd.DataFrame(rows).to_csv(out/'label_inventory.csv',index=False)
    return cells,protocol


def arrays(cells,c,ids,budget='all',stride=1):
    return assemble(cells,'nasa_aging',ids,budget,stride,length=c['length'],start=32)


class Cache:
    def __init__(self,out,cells,device,cache_root):
        self.device=device;self.path=out/'.runtime/features.pt'
        ph=sha256(out/'protocol.json')
        if self.path.exists():
            self.record=torch.load(self.path,weights_only=False,map_location='cpu')
            assert self.record['protocol_sha256']==ph
            return
        status(out,'extracting_frozen_features',device=device)
        pipe=Chronos2Pipeline.from_pretrained('amazon/chronos-2',revision=REVISION,device_map=device,
            dtype=torch.float32,cache_dir=str(cache_root),local_files_only=True)
        core=pipe.model.eval();[p.requires_grad_(False) for p in core.parameters()]
        records={};diagnostics=[]
        with torch.no_grad():
            for length in [8,16,32]:
                for (g,b),cell in cells.items():
                    if g!='nasa_aging':continue
                    c=dict(length=length)
                    a=arrays(cells,c,[b]);x=torch.as_tensor(a['context'],device=device)
                    parts=[]
                    for start in range(0,len(x),64):
                        z=x[start:start+64]
                        encoded,ls,_,n=core.encode(context=z)
                        parts.append(torch.cat([encoded[0][:,n,:],raw_features(z,ls)],dim=1).cpu())
                    feat=torch.cat(parts).numpy();assert np.isfinite(feat).all()
                    for t,f in zip(a['cycle'],feat):records[f'{b}/{length}/{int(t)}']=f
                    # Frozen transformations are independent per sample, including across batteries.
                    e,ls,_,n=core.encode(context=x[:1]);single=torch.cat([e[0][:,n,:],raw_features(x[:1],ls)],dim=1)
                    np.testing.assert_allclose(single.cpu().numpy()[0],feat[0],rtol=2e-4,atol=2e-5)
                    diagnostics.append(dict(battery=b,length=length,n=len(x),embedding_std=float(feat[:,:768].std()),
                        meta_std=feat[:,-3:].std(axis=0).tolist()))
        self.record=dict(protocol_sha256=ph,records=records,diagnostics=diagnostics)
        save(self.path,self.record);write(out/'feature_diagnostics.json',diagnostics)
        del core,pipe
        if device=='cuda':torch.cuda.empty_cache()

    def get(self,a,length):
        return np.stack([self.record['records'][f'{b}/{length}/{int(t)}'] for b,t in zip(a['battery'],a['cycle'])])


def fit(c,config,lr,seed,cells,cache,out,protocol):
    key=f'{c["fold"]}__L{c["length"]}__{config["name"]}__lr{lr}__s{seed}'
    directory=out/'candidates'/key
    if (directory/'selected.json').exists():
        result=read(directory/'selected.json');assert result['weights_sha256']==sha256(directory/'weights.pt');return result
    train=arrays(cells,c,c['train_ids'],'16');val=arrays(cells,c,c['validation_ids'])
    tf,vf=cache.get(train,c['length']),cache.get(val,c['length'])
    scaler=fit_scaler(tf,train['battery'],config['normalization'])
    device=cache.device;tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device=device)
    tx,vx=tensor(transform(tf,scaler)),tensor(transform(vf,scaler))
    y,vy,vw=tensor(train['target']),tensor(val['target']),tensor(cell_weights(val['battery']))
    const=constants(train['target'],train['battery']);scale=max(1.,const['source_median'])
    head=new_head(771,scale,const['source_mean'],seed,device)
    opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=config['weight_decay'])
    history=[];best=None;state=None
    def evaluate(step):
        nonlocal best,state
        with torch.no_grad():
            score=float(torch.sum(vw*abs(output(head,vx,scale)-vy)))
        history.append(dict(step=step,validation_mae=score))
        if best is None or score<best['validation_mae']:
            best=history[-1];state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    evaluate(0);start=time.monotonic()
    rng=np.random.default_rng(seed)
    indices=torch.as_tensor(rng.choice(len(y),size=(protocol['steps'],protocol['batch_size']),p=cell_weights(train['battery'])),device=device)
    for step in range(1,protocol['steps']+1):
        idx=indices[step-1];opt.zero_grad(set_to_none=True)
        pred=output(head,tx[idx],scale);loss=loss_value(pred,y[idx],scale,config['loss'])
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
        opt.param_groups[0]['lr']=lr*(1-step/protocol['steps'])
        if step%protocol['eval_every']==0:
            assert torch.isfinite(loss);evaluate(step)
    bundle=dict(head=state,scaler=scaler,scale=scale,constants=const)
    save(directory/'weights.pt',bundle);pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
    result=dict(**best,key=key,fold=c['fold'],length=c['length'],config=config,lr=lr,seed=seed,
        seconds=time.monotonic()-start,weights_path=str((directory/'weights.pt').relative_to(out)),
        weights_sha256=sha256(directory/'weights.pt'),train_labels=len(y),parameters=51015,
        protocol_sha256=sha256(out/'protocol.json'))
    write(directory/'selected.json',result)
    return result


def reference(c,seed,cells,cache,out):
    key=f'nasa_aging__{c["fold"]}__s{seed}__L{c["length"]}'
    directory=REFERENCE/'checkpoints'/key;s=read(directory/'selected.json')
    assert s['weights_sha256']==sha256(directory/'weights.pt')
    state=torch.load(directory/'weights.pt',weights_only=True,map_location='cpu')
    head=new_head(771,s['scale'],s['constants']['source_mean'],seed,cache.device);head.load_state_dict(state)
    val=arrays(cells,c,c['validation_ids']);features=cache.get(val,c['length'])
    with torch.no_grad(): pred=output(head,torch.as_tensor(features,device=cache.device),s['scale']).cpu().numpy()
    score=float(np.sum(cell_weights(val['battery'])*abs(pred-val['target'])))
    assert np.isclose(score,s['validation_mae'],rtol=2e-4,atol=.01),(score,s['validation_mae'])
    rel=Path('references')/(key+'.pt')
    save(out/rel,dict(head=state,scaler=fit_scaler(features,val['battery'],'none'),scale=s['scale'],constants=s['constants']))
    return dict(key=key,fold=c['fold'],length=c['length'],seed=seed,config=dict(name='historical',normalization='none',loss='huber1',weight_decay=0.),
        lr=s['lr'],step=s['step'],validation_mae=score,weights_path=rel.as_posix(),weights_sha256=sha256(out/rel),parameters=51015)


def train(out,cells,protocol,cache):
    if (out/'selection_lock.json').exists():return read(out/'selection_lock.json')['models']
    results=[];references={};total=len(protocol['cases'])*len(protocol['grid'])*len(protocol['lrs']);done=0
    for c in protocol['cases']:
        ref=reference(c,42,cells,cache,out);results.append(ref);references[(c['fold'],c['length'])]=ref
        for config in protocol['grid']:
            for lr in protocol['lrs']:
                result=fit(c,config,lr,42,cells,cache,out,protocol);results.append(result);done+=1
                status(out,'validation_search',completed=done,total=total,case=result['key'],validation_mae=result['validation_mae'],seconds=result['seconds'])
    write(out/'validation_search.json',results)
    selections=[]
    for fold in sorted({c['fold'] for c in protocol['cases']}):
        relevant=[r for r in results if r['fold']==fold]
        selectors={'historical_L32':references[fold,32],
            'tuned_L32':min([r for r in relevant if r['length']==32],key=lambda r:r['validation_mae']),
            'tuned_context':min(relevant,key=lambda r:r['validation_mae'])}
        for method,winner in selectors.items():
            c=next(c for c in protocol['cases'] if c['fold']==fold and c['length']==winner['length'])
            for seed in protocol['seeds']:
                if seed==42:r=winner
                elif winner['config']['name']=='historical':r=reference(c,seed,cells,cache,out)
                else:r=fit(c,winner['config'],winner['lr'],seed,cells,cache,out,protocol)
                selections.append(dict(**r,method=method,test_ids=c['test_ids'],validation_ids=c['validation_ids'],train_ids=c['train_ids']))
    write(out/'selection_lock.json',dict(models=selections,protocol_sha256=sha256(out/'protocol.json'),
        locked_at=datetime.now(timezone.utc).isoformat(),test_inference_started=False))
    return selections


def evaluate(out,cells,cache,models):
    assert (out/'selection_lock.json').exists()
    rows=[]
    for model in models:
        path=out/model['weights_path'];assert sha256(path)==model['weights_sha256']
        state=torch.load(path,weights_only=False,map_location='cpu')
        head=new_head(771,state['scale'],state['constants']['source_mean'],model['seed'],cache.device)
        head.load_state_dict(state['head']);c=dict(length=model['length'])
        for b in model['test_ids']:
            assert b not in model['train_ids']+model['validation_ids']
            a=arrays(cells,c,[b]);features=transform(cache.get(a,model['length']),state['scaler'])
            with torch.no_grad():
                pred=output(head,torch.as_tensor(features,device=cache.device),state['scale']).cpu().numpy()
            assert np.isfinite(pred).all()
            for t,y,p in zip(a['cycle'],a['target'],pred):
                rows.append(dict(method=model['method'],seed=model['seed'],battery_id=b,origin=int(t),
                    true_rul=float(y),predicted_rul=float(p),length=model['length'],
                    original_stride16=(t-32)%16==0,life_fraction=float(t/cells['nasa_aging',b]['eol'])))
    frame=pd.DataFrame(rows);frame.to_csv(out/'predictions.csv',index=False)
    rows=[]
    for scope,subset in [('original_stride16',frame[frame.original_stride16]),('dense_stride1',frame)]:
        for (method,seed,b),a in subset.groupby(['method','seed','battery_id']):
            y,p=a.true_rul.to_numpy(),a.predicted_rul.to_numpy();d=p-y;den=np.sum((y-y.mean())**2)
            rows.append(dict(scope=scope,method=method,seed=seed,battery_id=b,n=len(y),mae=float(np.mean(abs(d))),
                rmse=float(np.sqrt(np.mean(d*d))),r2=float(1-np.sum(d*d)/den) if den>0 else None,bias=float(d.mean())))
    cm=pd.DataFrame(rows);cm.to_csv(out/'cell_metrics.csv',index=False)
    sm=cm.groupby(['scope','method','seed'])[['mae','rmse','r2','bias']].mean().reset_index();sm.to_csv(out/'seed_metrics.csv',index=False)
    summary=sm.groupby(['scope','method'])[['mae','rmse','r2','bias']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'summary.csv',index=False)
    write(out/'verification.json',dict(status='passed',architecture_changed=False,head_parameters=51015,
        source_cells=2,labels_per_source_cell=16,selected_models=len(models),predictions=len(frame),
        source_test_disjoint=True,train_only_preprocessing=True,selection_locked_before_test=True,
        pretrained_feature_single_batch_parity=True,reference_validation_parity=True,
        predictions_sha256=sha256(out/'predictions.csv')))
    status(out,'completed',models=len(models),prediction_rows=len(frame))
    print(summary.to_string(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--cache-root',type=Path,default=ROOT/'models/huggingface')
    p.add_argument('--device',default='cuda',choices=['cuda','xpu','cpu']);p.add_argument('--prepare-only',action='store_true');p.add_argument('--fit-only',action='store_true');a=p.parse_args()
    out=a.out.resolve();cells,protocol=prepare(out)
    if a.prepare_only:status(out,'prepared',cases=len(protocol['cases']),search_fits=168);return
    if a.device=='cuda':assert torch.cuda.is_available()
    torch.set_num_threads(4)
    cache=Cache(out,cells,a.device,a.cache_root);models=train(out,cells,protocol,cache)
    if not a.fit_only:evaluate(out,cells,cache,models)


if __name__=='__main__':main()
