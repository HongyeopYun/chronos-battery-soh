"""Unchanged-architecture full-weight few-shot follow-up, after head validation."""
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from src.c2_v4_experiments import DirectRUL
from src.c2_rul_context_nominal import load
from src.c2_rul_learning_curve import cell_weights
from src.c2_rul_improvement import loss_value
from src.battery_zeroshot import sha256
from scripts.run_c2_nasa_fewshot_improvement import arrays,write,read,save,status,REVISION

HEAD=ROOT/'results/c2-battery/rul-nasa-fewshot-improvement-v1'


def prepare(out):
    hp=read(HEAD/'protocol.json');locked=read(HEAD/'selection_lock.json')
    assert locked['protocol_sha256']==sha256(HEAD/'protocol.json')
    models=[m for m in locked['models'] if m['method']=='tuned_L32'];assert len(models)==12
    cells,hashes,_=load(ROOT);assert hashes==hp['data_artifact_sha256']
    sources=[Path(__file__),ROOT/'src/c2_rul_improvement.py',ROOT/'src/c2_v4_experiments.py',ROOT/'scripts/run_c2_nasa_fewshot_improvement.py']
    protocol=dict(task='NASA full-weight few-shot with unchanged architecture',revision=REVISION,models=models,
        head_protocol_sha256=sha256(HEAD/'protocol.json'),head_selection_lock_sha256=sha256(HEAD/'selection_lock.json'),
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in sources},data_artifact_sha256=hashes,
        backbone_lrs=[1e-6,1e-5],head_lr=1e-4,steps=200,batch_size=16,eval_every=20,
        structure='Exactly existing DirectRUL(mode=full), no adapter/new layer/changed dimensions. Existing output SOH head unused/frozen. Existing encoder and scalar head weights trained.',
        budget='Same2 source cells x16 exact RUL labels; same validation1/test1 cells, t>=32,L32.',
        optimization='Warm start from validation-selected same-budget head. Keep its external source-only feature scaler and loss. AdamW no WD, linear decay, grad clip1. Baseline step0 allowed.',
        selection='Lock all12 models using validation MAE only before any full-model test prediction. Fixed LR grid declared independently of test outcome.',
        limitation='Historical datasets previously viewed; exploratory follow-up. Architecture unchanged but parameter-training scope differs from Head-only.')
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    else:write(out/'protocol.json',protocol)
    return cells,protocol


def build(pristine,warm):
    model=DirectRUL(pristine,'full',warm['scale'],warm['constants']['source_mean'],42)
    model.head.load_state_dict(warm['head']);model.eval()
    center=torch.as_tensor(warm['scaler']['center'],device='cuda')
    scale=torch.as_tensor(warm['scaler']['scale'],device='cuda')
    return model,center,scale


def predict(model,x,center,feature_scale):
    return model.from_features((model.features(x)-center)/feature_scale)


def fit(pristine,m,cells,out,protocol):
    case=f'{m["fold"]}__s{m["seed"]}';directory=out/'checkpoints'/case
    if (directory/'selected.json').exists():
        s=read(directory/'selected.json')
        if s['weights_path']:assert sha256(out/s['weights_path'])==s['weights_sha256']
        return s
    warm=torch.load(HEAD/m['weights_path'],weights_only=False,map_location='cpu')
    assert sha256(HEAD/m['weights_path'])==m['weights_sha256']
    train=arrays(cells,m,m['train_ids'],'16');val=arrays(cells,m,m['validation_ids'])
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
    x,y,vx,vy,vw=tensor(train['context']),tensor(train['target']),tensor(val['context']),tensor(val['target']),tensor(cell_weights(val['battery']))
    best=None;best_state=None;history=[];grad_audit=[];start=time.monotonic()
    for lr in protocol['backbone_lrs']:
        model,center,feature_scale=build(pristine,warm)
        head=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('head.')]
        backbone=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('head.')]
        opt=torch.optim.AdamW([dict(params=head,lr=protocol['head_lr']),dict(params=backbone,lr=lr)],weight_decay=0.)
        def validation(step):
            nonlocal best,best_state
            with torch.no_grad():
                predicted=torch.cat([predict(model,vx[i:i+64],center,feature_scale) for i in range(0,len(vx),64)])
                score=float(torch.sum(vw*abs(predicted-vy)))
            history.append(dict(step=step,backbone_lr=lr,validation_mae=score))
            if step==0:assert np.isclose(score,m['validation_mae'],rtol=.001,atol=.03),(case,score,m['validation_mae'])
            if best is None or score<best['validation_mae']:
                best=history[-1];best_state=model.trainable_state() if step>0 else None
        validation(0)
        rng=np.random.default_rng(m['seed']);indices=torch.as_tensor(rng.choice(len(y),size=(protocol['steps'],protocol['batch_size']),p=cell_weights(train['battery'])),device='cuda')
        for step in range(1,protocol['steps']+1):
            idx=indices[step-1];opt.zero_grad(set_to_none=True)
            p=predict(model,x[idx],center,feature_scale);loss=loss_value(p,y[idx],model.scale,m['config']['loss'])
            loss.backward()
            if step==5:
                audit=dict(backbone_lr=lr,head_grad=any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in head),
                    backbone_grad=any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in backbone),
                    unused_output_frozen=all(not p.requires_grad and p.grad is None for p in model.core().output_patch_embedding.parameters()))
                assert all(audit[k] for k in ['head_grad','backbone_grad','unused_output_frozen']);grad_audit.append(audit)
            torch.nn.utils.clip_grad_norm_([*head,*backbone],1.);opt.step()
            for g,base_lr in zip(opt.param_groups,[protocol['head_lr'],lr]):g['lr']=base_lr*(1-step/protocol['steps'])
            if step%protocol['eval_every']==0:
                assert torch.isfinite(loss);validation(step)
                status(out,'training_full',case=case,lr=lr,step=step,validation_mae=history[-1]['validation_mae'],elapsed=time.monotonic()-start)
        del opt,model,head,backbone;torch.cuda.empty_cache()
    directory.mkdir(parents=True,exist_ok=True)
    relative=None;weight_hash=None
    if best_state is not None:
        relative=str((directory/'weights.pt').relative_to(out));save(out/relative,best_state);weight_hash=sha256(out/relative)
    pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
    selected=dict(**best,case=case,reference=m,weights_path=relative,weights_sha256=weight_hash,
        baseline_validation_mae=m['validation_mae'],seconds=time.monotonic()-start,gradient_audit=grad_audit,
        architecture_changed=False,train_labels=len(y),trainable_parameters=115875399)
    write(directory/'selected.json',selected);return selected


def evaluate(pristine,cells,out,selected):
    assert (out/'selection_lock.json').exists();rows=[];reload_checks=[]
    for s in selected:
        m=s['reference'];warm=torch.load(HEAD/m['weights_path'],weights_only=False,map_location='cpu')
        model,center,feature_scale=build(pristine,warm)
        if s['weights_path']:
            assert sha256(out/s['weights_path'])==s['weights_sha256']
            model.load_trainable(torch.load(out/s['weights_path'],weights_only=True,map_location='cpu'))
        with torch.no_grad():
            val=arrays(cells,m,m['validation_ids']);vx=torch.as_tensor(val['context'],device='cuda')
            vp=torch.cat([predict(model,vx[i:i+64],center,feature_scale) for i in range(0,len(vx),64)]).cpu().numpy()
            vmae=float(np.sum(cell_weights(val['battery'])*abs(vp-val['target'])))
            assert np.isclose(vmae,s['validation_mae'],rtol=2e-4,atol=.01)
            reload_checks.append(dict(case=s['case'],validation_mae=vmae))
            for b in m['test_ids']:
                assert b not in m['train_ids']+m['validation_ids']
                a=arrays(cells,m,[b]);x=torch.as_tensor(a['context'],device='cuda')
                pred=torch.cat([predict(model,x[i:i+64],center,feature_scale) for i in range(0,len(x),64)]).cpu().numpy()
                single=predict(model,x[:1],center,feature_scale).cpu().numpy()[0]
                assert np.isclose(single,pred[0],rtol=2e-4,atol=.01)
                for t,y,p in zip(a['cycle'],a['target'],pred):
                    rows.append(dict(method='full_warm_L32',seed=m['seed'],battery_id=b,origin=int(t),true_rul=float(y),predicted_rul=float(p),length=32,
                        original_stride16=(t-32)%16==0,life_fraction=float(t/cells['nasa_aging',b]['eol'])))
        del model;torch.cuda.empty_cache()
    frame=pd.DataFrame(rows);frame.to_csv(out/'predictions.csv',index=False)
    write(out/'verification.json',dict(status='passed',architecture_changed=False,models=len(selected),reload_checks=reload_checks,
        source_test_disjoint=True,labels_per_source_cell=16,source_cells=2,selection_locked_before_test=True,
        predictions_sha256=sha256(out/'predictions.csv')))
    status(out,'completed',models=len(selected),predictions=len(frame))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    out=a.out.resolve();cells,protocol=prepare(out)
    if a.prepare_only:status(out,'prepared',models=12,updates=4800);return
    assert torch.cuda.is_available();torch.set_num_threads(4)
    pipe=Chronos2Pipeline.from_pretrained('amazon/chronos-2',revision=REVISION,device_map='cuda',dtype=torch.float32,cache_dir=str(a.cache_root),local_files_only=True)
    pristine=pipe.model.eval();selected=[]
    for m in protocol['models']:selected.append(fit(pristine,m,cells,out,protocol))
    write(out/'selection_lock.json',dict(models=selected,locked_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha256(out/'protocol.json')))
    evaluate(pristine,cells,out,selected)


if __name__=='__main__':main()
