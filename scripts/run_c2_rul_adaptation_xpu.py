"""Resumable laptop XPU Full/LoRA follow-up to the frozen nominal80 Head run."""
import argparse,gc,importlib.metadata,json,math,os,platform,sys,time,traceback
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from src.battery_zeroshot import sha256
from src.c2_v4_experiments import DirectRUL
from src.c2_rul_context_nominal import load,assemble,LABEL_FILE
from src.c2_rul_learning_curve import cell_weights,constants
from scripts.run_c2_v4_suite import write,read,now

REFERENCE=ROOT/'results/c2-battery/rul-context-nominal80-v1'
REVISION='29ec3766d36d6f73f0696f85560a422f50e8498c'
DEVICE='xpu'

def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)

def status(out,phase,**kw):
    value=dict(status=phase,updated_at=now(),pid=os.getpid(),**kw)
    write(out/'status.json',value);print(json.dumps(value),flush=True)

def save(path,state):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)

def source_files():
    return [Path(__file__),ROOT/'scripts/report_c2_rul_adaptation_xpu.py',ROOT/'scripts/run_c2_v4_suite.py',
            ROOT/'src/c2_rul_context_nominal.py',ROOT/'src/c2_rul_learning_curve.py',ROOT/'src/c2_v4_experiments.py',
            ROOT/'src/preprocessing_v4.py',ROOT/'src/c2_rul.py',ROOT/'src/soh_data.py',ROOT/'src/c2_head_protocol.py']

def prepare(out):
    ref=read(REFERENCE/'protocol.json');cells,hashes,audit=load(ROOT)
    assert hashes==ref['data_artifact_sha256']
    assert sha256(ROOT/LABEL_FILE)==ref['label_sha256']
    cases=[]
    # Begin with L32, then complete every prespecified length/seed. No test-driven ordering.
    for group in ['mit_batch1','mit_batch3','mit_batch2']:
        for length in [32,8,16,64,128]:
            for c in ref['cases']:
                if c['dataset']!=group or c['length']!=length:continue
                for mode in ['full','lora']:
                    q=deepcopy(c);q['reference_case_id']=q['case_id'];q['mode']=mode;q['case_id']+='__'+mode;cases.append(q)
    assert len(cases)==90
    protocol=dict(task='MIT nominal80 direct RUL Full/LoRA on laptop XPU',revision=REVISION,device=DEVICE,dtype='float32',
        reference_protocol_sha256=sha256(REFERENCE/'protocol.json'),reference_predictions_sha256=sha256(REFERENCE/'predictions.csv'),
        data_artifact_sha256=hashes,label_sha256=sha256(ROOT/LABEL_FILE),cases=cases,
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in source_files()},
        steps=1000,batch_size=16,eval_every=50,resume_every=100,inference_batch=128,
        backbone_lrs={'full':[1e-6,1e-5],'lora':[1e-4,3e-4]},head_lr=.001,
        optimizer='AdamW, no weight decay, linear LR decay, Huber scaled by balanced train median, clip norm1, same cell-balanced seed batches as Head reference.',
        adaptation='Full updates all encoder parameters; unused native future-SOH output head frozen. LoRA rank8 alpha16 dropout0 attention q/k/v/o plus regression head.',
        data_scope='Same source2/validation1/test cells, 16 labels/source cell, matched t>=128 and length8/16/32/64/128, seeds42/43/44 as reference. MIT nominal SOH=Q/1.1; audited official EOL. Never train on target transfer cells.',
        test_policy='All90 selections locked before test inference. Step0 allowed, equal-cell validation MAE selection. Previously inspected data: exploratory follow-up.',
        device_comparison='Reference Head results ran on CUDA. This run uses XPU FP32; saved-head device parity checked before testing. Device arithmetic may differ.',
        checkpoint_policy='Selected trainable tensors retained for all90 models. Current optimizer/model saved every100 updates for restart; completed cases hash-verified and skipped.')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol,'Frozen protocol changed'
    else:write(out/'protocol.json',protocol)
    audit.to_csv(out/'eol_audit.csv',index=False)
    # Reference label inventory proves identical source labels, including validation.
    labels=pd.read_csv(REFERENCE/'label_budget.csv');rows=[]
    for c in cases:
        for split,ids,budget in [('train',c['train_ids'],'16'),('validation',c['validation_ids'],'all')]:
            a=arrays(cells,c,ids,budget)
            z=labels[(labels.case_id==c['reference_case_id'])&(labels.split==split)]
            np.testing.assert_array_equal(z.battery_id,a['battery']);np.testing.assert_array_equal(z.origin_cycle,a['cycle']);np.testing.assert_array_equal(z.target,a['target'])
            rows.extend(dict(case_id=c['case_id'],split=split,battery_id=b,origin_cycle=int(t),target=float(y)) for b,t,y in zip(a['battery'],a['cycle'],a['target']))
        assert len(set(c['train_ids'])&set(c['test_ids']))==0
    pd.DataFrame(rows).to_csv(out/'label_budget.csv',index=False)
    return cells,protocol

def arrays(cells,c,ids,budget='all',group=None,stride=1):
    return assemble(cells,group or c['dataset'],ids,budget,stride,length=c['length'],start=c['start'])

@torch.no_grad()
def predict(model,x,batch=128):
    model.eval();values=[]
    for i in range(0,len(x),batch):values.append(model(tensor(x[i:i+batch])).cpu().numpy())
    q=np.concatenate(values);assert np.isfinite(q).all() and (q>=0).all();return q

def make(pristine,c,scale,mean):return DirectRUL(pristine,c['mode'],scale,mean,seed=c['seed'])

def fit(pristine,cells,c,out,p,index):
    directory=out/'checkpoints'/c['case_id'];directory.mkdir(parents=True,exist_ok=True)
    ph=sha256(out/'protocol.json')
    if (directory/'selected.json').exists():
        s=read(directory/'selected.json');assert s['protocol_sha256']==ph and s['weights_sha256']==sha256(directory/'weights.pt');return
    train=arrays(cells,c,c['train_ids'],'16');val=arrays(cells,c,c['validation_ids'])
    const=constants(train['target'],train['battery']);scale=max(1.,const['source_median']);vw=cell_weights(val['battery'])
    started=time.monotonic();history=[];best=None;audits={};resume_path=out/'.runtime/resume.pt'
    resume=torch.load(resume_path,map_location='cpu',weights_only=True) if resume_path.exists() else None
    if resume is not None:
        assert resume['protocol_sha256']==ph
        if resume['case_id']!=c['case_id']:resume=None
    if (directory/'progress.json').exists():
        progress=read(directory/'progress.json');assert progress['protocol_sha256']==ph
        history=progress['history'];best=progress['best'];audits=progress['gradient_audit']
    for li,lr in enumerate(p['backbone_lrs'][c['mode']]):
        if resume is not None and li<resume['lr_index']:continue
        if history and any(r['lr']==lr and r['step']==p['steps'] for r in history):continue
        model=make(pristine,c,scale,const['source_mean']);model.eval()
        frozen={n:v._version for n,v in model.named_parameters() if not v.requires_grad}
        params={n:v for n,v in model.named_parameters() if v.requires_grad}
        head=[v for n,v in params.items() if n.startswith('head.')];backbone=[v for n,v in params.items() if n.startswith('backbone.')]
        if c['mode']=='lora':assert all('lora_' in n for n in params if n.startswith('backbone.'))
        opt=torch.optim.AdamW([dict(params=head,lr=p['head_lr']),dict(params=backbone,lr=lr)],weight_decay=0.)
        start_step=0
        if resume is not None and li==resume['lr_index']:
            model.load_trainable(resume['model']);opt.load_state_dict(resume['optimizer']);start_step=resume['step']
            # Loading trainable tensors leaves frozen parameters untouched.
            resume=None
        x,y=tensor(train['context']),tensor(train['target'])
        idxs=np.random.default_rng(c['seed']).choice(len(y),size=(p['steps'],p['batch_size']),p=cell_weights(train['battery']))
        idxs=torch.as_tensor(idxs,device=DEVICE)
        def evaluate(step):
            nonlocal best
            score=float(np.sum(vw*abs(predict(model,val['context'],p['inference_batch'])-val['target'])))
            row=dict(lr=lr,step=step,validation_mae=score);history.append(row)
            if best is None or score<best['validation_mae']:
                best=row;save(directory/'best.pt',model.trainable_state())
            write(directory/'progress.json',dict(protocol_sha256=ph,history=history,best=best,gradient_audit=audits))
        if start_step==0:evaluate(0)
        for step in range(start_step+1,p['steps']+1):
            opt.zero_grad(set_to_none=True)
            pred=model(x[idxs[step-1]])
            loss=torch.nn.functional.huber_loss(pred/scale,y[idxs[step-1]]/scale,delta=1.)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            if step==5:
                audit=dict(head_nonzero=any(v.grad is not None and bool(v.grad.abs().sum()>0) for v in head),
                    backbone_nonzero=any(v.grad is not None and bool(v.grad.abs().sum()>0) for v in backbone),
                    trainable_parameters=sum(v.numel() for v in params.values()),trainable_names=list(params))
                assert audit['head_nonzero'] and audit['backbone_nonzero'];audits[str(lr)]=audit
            torch.nn.utils.clip_grad_norm_(list(params.values()),1.);opt.step()
            for pg,initial in zip(opt.param_groups,[p['head_lr'],lr]):pg['lr']=initial*(1-step/p['steps'])
            if step%p['eval_every']==0:
                evaluate(step)
                status(out,'training',case=index,total=len(p['cases']),case_id=c['case_id'],lr=lr,step=step,steps=p['steps'],best_validation_mae=best['validation_mae'])
            if step%p['resume_every']==0:
                save(resume_path,dict(case_id=c['case_id'],protocol_sha256=ph,lr_index=li,step=step,model=model.trainable_state(),optimizer=opt.state_dict()))
        parameters=dict(model.named_parameters())
        assert all(parameters[n]._version==v and parameters[n].grad is None for n,v in frozen.items())
        del model,opt,params,head,backbone,parameters,x,y,idxs;gc.collect();torch.xpu.empty_cache()
    best_path=directory/'best.pt' if (directory/'best.pt').exists() else directory/'weights.pt'
    state=torch.load(best_path,map_location='cpu',weights_only=True)
    model=make(pristine,c,scale,const['source_mean']);model.load_trainable(state)
    q=predict(model,val['context'],p['inference_batch']);score=float(np.sum(vw*abs(q-val['target'])))
    assert np.isclose(score,best['validation_mae'],rtol=2e-5,atol=.002)
    # Rename only the current case's owned temporary best checkpoint.
    if best_path.name=='best.pt':best_path.replace(directory/'weights.pt')
    pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
    write(directory/'selected.json',dict(**best,mode=c['mode'],scale=scale,mean=const['source_mean'],
        protocol_sha256=ph,weights_sha256=sha256(directory/'weights.pt'),train_labels=len(train['target']),validation_labels=len(val['target']),
        reload_verified=True,frozen_verified=True,gradient_audit=audits,elapsed_seconds=time.monotonic()-started))
    if resume_path.exists():resume_path.unlink()
    del model,state;gc.collect();torch.xpu.empty_cache()

def parity(pristine,cells,out,p):
    result=[]
    for g in ['mit_batch1','mit_batch2','mit_batch3']:
        c=next(c for c in p['cases'] if c['dataset']==g and c['length']==32 and c['seed']==42)
        directory=REFERENCE/'checkpoints'/c['reference_case_id'];s=read(directory/'selected.json')
        model=DirectRUL(pristine,'head',s['scale'],s['constants']['source_mean'],seed=42)
        model.head.load_state_dict(torch.load(directory/'weights.pt',weights_only=True,map_location='cpu'))
        a=arrays(cells,c,c['validation_ids'])
        score=float(np.sum(cell_weights(a['battery'])*abs(predict(model,a['context'])-a['target'])))
        assert np.isclose(score,s['validation_mae'],rtol=.002,atol=.1),(g,score,s['validation_mae'])
        result.append(dict(dataset=g,xpu_validation_mae=score,cuda_validation_mae=s['validation_mae']))
        del model;gc.collect();torch.xpu.empty_cache()
    write(out/'head_device_parity.json',dict(status='passed',checks=result))

def score_all(pristine,cells,out,p):
    paths=[];native_checks=0
    for c in p['cases']:
        path=out/'case_predictions'/f'{c["case_id"]}.csv';paths.append(path)
        if path.exists():continue
        directory=out/'checkpoints'/c['case_id'];s=read(directory/'selected.json')
        assert sha256(directory/'weights.pt')==s['weights_sha256']
        model=make(pristine,c,s['scale'],s['mean']);model.load_trainable(torch.load(directory/'weights.pt',map_location='cpu',weights_only=True))
        targets=[(c['dataset'],c['test_ids'],'within_dataset')]+[(g,ids,'transfer' if g!=c['dataset'] else 'censored') for g,ids in c['extra_tests'].items()]
        rows=[]
        for g,ids,scope in targets:
            a=arrays(cells,c,ids,group=g,stride=16);pred=predict(model,a['context'])
            idx=np.unique(np.linspace(0,len(pred)-1,min(3,len(pred)),dtype=int));check=predict(model,a['context'][idx],1)
            np.testing.assert_allclose(check,pred[idx],rtol=.002,atol=.1);native_checks+=len(idx)
            for b,t,y,v in zip(a['battery'],a['cycle'],a['target'],pred):
                cell=cells[g,str(b)]
                rows.append(dict(case_id=c['case_id'],dataset=g,source_dataset=c['dataset'],scope=scope,seed=c['seed'],length=c['length'],
                    battery_id=b,origin_cycle=int(t),method=c['mode'],event_observed=cell['eol'] is not None,
                    true_rul=float(y),predicted_rul=float(v),lower_bound=max(0,cell['last']-int(t))))
        path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(path.with_suffix('.tmp'),index=False);path.with_suffix('.tmp').replace(path)
        del model;gc.collect();torch.xpu.empty_cache()
        status(out,'scoring',completed=len(list(path.parent.glob('*.csv'))),total=90)
    pd.concat([pd.read_csv(path) for path in paths],ignore_index=True).to_csv(out/'predictions.csv',index=False)
    return native_checks

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();out=a.out.resolve()
    if (out/'complete.json').exists():print('Already complete');return
    cells,protocol=prepare(out)
    if a.prepare_only:status(out,'prepared',cases=len(protocol['cases']));return
    assert torch.xpu.is_available(),'This run requires the laptop XPU'
    torch.set_num_threads(4)
    # A single foreground/background owner prevents simultaneous GPU jobs in one result directory.
    lock=out/'.runtime/runner.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    import msvcrt
    handle=lock.open('a+b');handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
    msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    try:
        status(out,'loading_model')
        cache=ROOT/f'models/huggingface/models--amazon--chronos-2/snapshots/{REVISION}'
        pipe=Chronos2Pipeline.from_pretrained(str(cache),device_map=DEVICE,dtype=torch.float32,local_files_only=True)
        pristine=pipe.model.eval()
        if not (out/'environment.json').exists():write(out/'environment.json',dict(started_at=now(),host=platform.node(),device=torch.xpu.get_device_name(),
              packages={n:importlib.metadata.version(n) for n in ['torch','peft','chronos-forecasting','numpy','pandas']}))
        if not (out/'head_device_parity.json').exists():parity(pristine,cells,out,protocol)
        for i,c in enumerate(protocol['cases'],1):
            status(out,'case_start',case=i,total=90,case_id=c['case_id']);fit(pristine,cells,c,out,protocol,i)
        selections={x.relative_to(out).as_posix():sha256(x) for x in (out/'checkpoints').glob('*/selected.json')}
        assert len(selections)==90
        if not (out/'selection_lock.json').exists():write(out/'selection_lock.json',dict(locked_at=now(),selected_sha256=selections,protocol_sha256=sha256(out/'protocol.json')))
        else:assert selections==read(out/'selection_lock.json')['selected_sha256']
        status(out,'all_selections_locked',models=90)
        checked=score_all(pristine,cells,out,protocol)
        assert all(sha256(out/k)==v for k,v in selections.items())
        write(out/'execution_finished.json',dict(finished_at=now(),models=90,selection_unchanged=True,checked_single_batch_points=checked,
            predictions_sha256=sha256(out/'predictions.csv'),protocol_sha256=sha256(out/'protocol.json'),peak_xpu_memory_bytes=torch.xpu.max_memory_allocated()))
        del pipe,pristine;gc.collect();torch.xpu.empty_cache()
        from scripts.report_c2_rul_adaptation_xpu import report
        report(out);status(out,'complete',models=90)
    except BaseException:
        status(out,'failed',error=traceback.format_exc());raise
    finally:
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1);handle.close()

if __name__=='__main__':main()
