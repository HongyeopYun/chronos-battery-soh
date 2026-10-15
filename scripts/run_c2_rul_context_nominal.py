"""CUDA head-only context sweep. Historical artifacts are never changed."""
import argparse, importlib.metadata, platform, math, time, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.battery_zeroshot import sha256
from src.c2_v4_experiments import DirectRUL
from src.c2_rul_context_nominal import load, make_cases, assemble as build_arrays, LABEL_FILE
from src.c2_rul_learning_curve import cell_weights, constants, fit_ridge, predict_ridge
from scripts.run_c2_v4_suite import REVISION, load_model, tensor, write, read, now, status

def prepare(out):
    cells,hashes,audit=load(ROOT)
    paths=[Path(__file__),ROOT/'scripts/run_c2_v4_suite.py',*sorted((ROOT/'src').glob('*.py'))]
    protocol=dict(revision=REVISION,task='head-only direct RUL context sweep',
        source_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in paths},
        data_artifact_sha256=hashes,label_sha256=sha256(ROOT/LABEL_FILE),cases=make_cases(cells),
        steps=1000,batch_size=16,eval_every=50,lrs=[.001,.003],
        input='MIT Q/1.1 nominal SOH; others unchanged Q/initial reference. QC NaNs preserved. No cycle or EOL as input.',
        target='MIT audited official nominal80 cycle_life; others initial-reference80. RUL=EOL-t, t<EOL.',
        matched_origins='NASA t>=32 for L8/16/32; other groups t>=128 for L8/16/32/64/128. Stride16 evaluation; exact same origins and training labels per group across lengths.',
        budget='16 nested distinct labels per each of two source cells. Validation all eligible matched origins. Fixed cells and seeds42/43/44 across lengths.',
        split='Existing ID-ordered train2/val1/rest test for MIT; four-fold LOBO for NASA/CALCE/XJTU2. MIT B1/B3 also evaluated as transfer from MIT B2.',
        selection='Equal-cell validation MAE, step0 allowed, two learning rates. All selections locked before test.',
        limitation='Exploratory reused datasets. NASA L64/128 unavailable under complete-history cell-disjoint protocol. All-length scores condition on survival to t128; not early-life performance. CALCE record positions not assured physical cycles.',
        censored='MIT five author non-reaching cells remain censored; never replace with last+1. QC-excluded channels remain excluded.')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    else:write(out/'protocol.json',protocol)
    audit.to_csv(out/'eol_audit.csv',index=False)
    rows=[]
    for c in protocol['cases']:
        for split,ids,budget in [('train',c['train_ids'],'16'),('validation',c['validation_ids'],'all')]:
            a=build_arrays(cells,c['dataset'],ids,budget,length=c['length'],start=c['start'])
            rows.extend(dict(case_id=c['case_id'],split=split,battery_id=b,origin_cycle=int(t),target=float(y))
                        for b,t,y in zip(a['battery'],a['cycle'],a['target']))
    pd.DataFrame(rows).to_csv(out/'label_budget.csv',index=False)
    return cells,protocol

class FeatureCache:
    def __init__(self,model):self.model,self.cache=model,{}
    @torch.no_grad()
    def get(self,group,data):
        keys=[(group,str(b),int(t),data['context'].shape[1]) for b,t in zip(data['battery'],data['cycle'])]
        missing=[i for i,k in enumerate(keys) if k not in self.cache]
        for start in range(0,len(missing),128):
            idx=missing[start:start+128]
            feat=self.model.features(tensor(data['context'][idx])).detach().cpu()
            for i,row in zip(idx,feat):self.cache[keys[i]]=row.clone()
        return torch.stack([self.cache[k] for k in keys]).cuda()

def new_head(dim, scale, mean, seed):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,64), nn.GELU(), nn.Linear(64,1)).cuda()
    nn.init.zeros_(head[-1].weight)
    nn.init.constant_(head[-1].bias, math.log(math.expm1(mean / scale)))
    return head.eval()


def output(head, features, scale):
    return torch.nn.functional.softplus(head(features).squeeze(-1)) * scale


def fit_case(c, cells, cache, out, protocol):
    assemble = lambda cells, group, ids, budget='all': build_arrays(cells, group, ids, budget, length=c['length'], start=c['start'])
    directory = out/'checkpoints'/c['case_id']
    ph = sha256(out/'protocol.json')
    if (directory/'selected.json').exists():
        s = read(directory/'selected.json')
        assert s['protocol_sha256'] == ph and s['weights_sha256'] == sha256(directory/'weights.pt')
        return
    started = time.monotonic()
    train = assemble(cells, c['dataset'], c['train_ids'], c['budget'])
    val = assemble(cells, c['dataset'], c['validation_ids'])
    const = constants(train['target'], train['battery'])
    scale = max(1., const['source_median'])
    tx, vx = cache.get(c['dataset'], train), cache.get(c['dataset'], val)
    y, vy, vw = tensor(train['target']), tensor(val['target']), tensor(cell_weights(val['battery']))
    p = cell_weights(train['battery'])
    history, best, weights = [], None, None
    for lr in protocol['lrs']:
        head = new_head(tx.shape[1], scale, const['source_mean'], c['seed'])
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.)
        def evaluate(step):
            nonlocal best, weights
            with torch.no_grad():
                score = float(torch.sum(vw * abs(output(head,vx,scale)-vy)))
            row = dict(step=step, lr=lr, validation_mae=score)
            history.append(row)
            if best is None or score < best['validation_mae']:
                best = row
                weights = {n:v.detach().cpu().clone() for n,v in head.state_dict().items()}
        evaluate(0)
        rng = np.random.default_rng(c['seed'])
        all_indices = torch.as_tensor(rng.choice(len(y),size=(protocol['steps'],protocol['batch_size']),p=p),device='cuda')
        for step in range(1, protocol['steps']+1):
            idx = all_indices[step-1]
            opt.zero_grad(set_to_none=True)
            pred = output(head,tx[idx],scale)
            loss = torch.nn.functional.huber_loss(pred/scale,y[idx]/scale,delta=1.)
            loss.backward()
            if step == 5:
                assert any(v.grad is not None and bool(v.grad.abs().sum()>0) for v in head.parameters())
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            opt.step()
            opt.param_groups[0]['lr'] = lr*(1-step/protocol['steps'])
            if step % protocol['eval_every'] == 0:
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                evaluate(step)
    directory.mkdir(parents=True,exist_ok=True)
    torch.save(weights,directory/'weights.pt')
    head.load_state_dict(torch.load(directory/'weights.pt',weights_only=True,map_location='cpu'))
    with torch.no_grad():
        reload_mae = float(torch.sum(vw*abs(output(head,vx,scale)-vy)))
    assert np.isclose(reload_mae,best['validation_mae'],rtol=1e-5,atol=1e-4)
    ridge = fit_ridge(train,val)
    pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
    write(directory/'selected.json',dict(**best,scale=scale,dim=tx.shape[1],constants=const,ridge=ridge,
        protocol_sha256=ph,weights_sha256=sha256(directory/'weights.pt'),train_labels=len(y),validation_labels=len(vy),
        reload_verified=True,trainable_parameters=sum(p.numel() for p in head.parameters()),elapsed_seconds=time.monotonic()-started))

def score(cells,cache,out,protocol):
    rows=[]; native_checks=0
    for c in protocol['cases']:
        directory=out/'checkpoints'/c['case_id'];s=read(directory/'selected.json')
        head=new_head(s['dim'],s['scale'],s['constants']['source_mean'],c['seed'])
        head.load_state_dict(torch.load(directory/'weights.pt',map_location='cpu',weights_only=True))
        targets=[(c['dataset'],c['test_ids'],'within_dataset')]
        targets += [(g,ids,'transfer' if g!=c['dataset'] else 'censored') for g,ids in c['extra_tests'].items()]
        for group,ids,scope in targets:
            a=build_arrays(cells,group,ids,length=c['length'],start=c['start'],stride=16)
            with torch.no_grad():pred=output(head,cache.get(group,a),s['scale']).cpu().numpy()
            # Reload every selected head into the original uncached encoder path.
            idx=np.unique(np.linspace(0,len(pred)-1,min(3,len(pred)),dtype=int))
            cache.model.head.load_state_dict(head.state_dict());cache.model.scale=s['scale']
            with torch.no_grad():native=cache.model(tensor(a['context'][idx])).cpu().numpy()
            np.testing.assert_allclose(native,pred[idx],rtol=2e-4,atol=.03);native_checks+=len(idx)
            preds=dict(head=pred,ridge=predict_ridge(s['ridge'],a['context']),
                       **{k:np.full(len(pred),v) for k,v in s['constants'].items()})
            for method,values in preds.items():
                assert np.isfinite(values).all() and (values>=0).all()
                for b,t,y,p in zip(a['battery'],a['cycle'],a['target'],values):
                    cell=cells[group,str(b)]
                    rows.append(dict(case_id=c['case_id'],dataset=group,source_dataset=c['dataset'],scope=scope,
                        seed=c['seed'],length=c['length'],battery_id=b,origin_cycle=int(t),method=method,
                        event_observed=cell['eol'] is not None,true_rul=float(y),predicted_rul=float(p),
                        lower_bound=max(0,cell['last']-int(t))))
    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
    return native_checks

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--cache-root',type=Path)
    a=p.parse_args();out=a.out.resolve()
    if (out/'execution_finished.json').exists():raise ValueError('Finished run is immutable')
    cells,protocol=prepare(out)
    if a.prepare_only:status(out,'prepared',cases=len(protocol['cases']));return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=DirectRUL(load_model(a.cache_root).model,'head',100.,100.)
    for param in model.parameters():param.requires_grad_(False);param.grad=None
    versions={n:p._version for n,p in model.backbone.named_parameters()}
    cache=FeatureCache(model)
    write(out/'environment.json',dict(started_at=now(),host=platform.node(),gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
          packages={p:importlib.metadata.version(p) for p in ('torch','numpy','pandas','chronos-forecasting')}))
    for i,c in enumerate(protocol['cases']):
        status(out,'training',case=i+1,total=len(protocol['cases']),case_id=c['case_id'])
        fit_case(c,cells,cache,out,protocol)
    selections={p.relative_to(out).as_posix():sha256(p) for p in sorted((out/'checkpoints').rglob('selected.json'))}
    assert len(selections)==len(protocol['cases'])
    write(out/'selection_lock.json',dict(locked_at=now(),selected_sha256=selections,protocol_sha256=sha256(out/'protocol.json')))
    status(out,'all_selections_locked',count=len(selections))
    native=score(cells,cache,out,protocol)
    assert all(p._version==versions[n] and p.grad is None for n,p in model.backbone.named_parameters())
    assert all(sha256(out/p)==h for p,h in selections.items())
    write(out/'execution_finished.json',dict(finished_at=now(),backbone_frozen_verified=True,selection_unchanged=True,
          native_reload_test_points=native,protocol_sha256=sha256(out/'protocol.json'),predictions_sha256=sha256(out/'predictions.csv'),
          peak_cuda_memory_bytes=torch.cuda.max_memory_allocated()))
    status(out,'predictions_finished')

if __name__=='__main__':main()

