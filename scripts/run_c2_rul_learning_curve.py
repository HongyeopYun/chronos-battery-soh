"""Frozen-backbone CUDA RUL source-cell and label-budget learning curves."""
import argparse
from copy import deepcopy
import importlib.metadata
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.battery_zeroshot import sha256
from src.c2_v4_experiments import DirectRUL, load_cells, rul_windows
from src.c2_rul_learning_curve import (make_cases, assemble, cell_weights, constants,
                                      fit_ridge, predict_ridge, SEEDS, BUDGETS)
from scripts.run_c2_v4_suite import REVISION, load_model, write, read, now, tensor, status


def prepare(out):
    cells, fingerprints = load_cells(ROOT / 'data/processed/reviewed_v4/forecast_safe')
    cases = make_cases(cells)
    paths = [Path(__file__), ROOT/'scripts/run_c2_v4_suite.py', *sorted((ROOT/'src').glob('*.py'))]
    protocol = dict(schema_version=1, task='RUL head-only learning curve', revision=REVISION,
        created_definition='Immutable design; creation timestamp stored separately.',
        data_artifact_sha256=fingerprints, source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in paths},
        seeds=list(SEEDS), budgets=list(BUDGETS), cases=cases, context_length=32,
        threshold=.8, eol='First eligible Q/Q_ref <= .8; original record positions, absorbing event.',
        source_origins='Eligible pre-EOL stride 1; nested 4,16,all. Histories remain 32 positions.',
        evaluation_origins='Original t=32,48,64,..., identical test origins in every budget.',
        label_cost='Labels within each completed source cell share an EOL and are correlated; all is not few-shot.',
        test_policy='MIT 20 common test IDs sampled from previous 40 test cells with RNG 20261013, without lifetime/error ranking. Previously inspected dataset: exploratory follow-up, not a pristine external test.',
        source_policy='MIT keeps b2c0,b2c1 at n=2; nested additional cells permuted by seed. 3 fixed validation cells. Other groups retain original four LOBO folds, train2/validation1/test1.',
        seeds_policy='MIT seeds change nested additional sources and initialization; elsewhere seeds change initialization. Fixed test set, not three independent test populations.',
        head='Unchanged v4 LayerNorm(771),Linear(771,64),GELU,Linear(64,1),softplus; C2 REG plus loc,scale,last SOH.',
        steps=1000, batch_size=16, eval_every=50, lrs=[1e-3,3e-3],
        optimization='AdamW no weight decay, linear LR decay, normalized Huber, gradient clip1, source-cell-balanced sampling.',
        selection='All eligible validation origins, equal-cell MAE, step0 allowed; all 144 selections locked before test predictions.',
        constants='Training-cell-balanced source mean and weighted median; median included because objective is MAE.',
        ridge='Train-standardized last/first/mean/std/slope/observed-fraction from same SOH32 input; cell-balanced ridge, nonnegative output; alpha selected on validation only.',
        ridge_alphas=[.001,.01,.1,1.,10.],
        scope='Observed-EOL within-dataset only. Censored cells and cross-dataset transfers omitted from this diagnostic.',
        limitations=['Existing v4 result remains unchanged.', 'Small groups cannot increase source-cell count with independent validation and test.',
            'Optimization/validation budget differs from old v4 suite; data effects must be compared within this new controlled sweep.',
            'CALCE record positions are not guaranteed physical cycle counts.', 'No test-driven revision of split or hyperparameters.'])
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        if read(out/'protocol.json') != protocol:
            raise ValueError('Frozen protocol mismatch; use a new result directory')
    else:
        write(out/'protocol.json', protocol)
    budgets = []
    for c in cases:
        for split, ids, budget in [('train', c['train_ids'], c['budget']), ('validation', c['validation_ids'], 'all')]:
            a = assemble(cells, c['dataset'], ids, budget)
            budgets.extend(dict(case_id=c['case_id'], split=split, battery_id=b, origin_cycle=int(t), target=float(y))
                           for b,t,y in zip(a['battery'], a['cycle'], a['target']))
    pd.DataFrame(budgets).to_csv(out/'label_budget.csv', index=False)
    pd.DataFrame([dict(dataset=g,battery_id=b,eol=c['eol']) for (g,b),c in cells.items()
                  if g in {c['dataset'] for c in cases} and c['eol'] is not None]).to_csv(out/'eol_audit.csv',index=False)
    return cells, protocol


def new_head(dim, scale, mean, seed):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,64), nn.GELU(), nn.Linear(64,1)).cuda()
    nn.init.zeros_(head[-1].weight)
    nn.init.constant_(head[-1].bias, math.log(math.expm1(mean / scale)))
    return head.eval()


def output(head, features, scale):
    return torch.nn.functional.softplus(head(features).squeeze(-1)) * scale


class FeatureCache:
    def __init__(self, model, cells):
        self.model, self.cells, self.cache = model, cells, {}

    @torch.no_grad()
    def get(self, group, data):
        result = []
        for b,t in zip(data['battery'], data['cycle']):
            key = (group, str(b))
            if key not in self.cache:
                w = rul_windows(self.cells[key], 1)
                features = []
                for start in range(0,len(w['context']),128):
                    features.append(self.model.features(tensor(w['context'][start:start+128])).detach())
                self.cache[key] = (dict(zip(w['cycle'].tolist(), range(len(w['cycle'])))), torch.cat(features))
            indices, features = self.cache[key]
            result.append(features[indices[int(t)]])
        return torch.stack(result)


def fit_case(c, cells, cache, out, protocol):
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


def score(cells, cache, out, protocol):
    rows = []
    for c in protocol['cases']:
        directory = out/'checkpoints'/c['case_id']
        s = read(directory/'selected.json')
        test = assemble(cells,c['dataset'],c['test_ids'],stride=16)
        features = cache.get(c['dataset'],test)
        head = new_head(s['dim'],s['scale'],s['constants']['source_mean'],c['seed'])
        head.load_state_dict(torch.load(directory/'weights.pt',map_location='cpu',weights_only=True))
        with torch.no_grad():
            predicted = output(head,features,s['scale']).cpu().numpy()
        predictions = dict(head=predicted,ridge=predict_ridge(s['ridge'],test['context']),
                           **{k:np.full(len(predicted),v) for k,v in s['constants'].items()})
        for method,pred in predictions.items():
            assert np.isfinite(pred).all() and (pred>=0).all()
            for b,t,y,p in zip(test['battery'],test['cycle'],test['target'],pred):
                rows.append(dict(**{k:c[k] for k in ('case_id','dataset','fold','seed','n_train_cells','budget')},
                    train_labels=s['train_labels'],method=method,battery_id=b,origin_cycle=int(t),true_rul=float(y),predicted_rul=float(p)))
    pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--cache-root',type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if (out/'execution_finished.json').exists():
        raise ValueError('Finished run is immutable')
    cells, protocol = prepare(out)
    if args.prepare_only:
        status(out,'prepared',cases=len(protocol['cases']))
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    pipe = load_model(args.cache_root)
    feature_model = DirectRUL(pipe.model,'head',100.,100.)
    for p in feature_model.parameters():
        p.requires_grad_(False)
        p.grad=None
    versions = {n:p._version for n,p in feature_model.named_parameters()}
    cache = FeatureCache(feature_model,cells)
    write(out/'environment.json',dict(started_at=now(),host=platform.node(),gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,packages={p:importlib.metadata.version(p) for p in ('torch','numpy','pandas','chronos-forecasting')}))
    # A cached head must produce exactly the same result as the existing v4 path.
    first = protocol['cases'][0]
    data = assemble(cells,first['dataset'],first['train_ids'],'4')
    x = tensor(data['context'])
    with torch.no_grad():
        h = new_head(771,100.,100.,42)
        h.load_state_dict(feature_model.head.state_dict())
        a = output(h,cache.get(first['dataset'],data),100.)
        torch.testing.assert_close(a,feature_model(x),rtol=2e-5,atol=1e-4)
        torch.testing.assert_close(feature_model.features(x[:1]),feature_model.features(x)[:1],rtol=2e-4,atol=.01)
    write(out/'smoke.json',dict(cached_native_parity=True,batch_independence=True))
    for i,c in enumerate(protocol['cases']):
        status(out,'training',case=i+1,total=len(protocol['cases']),case_id=c['case_id'])
        fit_case(c,cells,cache,out,protocol)
    selections={p.relative_to(out).as_posix():sha256(p) for p in sorted((out/'checkpoints').rglob('selected.json'))}
    assert len(selections)==144
    write(out/'selection_lock.json',dict(locked_at=now(),selected_sha256=selections,protocol_sha256=sha256(out/'protocol.json')))
    status(out,'all_selections_locked',count=len(selections))
    score(cells,cache,out,protocol)
    assert all(p._version==versions[n] and p.grad is None for n,p in feature_model.named_parameters())
    assert all(sha256(out/p)==h for p,h in selections.items())
    write(out/'execution_finished.json',dict(finished_at=now(),backbone_frozen_verified=True,selection_unchanged=True,
        protocol_sha256=sha256(out/'protocol.json'),predictions_sha256=sha256(out/'predictions.csv'),
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated()))
    status(out,'predictions_finished')


if __name__=='__main__':
    main()
