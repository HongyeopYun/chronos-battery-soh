"""Locally reload every selected head in the 12-partition factorial audit."""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.battery_zeroshot import sha256
from src.c2_rul_context_nominal import load
from scripts.run_c2_nasa_fewshot_improvement import arrays

OUT = ROOT/'results/c2-battery/rul-nasa-fewshot-robustness-v1'
HEAD = ROOT/'results/c2-battery/rul-nasa-fewshot-improvement-v1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(4)
    protocol, lock = read(OUT/'protocol.json'), read(OUT/'selection_lock.json')
    assert lock['protocol_sha256'] == sha256(OUT/'protocol.json')
    assert protocol['frozen_features_sha256'] == sha256(HEAD/'.runtime/features.pt')
    for p, h in protocol['source_sha256'].items():
        assert sha256(ROOT/p) == h
    cells, hashes, _ = load(ROOT)
    assert hashes == protocol['data_artifact_sha256']
    cached = torch.load(HEAD/'.runtime/features.pt', weights_only=False, map_location='cpu')['records']
    pred = pd.read_csv(OUT/'predictions.csv')
    assert sha256(OUT/'predictions.csv') == read(OUT/'verification.json')['predictions_sha256']
    assert len(pred) == 8388 and len(lock['models']) == 144
    assert not pred.duplicated(['method','seed','fold','origin']).any()
    search = read(OUT/'validation_search.json')
    assert len(search) == 288
    for c in search:
        assert sha256(OUT/c['weights_path']) == c['weights_sha256']
        h = pd.read_csv((OUT/c['weights_path']).parent/'history.csv')
        best = h.loc[h.validation_mae.idxmin()]
        assert best.step == c['step'] and np.isclose(best.validation_mae, c['validation_mae'])
    rows = []
    for m in lock['models']:
        roles = [set(m[k]) for k in ['train_ids','validation_ids','test_ids']]
        assert [len(v) for v in roles] == [2,1,1] and len(set.union(*roles)) == 4
        eligible = [s for s in search if s['fold']==m['fold'] and s['seed']==m['seed'] and s['config']['name']==m['method']]
        assert len(eligible)==2 and m['key']==min(eligible,key=lambda r:r['validation_mae'])['key']
        state = torch.load(OUT/m['weights_path'],weights_only=False,map_location='cpu')
        def features(a):
            return np.stack([cached[f'{b}/32/{int(t)}'] for b,t in zip(a['battery'],a['cycle'])])
        train = arrays(cells,m,m['train_ids'],'16')
        assert len(train['target'])==32 and all(sum(train['battery']==b)==16 for b in roles[0])
        center,scale=np.zeros(771),np.ones(771)
        if m['config']['normalization']=='metadata':
            x=features(train).astype('float64')
            std=x[:,-3:].std(0);std[std<1e-6]=1
            center[-3:],scale[-3:]=x[:,-3:].mean(0),std
        np.testing.assert_allclose(state['scaler']['center'],center.astype('float32'),atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(state['scaler']['scale'],scale.astype('float32'),atol=1e-6,rtol=1e-6)
        head=nn.Sequential(nn.LayerNorm(771),nn.Linear(771,64),nn.GELU(),nn.Linear(64,1)).eval()
        head.load_state_dict(state['head'])
        assert sum(p.numel() for p in head.parameters())==51015
        def infer(a):
            x=(features(a)-state['scaler']['center'])/state['scaler']['scale']
            with torch.no_grad():
                return (torch.nn.functional.softplus(head(torch.tensor(x)).squeeze(-1))*state['scale']).numpy()
        val=arrays(cells,m,m['validation_ids'])
        vmae=float(np.mean(abs(infer(val)-val['target'])))
        assert np.isclose(vmae,m['validation_mae'],rtol=2e-4,atol=.01)
        test=arrays(cells,m,m['test_ids'])
        a=pred[(pred.method==m['method'])&(pred.seed==m['seed'])&(pred.fold==m['fold'])].sort_values('origin')
        np.testing.assert_array_equal(a.origin,test['cycle'])
        np.testing.assert_array_equal(a.true_rul,test['target'])
        delta=float(max(abs(infer(test)-a.predicted_rul.to_numpy())))
        assert delta<.01
        rows.append(dict(key=m['key'],validation_mae=vmae,prediction_max_difference=delta))
    parts=pred[['battery_id','validation_id']].drop_duplicates().groupby('battery_id').validation_id.nunique()
    assert (parts==3).all()
    pairs=pred.groupby(['battery_id','validation_id','method','seed']).origin.count()
    assert len(pairs)==144
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),selected_heads_reloaded=144,
                candidates_and_histories_verified=288,source_labels=32,head_parameters=51015,
                train_only_scaler_recomputed=True,source_validation_test_disjoint=True,validation_only_lr_step_selection=True,
                all_three_validation_choices_per_test_cell=True,
                max_prediction_difference=max(r['prediction_max_difference'] for r in rows),
                predictions_sha256=sha256(OUT/'predictions.csv'))
    pd.DataFrame(rows).to_csv(OUT/'independent_reload_checks.csv',index=False)
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
