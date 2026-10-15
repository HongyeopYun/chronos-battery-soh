"""Matched-origin direct RUL contexts, with audited MIT nominal-80% labels."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.c2_v4_experiments import load_cells, small_folds
from src.c2_rul_learning_curve import nested_indices
from src.c2_head_protocol import natural_key

LENGTHS = (8,16,32,64,128)
SEEDS = (42,43,44)
LABEL_FILE = 'results/c2-battery/mit-eol-label-audit-v1/usable_official_eol_labels.csv'

def load(root):
    cells, hashes = load_cells(Path(root)/'data/processed/reviewed_v4/forecast_safe')
    labels = pd.read_csv(Path(root)/LABEL_FILE)
    lookup = labels.set_index(['dataset','battery_id'])
    audit=[]
    for key,c in cells.items():
        old=c['eol']
        if key[0].startswith('mit_'):
            c['eol']=int(lookup.loc[key,'official_eol_cycle']) if key in lookup.index else None
            c['values']=np.where(c['keep'],c['frame'].capacity_ah.to_numpy()/1.1,np.nan)
            definition='official nominal 1.1Ah * 80%'
        else:definition='first eligible initial-reference SOH <= 80%'
        audit.append(dict(dataset=key[0],battery_id=key[1],old_eol=old,eol=c['eol'],
                          newly_labeled=old is None and c['eol'] is not None,definition=definition))
    assert sum(c['eol'] is not None for (g,b),c in cells.items() if g.startswith('mit_'))==124
    assert sum(r['newly_labeled'] for r in audit if r['dataset'].startswith('mit_'))==81
    return cells,hashes,pd.DataFrame(audit)

def windows(cell,length,start,stride=1):
    assert start>=length
    t=np.array([t for t in range(start,cell['last']+1,stride)
                if (cell['eol'] is None or t<cell['eol']) and np.isfinite(cell['values'][t-1])],dtype=int)
    x=np.stack([cell['values'][v-length:v] for v in t]).astype('float32') if len(t) else np.empty((0,length),dtype='float32')
    return dict(context=x,cycle=t,target=np.full(len(t),np.nan) if cell['eol'] is None else cell['eol']-t)

def assemble(cells,group,ids,budget='all',stride=1,*,length,start):
    pieces=[]
    for b in ids:
        w=windows(cells[group,b],length,start,stride)
        idx=np.arange(len(w['cycle'])) if budget=='all' else nested_indices(len(w['cycle']),budget)
        pieces.append(dict(context=w['context'][idx],cycle=w['cycle'][idx],target=w['target'][idx],battery=np.repeat(b,len(idx))))
    return {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}

def make_cases(cells):
    cases=[]
    for group in ('nasa_aging','calce_cs2_1c','mit_batch1','mit_batch2','mit_batch3','xjtu_2c'):
        lengths=(8,16,32) if group=='nasa_aging' else LENGTHS
        start=max(lengths)
        ids=sorted([b for (g,b),c in cells.items() if g==group and c['eol'] is not None],key=natural_key)
        assert all(len(windows(cells[group,b],start,start)['cycle'])>=16 for b in ids),group
        folds=small_folds(ids)
        for f in folds:
            for seed in SEEDS:
                for length in lengths:
                    c=dict(dataset=group,fold=f['fold_id'],seed=seed,length=length,start=start,budget='16',
                           n_train_cells=len(f['train_ids']),**{k:f[k] for k in ('train_ids','validation_ids','test_ids')})
                    c['case_id']=f'{group}__{c["fold"]}__s{seed}__L{length}'
                    c['extra_tests']={}
                    if f is folds[0]:
                        censored=[b for (g,b),cell in cells.items() if g==group and cell['eol'] is None]
                        if censored:c['extra_tests'][group]=censored
                        if group=='mit_batch2':
                            for target in ('mit_batch1','mit_batch3'):
                                c['extra_tests'][target]=sorted([b for (g,b) in cells if g==target],key=natural_key)
                        if group=='xjtu_2c':
                            c['extra_tests']['xjtu_3c']=sorted([b for (g,b) in cells if g=='xjtu_3c'],key=natural_key)
                    roles=[set(c[k]) for k in ('train_ids','validation_ids','test_ids')]
                    assert not any(roles[i]&roles[j] for i,j in ((0,1),(0,2),(1,2)))
                    cases.append(c)
    assert len(cases)==201
    return cases
