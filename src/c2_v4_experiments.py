"""Version-4 experiment data and direct RUL models. No future input to encoders."""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.preprocessing_v4 import load_v4_bundle
from src.c2_head_protocol import GROUPS, make_folds, natural_key
from src.c2_rul import eligible_mask, first_crossing

LENGTHS = (2, 4, 8, 16, 32, 64)
RUL_LENGTH = 32
METHODS = ('head', 'full', 'lora')


def load_cells(root):
    cells, fingerprints, seen = {}, {}, set()
    for group in GROUPS:
        data, meta, manifest = load_v4_bundle(Path(root) / group)
        fingerprints[group] = manifest['artifact_sha256']
        metadata = meta.set_index('battery_id')
        for battery, frame in data.groupby('battery_id', sort=True):
            frame = frame.sort_values('cycle').reset_index(drop=True)
            cycles = frame.cycle.to_numpy(dtype=int)
            if not np.array_equal(cycles, np.arange(1, cycles[-1] + 1)):
                raise ValueError(f'Nonconsecutive original record axis: {group}/{battery}')
            keep = eligible_mask(frame).to_numpy()
            physical = str(metadata.loc[battery].get('physical_battery_id', battery))
            identity = ('mit' if group.startswith('mit_') else group, physical)
            if identity in seen:
                raise ValueError(f'Duplicate physical cell: {identity}')
            seen.add(identity)
            if not keep.any():
                continue
            values = np.where(keep, frame.soh.to_numpy(dtype=float), np.nan)
            eol = first_crossing(cycles, values, .8)
            cells[(group, battery)] = dict(group=group, battery_id=battery,
                physical_battery_id=physical, cycles=cycles, values=values, keep=keep,
                eol=eol, last=int(cycles[keep][-1]), frame=frame)
    return cells, fingerprints


def soh_windows(cell):
    """Matched 64-position target axis, with at least one valid last-two input."""
    y = cell['values']
    windows = np.lib.stride_tricks.sliding_window_view(y, 65)
    valid = np.isfinite(windows[:, -1]) & np.isfinite(windows[:, -3:-1]).any(axis=1)
    x = windows[valid, :-1].copy().astype(np.float32)
    t = cell['cycles'][64:][valid]
    return dict(context=x, target=windows[valid, -1].copy(), cycle=t)


def rul_windows(cell, stride=16):
    """Fixed-clock origins, eligible current value, absorbing first SOH<=.8."""
    origins = []
    for t in range(RUL_LENGTH, cell['last'] + 1, stride):
        if cell['eol'] is not None and t >= cell['eol']:
            continue
        if np.isfinite(cell['values'][t-1]):
            origins.append(t)
    origins = np.asarray(origins, dtype=int)
    x = np.stack([cell['values'][t-RUL_LENGTH:t] for t in origins]).astype(np.float32) if len(origins) else np.empty((0, RUL_LENGTH), np.float32)
    return dict(context=x, cycle=origins,
                target=np.full(len(origins), np.nan) if cell['eol'] is None else cell['eol'] - origins)


def four_labels(windows):
    if len(windows['cycle']) < 4:
        raise ValueError('Four distinct source label origins required')
    return np.linspace(0, len(windows['cycle'])-1, 4, dtype=int).tolist()


def small_folds(ids):
    ids = sorted(ids, key=natural_key)
    if len(ids) < 4:
        raise ValueError('Two train + one validation + one test cell required')
    if len(ids) == 4:
        return [dict(fold_id=f'test_{test}', train_ids=[b for b in ids if b not in (test, ids[(i+1)%4])],
                     validation_ids=[ids[(i+1)%4]], test_ids=[test]) for i, test in enumerate(ids)]
    return [dict(fold_id='fixed', train_ids=ids[:2], validation_ids=ids[2:3], test_ids=ids[3:])]


def build_design(cells):
    soh, rul, coverage = {}, {}, []
    for group in GROUPS:
        records = [c for (g, _), c in cells.items() if g == group]
        usable = [c for c in records if len(soh_windows(c)['cycle']) >= 4]
        soh[group] = make_folds(group, usable)
        event = [c['battery_id'] for c in records if c['eol'] is not None and len(rul_windows(c, 1)['cycle']) >= 4]
        if len(event) >= 4:
            rul[group] = small_folds(event)
            # Censored test cells do not contribute exact training/validation labels.
            rul[group][0]['test_ids'] += sorted([c['battery_id'] for c in records if c['eol'] is None], key=natural_key)
        coverage.append(dict(dataset=group, usable_cells=len(records), event_cells=sum(c['eol'] is not None for c in records),
                             censored_cells=sum(c['eol'] is None for c in records), direct_within_dataset=len(event)>=4))
    # Fixed source transfer, declared before inference. These target groups never
    # contribute labels or validation to their transferred model.
    transfers = {'mit_batch1': 'mit_batch2', 'mit_batch3': 'mit_batch2', 'xjtu_3c': 'xjtu_2c'}
    for target, source in transfers.items():
        if target in rul:
            raise ValueError('Transfer target unexpectedly has a within-group model')
        rul[source][0].setdefault('transfer_test', {})[target] = sorted([b for (g,b) in cells if g==target], key=natural_key)
    budgets = []
    for task, groups in [('soh', soh), ('rul', rul)]:
        for group, folds in groups.items():
            for fold in folds:
                roles = [set(fold[k]) for k in ('train_ids','validation_ids','test_ids')]
                if any(roles[i] & roles[j] for i,j in ((0,1),(0,2),(1,2))):
                    raise ValueError('Cell leakage')
                for split, key in [('train','train_ids'),('validation','validation_ids')]:
                    for battery in fold[key]:
                        w = soh_windows(cells[(group,battery)]) if task=='soh' else rul_windows(cells[(group,battery)],1)
                        for i in four_labels(w):
                            budgets.append(dict(task=task,dataset=group,fold=fold['fold_id'],split=split,
                                                battery_id=battery,index=i,cycle=int(w['cycle'][i]),target=float(w['target'][i])))
    return dict(soh=soh, rul=rul, transfers=transfers), pd.DataFrame(budgets), pd.DataFrame(coverage)


def source_arrays(cells, budgets, task, group, fold, split, length):
    rows = budgets[(budgets.task==task)&(budgets.dataset==group)&(budgets.fold==fold)&(budgets.split==split)]
    xs, ys = [], []
    for r in rows.itertuples():
        w = soh_windows(cells[(group,r.battery_id)]) if task=='soh' else rul_windows(cells[(group,r.battery_id)],1)
        assert int(w['cycle'][r.index]) == r.cycle and float(w['target'][r.index]) == r.target
        xs.append(w['context'][r.index,-length:]); ys.append(r.target)
    return np.stack(xs), np.asarray(ys,dtype=np.float32)


def raw_features(context, loc_scale):
    finite = torch.isfinite(context)
    last_idx = torch.where(finite, torch.arange(context.shape[1],device=context.device), -1).max(dim=1).values
    if (last_idx < 0).any():
        raise ValueError('All-missing context')
    last = context.gather(1,last_idx[:,None])
    return torch.cat([loc_scale[0].reshape(-1,1), loc_scale[1].reshape(-1,1), last],dim=1)


class DirectRUL(nn.Module):
    """Pretrained C2 REG features + absolute SOH -> nonnegative scalar RUL."""
    def __init__(self, pristine, mode, scale, mean_target, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.mode, self.scale = mode, float(scale)
        base = deepcopy(pristine)
        for p in base.parameters():
            p.requires_grad_(mode=='full')
        # The native future-SOH head is unused in this regression task.
        for p in base.output_patch_embedding.parameters():
            p.requires_grad_(False)
        if mode=='lora':
            from peft import LoraConfig, get_peft_model
            base = get_peft_model(base, LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,bias='none',
                target_modules=['self_attention.q','self_attention.k','self_attention.v','self_attention.o']))
        self.backbone = base
        torch.manual_seed(seed)  # identical head initialization across adaptation modes
        self.head = nn.Sequential(nn.LayerNorm(pristine.config.d_model+3),
            nn.Linear(pristine.config.d_model+3,64), nn.GELU(), nn.Linear(64,1)).to(pristine.device)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, math.log(math.expm1(float(mean_target)/self.scale)))
        self.eval()

    def core(self):
        return self.backbone.get_base_model() if self.mode=='lora' else self.backbone

    def features(self, context):
        core = self.core()
        if not core.chronos_config.use_reg_token:
            raise ValueError('This protocol requires the native REG token')
        encoded, loc_scale, _, n_patches = core.encode(context=context)
        return torch.cat([encoded[0][:,n_patches,:], raw_features(context,loc_scale)],dim=1)

    def from_features(self, features):
        return torch.nn.functional.softplus(self.head(features).squeeze(-1))*self.scale

    def forward(self, context):
        return self.from_features(self.features(context))

    def trainable_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def load_trainable(self,state):
        expected={n for n,p in self.named_parameters() if p.requires_grad}
        if set(state)!=expected:
            raise ValueError('Checkpoint trainable scope mismatch')
        parameters=dict(self.named_parameters())
        with torch.no_grad():
            for n,v in state.items(): parameters[n].copy_(v)
        self.eval()
