"""Training-only preprocessing for the unchanged 771-64-1 direct RUL head."""
import math
import numpy as np
import torch
from torch import nn
from src.c2_rul_learning_curve import cell_weights, constants


def fit_scaler(features, batteries, mode):
    x = np.asarray(features, dtype=np.float64)
    center, scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
    if mode not in ('none', 'all', 'metadata'):
        raise ValueError(mode)
    if mode != 'none':
        weights = cell_weights(batteries)
        mean = np.sum(x * weights[:, None], axis=0)
        std = np.sqrt(np.sum((x-mean)**2 * weights[:, None], axis=0))
        std[std < 1e-6] = 1.
        indices = np.arange(x.shape[1]) if mode == 'all' else np.arange(x.shape[1]-3, x.shape[1])
        center[indices], scale[indices] = mean[indices], std[indices]
    return dict(center=center.astype('float32'), scale=scale.astype('float32'), mode=mode)


def transform(features, scaler):
    return (np.asarray(features, dtype='float32') - scaler['center']) / scaler['scale']


def new_head(dim, target_scale, initial_mean, seed, device):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,64), nn.GELU(), nn.Linear(64,1)).to(device)
    nn.init.zeros_(head[-1].weight)
    nn.init.constant_(head[-1].bias, math.log(math.expm1(initial_mean/target_scale)))
    assert sum(p.numel() for p in head.parameters()) == 51015
    return head.eval()


def output(head, x, scale):
    return torch.nn.functional.softplus(head(x).squeeze(-1))*scale


def loss_value(pred, target, scale, name):
    p, y = pred/scale, target/scale
    if name == 'mae': return torch.nn.functional.l1_loss(p,y)
    if name == 'mse': return torch.nn.functional.mse_loss(p,y)
    if name in ('huber1','huber01'):
        return torch.nn.functional.huber_loss(p,y,delta=1. if name=='huber1' else .1)
    raise ValueError(name)


def configuration_grid():
    return [dict(name=name,normalization=normalization,loss=loss,weight_decay=wd)
        for name,normalization,loss,wd in [
            ('std_huber1','all','huber1',0.),
            ('std_huber01','all','huber01',0.),
            ('std_mae','all','mae',0.),
            ('std_mse','all','mse',0.),
            ('metadata_huber01','metadata','huber01',0.),
            ('metadata_mae','metadata','mae',0.),
            ('std_huber01_wd','all','huber01',.001),
        ]]
