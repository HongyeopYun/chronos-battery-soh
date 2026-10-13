"""Head-only adaptation on a few partially observed future SOH trajectories."""
import numpy as np
import torch

from src.c2_rul import eligible_mask


def source_trajectories(data, context_length=64, horizon=256):
    """One fixed t=64 trajectory per source cell, without synthetic future labels."""
    result = {}
    for battery, raw in data.groupby('battery_id', sort=True):
        frame = raw.sort_values('cycle').set_index('cycle').copy()
        frame['observed_soh'] = frame.soh.where(eligible_mask(frame))
        context = frame.reindex(np.arange(1, context_length+1)).observed_soh.to_numpy(dtype=np.float32)
        target = frame.reindex(np.arange(context_length+1, context_length+horizon+1)).observed_soh.to_numpy(dtype=np.float32)
        if not np.isfinite(context).any() or not np.isfinite(target).any():
            raise ValueError('Each source trajectory needs observed context and future labels')
        result[battery] = dict(context=context, target=target)
    return result


def masked_quantile_loss(normalized_prediction, normalized_target, levels):
    """Native C2 pinball form; equal cells, valid-point mean, native quantile sum.

    Official C2 averages masked zero loss over the full horizon. Correct each
    cell by H / valid_points so short records do not silently lose weight.
    """
    p, y = normalized_prediction, normalized_target
    if p.ndim != 3 or y.shape != (p.shape[0], p.shape[2]) or len(levels) != p.shape[1]:
        raise ValueError('Expected [cell,quantile,horizon] predictions and aligned targets')
    valid = torch.isfinite(y)
    count = valid.sum(dim=-1)
    if (count == 0).any():
        raise ValueError('Empty supervision is not a training example')
    target = torch.where(valid, y, 0.)[:,None,:]
    q = levels.to(p.device).view(1,-1,1)
    pinball = 2 * torch.abs((target-p) * ((target <= p).to(p.dtype)-q))
    per_cell = (pinball * valid[:,None,:]).sum(dim=(1,2)) / count
    return per_cell.mean()


@torch.no_grad()
def cache_features(model, context, target, horizon=256):
    """Cache only the frozen encoder's context-derived features (no target input)."""
    model.eval()
    x = torch.as_tensor(context, device=model.device, dtype=torch.float32)
    y = torch.as_tensor(target, device=model.device, dtype=torch.float32)
    n_patches = horizon // model.chronos_config.output_patch_size
    if n_patches * model.chronos_config.output_patch_size != horizon:
        raise ValueError('This pilot uses whole output patches')
    encoded, loc_scale, _, _ = model.encode(context=x, num_output_patches=n_patches)
    normalized_target, _ = model.instance_norm(y, loc_scale)
    return dict(embeddings=encoded[0][:,-n_patches:].detach().clone(),
                loc_scale=tuple(t.detach().clone() for t in loc_scale),
                normalized_target=normalized_target.detach().clone(), target=y, context=x,
                horizon=horizon)


def cached_forward(model, cache):
    embedded = cache['embeddings']
    batch, patches = embedded.shape[:2]
    width, nquantiles = model.chronos_config.output_patch_size, model.num_quantiles
    raw = model.output_patch_embedding(embedded)
    normalized = raw.reshape(batch,patches,nquantiles,width).permute(0,2,1,3).reshape(batch,nquantiles,patches*width)
    unscaled = model.instance_norm.inverse(normalized.reshape(batch,-1),cache['loc_scale']).reshape_as(normalized)
    loss = masked_quantile_loss(normalized,cache['normalized_target'],model.quantiles)
    return loss,unscaled


def subset_cache(cache, indices):
    indices = torch.as_tensor(indices, device=cache['context'].device, dtype=torch.long)
    return {key:tuple(v.index_select(0,indices) for v in value) if key=='loc_scale'
            else value.index_select(0,indices) if isinstance(value,torch.Tensor) else value
            for key,value in cache.items()}


@torch.no_grad()
def validate_cache_equivalence(model, cache):
    """Check cached outputs and valid-point-scaled loss against native forward."""
    model.eval()
    loss,q = cached_forward(model,cache)
    native = model(context=cache['context'],future_target=cache['target'],num_output_patches=16)
    # Cached features use a four-cell batch, while a fold validation uses one.
    # Float32 XPU batch kernels can differ near zero in the extreme quantiles.
    # 1e-5 SOH = 0.001 percentage point; final test inference is always native.
    torch.testing.assert_close(q,native.quantile_preds,rtol=2e-5,atol=1e-5)
    individual=[]
    for i in range(len(cache['context'])):
        single=model(context=cache['context'][i:i+1],future_target=cache['target'][i:i+1],num_output_patches=16)
        individual.append(single.loss * 256 / torch.isfinite(cache['target'][i]).sum())
    torch.testing.assert_close(loss,torch.stack(individual).mean(),rtol=2e-5,atol=2e-6)
    return dict(cached_vs_native_max_abs=float((q-native.quantile_preds).abs().max()),
                masked_loss_native_parity=True)


@torch.no_grad()
def validation_scores(model, cache):
    model.eval()
    loss,q=cached_forward(model,cache)
    levels=model.quantiles
    indices=[int(torch.argmin(abs(levels-level/10))) for level in range(1,10)]
    deciles=q[:,indices]
    return dict(validation_loss=float(loss),
                validation_crossing_fraction=float((deciles[:,1:]<deciles[:,:-1]).any(dim=1).float().mean()),
                validation_invalid_interval_fraction=float((deciles[:,0]>deciles[:,-1]).float().mean()))
