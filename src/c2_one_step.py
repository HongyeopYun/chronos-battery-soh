"""Matched-context, single-target SOH forecasting and head-only adaptation."""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode
from transformers import set_seed

from src.battery_zeroshot import sha256, write_json
from src.c2_partial_tuning import configure_subset, model_batch, choose_checkpoint, eligible_checkpoint

CONTEXTS = (64, 32, 16, 8)
METHODS = ("LastValue", "Linear", "zero-shot", "head-only")


def crop_windows(windows, length):
    """Only remove old inputs; never change targets, identities, or forecast times."""
    if length not in CONTEXTS or windows["context"].shape[1] != 64 or windows["target"].shape[1] != 1:
        raise ValueError("Expected master 64-to-1 windows and a predefined context length")
    result = dict(windows)
    result["context"] = windows["context"][:, -length:]
    result["context_cycle"] = windows["context_cycle"][:, -length:]
    if (np.any(np.diff(result["context_cycle"], axis=1) != 1)
            or np.any(result["context_cycle"][:, -1] + 1 != result["target_cycle"][:, 0])):
        raise ValueError("Inputs must end before the single target observation")
    return result


def fixed_inputs(windows, indices):
    if windows["target"].shape[1] != 1:
        raise ValueError("Only one supervised target per task is allowed")
    return [dict(target=np.concatenate([windows["context"][i], windows["target"][i]]).astype(np.float32))
            for i in indices]


def make_dataset(inputs, length, training=False):
    if any(len(item["target"]) != length + 1 for item in inputs):
        raise ValueError("A fixed L+1 sequence is needed to prevent random target resampling")
    return Chronos2Dataset(inputs=inputs, context_length=length, prediction_length=1,
                          batch_size=4, output_patch_size=16, min_past=length,
                          mode=DatasetMode.TRAIN if training else DatasetMode.VALIDATION)


def one_step_loss(model, batch):
    """C2 pads one target to 16 internally; restore the per-observed-target scale."""
    target = batch["future_target"]
    if target.ndim != 2 or target.shape[1] != 1 or not torch.isfinite(target).all():
        raise ValueError("Expected one finite SOH label per row")
    output = model(**batch)
    # The official model averages over the padded patch width. Its other 15
    # outputs are masked from loss, rather than being fabricated SOH labels.
    return output.loss * output.quantile_preds.shape[-1], output


@torch.no_grad()
def validation_scores(model, batches):
    model.eval()
    levels = model.quantiles.detach().cpu()
    indices = [int(torch.argmin(abs(levels - q / 10))) for q in range(1, 10)]
    if not torch.allclose(levels[indices], torch.arange(1, 10) / 10, atol=1e-6):
        raise ValueError("Expected native deciles")
    loss_sum = count = crossed = invalid = 0
    for batch in batches:
        loss, output = one_step_loss(model, batch)
        n = batch["future_target"].shape[0]
        q = output.quantile_preds[:, indices, :1]
        loss_sum += float(loss) * n
        count += n
        crossed += int((q[:, 1:] < q[:, :-1]).any(dim=1).sum())
        invalid += int((q[:, 0] > q[:, -1]).sum())
    return dict(validation_loss=loss_sum / count, validation_crossing_fraction=crossed / count,
                validation_invalid_interval_fraction=invalid / count)


def restore_head(model, weights):
    parameters = dict(model.named_parameters())
    expected = {n for n in parameters if n.startswith("output_patch_embedding.")}
    if set(weights) != expected:
        raise ValueError("Checkpoint must contain exactly the output head")
    with torch.no_grad():
        for n, value in weights.items():
            parameters[n].copy_(value)
    model.eval()


def train_head(pipeline, initial, tasks, length, directory, weights_path, steps=100):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = pipeline.model
    restore_head(model, initial)
    names = configure_subset(model, "head")
    parameters = dict(model.named_parameters())
    frozen = {n: p._version for n, p in parameters.items() if n not in names}
    batches = [model_batch(b, model.device) for b in make_dataset(tasks[1], length)]
    baseline = dict(step=0, learning_rate=0., **validation_scores(model, batches), eligible=True)
    history, winner, winner_weights = [baseline], baseline, {n: v.clone() for n, v in initial.items()}
    start = time.perf_counter()
    for lr in (1e-6, 1e-5):
        restore_head(model, initial)
        set_seed(42)
        configure_subset(model, "head")
        optimizer = torch.optim.AdamW([parameters[n] for n in names], lr=lr, weight_decay=0., eps=1e-8)
        assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(parameters[n]) for n in names}
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1 - step / steps))
        iterator = iter(make_dataset(tasks[0], length, training=True))
        for step in range(1, steps + 1):
            batch = model_batch(next(iterator), model.device)
            assert batch["context"].shape == (4, length)
            assert batch["future_target"].shape == (4, 1)
            assert torch.isnan(batch["future_covariates"]).all()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = one_step_loss(model, batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameters[n] for n in names], 1.)
            optimizer.step()
            scheduler.step()
            if step % 20 == 0 or step == steps:
                row = dict(step=step, learning_rate=lr, **validation_scores(model, batches))
                row["eligible"] = eligible_checkpoint(row, baseline)
                history.append(row)
                if choose_checkpoint(history, baseline) is row:
                    winner = row
                    winner_weights = {n: parameters[n].detach().cpu().clone() for n in names}
                pd.DataFrame(history).to_csv(directory / "validation_history.csv", index=False)
                configure_subset(model, "head")
        if any(parameters[n]._version != version for n, version in frozen.items()):
            raise ValueError("Frozen backbone was changed")
        if any(parameters[n].grad is not None for n in frozen):
            raise ValueError("Frozen backbone received gradients")
        del optimizer, scheduler, iterator
    assert winner == choose_checkpoint(history, baseline)
    restore_head(model, initial)
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(winner_weights, weights_path)
    result = dict(**winner, baseline=baseline, context_length=length, prediction_length=1,
                  train_windows=len(tasks[0]), validation_windows=len(tasks[1]),
                  trainable_parameters=sum(parameters[n].numel() for n in names),
                  total_parameters=sum(p.numel() for p in parameters.values()),
                  trainable_names=names, frozen_backbone_verified=True,
                  selected_weights=str(weights_path), selected_weights_sha256=sha256(weights_path),
                  elapsed_seconds=time.perf_counter() - start)
    write_json(directory / "selected.json", result)
    return result


def baselines(context):
    context = np.asarray(context, dtype=float)
    if context.ndim != 2 or np.isinf(context).any():
        raise ValueError('Expected finite or missing history in a 2D array')
    finite = np.isfinite(context)
    if (finite.sum(axis=1) < 2).any():
        raise ValueError('Linear baseline needs at least two observed positions')
    if not finite.all():
        # Preserve the actual positions: fitting [1, NaN, 3] means t=[0,2],
        # not a compressed two-point history. LastValue uses the last valid Q.
        x = np.broadcast_to(np.arange(context.shape[1], dtype=float), context.shape)
        count = finite.sum(axis=1)
        xmean = np.where(finite, x, 0.).sum(axis=1)/count
        ymean = np.nansum(context, axis=1)/count
        dx = np.where(finite, x-xmean[:, None], 0.)
        dy = np.where(finite, context-ymean[:, None], 0.)
        slope = (dx*dy).sum(axis=1)/(dx*dx).sum(axis=1)
        last = np.max(np.where(finite, x, -1), axis=1).astype(int)
        return {'LastValue': context[np.arange(len(context)), last],
                'Linear': ymean+slope*(context.shape[1]-xmean)}
    x = np.arange(context.shape[1], dtype=float)
    centered = x - x.mean()
    slope = (context * centered).sum(axis=1) / np.square(centered).sum()
    return {"LastValue": context[:, -1], "Linear": context.mean(axis=1) + slope * (len(x) - x.mean())}


def point_metrics(target, point):
    y, p = np.asarray(target, dtype=float).reshape(-1), np.asarray(point, dtype=float).reshape(-1)
    if y.shape != p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Invalid one-step point arrays")
    error = p - y
    sst = np.square(y - y.mean()).sum()
    return dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.square(error).mean())),
                bias=float(error.mean()), temporal_r2=float(1 - np.square(error).sum() / sst)
                if len(y) > 1 and sst > 0 else float("nan"))


def probability_metrics(target, quantiles):
    y, q = np.asarray(target, dtype=float).reshape(-1), np.asarray(quantiles, dtype=float)
    if q.shape != (len(y), 9) or not np.isfinite(q).all():
        raise ValueError("Expected nine finite quantiles per one-step prediction")
    levels = np.arange(1, 10) / 10
    error = y[:, None] - q
    denominator = np.abs(y).sum()
    invalid = q[:, 0] > q[:, -1]
    return dict(wql=float(2 * np.maximum(levels * error, (levels - 1) * error).sum(axis=0).mean() / denominator)
                if denominator else float("nan"),
                picp80=float(((q[:, 0] <= y) & (y <= q[:, -1])).mean()) if not invalid.any() else float("nan"),
                width80=float((q[:, -1] - q[:, 0]).mean()) if not invalid.any() else float("nan"),
                invalid_interval_fraction=float(invalid.mean()),
                quantile_crossing_fraction=float((np.diff(q, axis=1) < 0).any(axis=1).mean()))
