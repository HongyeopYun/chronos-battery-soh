"""Small, explicit Chronos-2 parameter subsets and validation-only selection."""
import torch

from src.c2_fewshot_trainer import HEAD_PREFIX, target_loss_factor


def parameter_prefixes(model, mode):
    if mode not in ("head", "head_group"):
        raise ValueError(f"Unknown tuning mode: {mode}")
    prefixes = (HEAD_PREFIX,)
    if mode == "head_group":
        prefixes += (f"encoder.block.{len(model.encoder.block) - 1}.layer.1.",)
    return prefixes


def configure_subset(model, mode, training=True):
    prefixes = parameter_prefixes(model, mode)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
        parameter.grad = None
    # eval() does not disable gradients through frozen downstream layers.
    model.eval()
    if training:
        model.output_patch_embedding.train()
        if mode == "head_group":
            model.encoder.block[-1].layer[1].train()
    return [name for name, p in model.named_parameters() if p.requires_grad]


def model_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items() if key != "target_idx_ranges"}


def corrected_loss(model, batch):
    result = model(**batch)
    return result.loss * target_loss_factor(batch["future_target"]).to(result.loss.device)


@torch.no_grad()
def validation_scores(model, batches):
    """Equal SOH-window weights, including the final incomplete batch."""
    model.eval()
    levels = model.quantiles.detach().cpu()
    indices = [int(torch.argmin(abs(levels - q / 10))) for q in range(1, 10)]
    if not torch.allclose(levels[indices], torch.arange(1, 10) / 10, atol=1e-6):
        raise ValueError("Expected native decile quantiles")
    loss_sum = count = crossed = invalid = points = 0
    for batch in batches:
        active = torch.isfinite(batch["future_target"]).any(dim=-1)
        n = int(active.sum())
        output = model(**batch)
        loss = output.loss * target_loss_factor(batch["future_target"]).to(output.loss.device)
        q = output.quantile_preds[active][:, indices, :16]
        loss_sum += float(loss) * n
        count += n
        crossed += int((q[:, 1:] < q[:, :-1]).any(dim=1).sum())
        invalid += int((q[:, 0] > q[:, -1]).sum())
        points += n * 16
    return dict(validation_loss=loss_sum / count,
                validation_crossing_fraction=crossed / points,
                validation_invalid_interval_fraction=invalid / points)


def eligible_checkpoint(candidate, baseline):
    # Guard raw quantiles: no sorting, clipping, or using test-set metrics.
    return (candidate["validation_crossing_fraction"] <= baseline["validation_crossing_fraction"]
            and candidate["validation_invalid_interval_fraction"] <= baseline["validation_invalid_interval_fraction"])


def choose_checkpoint(candidates, baseline):
    eligible = [row for row in candidates if eligible_checkpoint(row, baseline)]
    if not eligible:
        raise ValueError("Include the step-zero baseline in checkpoint selection")
    return min(eligible, key=lambda row: (row["validation_loss"], row["step"], row["learning_rate"]))
