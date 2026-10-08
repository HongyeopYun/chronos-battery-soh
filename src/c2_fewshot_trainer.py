"""Keep the SOH loss scale equal when adding masked past-only sensor rows."""
import torch
from chronos.chronos2.trainer import Chronos2Trainer


def target_loss_factor(future_target):
    # Official C2 loss averages all variates, including zero-loss sensor rows.
    # Fixed complete SOH horizons allow a direct all-rows / SOH-rows correction.
    valid = torch.isfinite(future_target)
    active = valid.any(dim=-1)
    if not active.any() or not valid[active].all():
        raise ValueError("Expected complete SOH horizons and all-NaN covariate labels")
    return future_target.shape[0] / active.sum()


class TargetOnlyChronos2Trainer(Chronos2Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        factor = target_loss_factor(inputs["future_target"])
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        if return_outputs:
            loss, outputs = result
            return loss * factor.to(loss.device), outputs
        return result * factor.to(result.device)


HEAD_PREFIX = "output_patch_embedding."


def freeze_except_output_head(model):
    """Freeze the encoder and input embedding before the optimizer is constructed.

    C2 has an encoder and a residual output-patch head, not a T5 decoder.
    Pipeline.fit creates a fresh model, so freezing the source pipeline is not enough.
    """
    names = [name for name, _ in model.named_parameters() if name.startswith(HEAD_PREFIX)]
    if not names or not hasattr(model, "encoder"):
        raise ValueError("Expected a Chronos-2 encoder and output_patch_embedding")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(HEAD_PREFIX))
    return names


class HeadOnlyChronos2Trainer(TargetOnlyChronos2Trainer):
    def __init__(self, *args, model=None, **kwargs):
        if model is None:
            raise ValueError("Pass the copied Chronos-2 model by keyword")
        freeze_except_output_head(model)
        super().__init__(*args, model=model, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Trainer calls model.train() repeatedly. Keep frozen features deterministic;
        # the trainable output head retains its normal training/evaluation mode.
        for name, module in model.named_children():
            if name != "output_patch_embedding":
                module.eval()
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
