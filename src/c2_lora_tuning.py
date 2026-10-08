"""Explicit time-attention / output-head LoRA with immutable foundation weights."""
import re
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import set_seed

MODES = ("time", "time_head")


def attach_lora(base, rank=8):
    targets = [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear) and
        (re.fullmatch(r"encoder\.block\.\d+\.layer\.0\.self_attention\.[qkvo]", n) or
         re.fullmatch(r"output_patch_embedding\.(hidden_layer|output_layer|residual_layer)", n))]
    if len(targets) != len(base.encoder.block)*4+3:
        raise ValueError("Unexpected C2 architecture")
    model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2*rank,
        lora_dropout=0., target_modules=targets, bias="none"))
    return model, targets


def reset_lora(model, seed):
    set_seed(seed)
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.reset_lora_parameters("default", init_lora_weights=True)
    model.eval()


def configure_lora(model, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown LoRA mode: {mode}")
    names = []
    for n, p in model.named_parameters():
        active = ".lora_" in n and (".layer.0.self_attention." in n or
            (mode == "time_head" and ".output_patch_embedding." in n))
        p.requires_grad_(active)
        p.grad = None
        if active:
            names.append(n)
    if not names:
        raise ValueError("No trainable adapters")
    # Gradients pass through frozen downstream layers. Only their dropout is disabled.
    model.eval()
    return names


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if ".lora_" in n}


def load_adapter_state(model, state):
    params = dict(model.named_parameters())
    expected = {n for n in params if ".lora_" in n}
    if set(state) != expected:
        raise ValueError("Adapter checkpoint keys differ")
    with torch.no_grad():
        for n, tensor in state.items():
            params[n].copy_(tensor)


def frozen_state_fingerprint(model):
    """Hash every foundation parameter, excluding newly added adapters."""
    import hashlib
    digest = hashlib.sha256()
    for n, p in model.named_parameters():
        if ".lora_" not in n:
            digest.update(n.encode())
            digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
