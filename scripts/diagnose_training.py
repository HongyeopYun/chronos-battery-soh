"""One-factor LoRA ablations with original seeds, windows, and update budgets.

These are post-hoc explanatory experiments, not a new unbiased model-selection
benchmark. Existing checkpoints and experiment outputs are left untouched.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_experiment import (load_config, load_soh, make_windows, temporal_split,
    fresh_pipeline, seed_everything, clear_device_cache, write_json, file_sha256,
    _training_loader, _optimizer)
from scripts.diagnose_lora import raw_predict, distribution_metrics, teacher_forced, token_diagnostics


@contextmanager
def preserve_rng():
    cpu = torch.get_rng_state()
    xpu = torch.xpu.get_rng_state_all() if torch.xpu.is_available() else None
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    python, numpy = random.getstate(), np.random.get_state()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if xpu is not None:
            torch.xpu.set_rng_state_all(xpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        random.setstate(python)
        np.random.set_state(numpy)


def gradient_parts(pipeline, batch, horizon):
    model = pipeline.model.model
    model.eval()
    device = next(model.parameters()).device
    ids, mask, labels = [t.to(device) for t in batch]
    output = model(input_ids=ids, attention_mask=mask, labels=labels)
    ce = F.cross_entropy(output.logits.float().reshape(-1, output.logits.shape[-1]), labels.flatten(), reduction="none")
    ce = ce.reshape(len(ids), -1)
    forecast = ce[:, :horizon].sum(dim=1).mean() / (horizon + 1)
    eos = ce[:, horizon].mean() / (horizon + 1)
    parameters = [p for p in model.parameters() if p.requires_grad]
    gf = torch.autograd.grad(forecast, parameters, retain_graph=True)
    ge = torch.autograd.grad(eos, parameters)
    f2 = sum(g.float().square().sum() for g in gf)
    e2 = sum(g.float().square().sum() for g in ge)
    dot = sum((a.float() * b.float()).sum() for a, b in zip(gf, ge))
    return {"forecast_ce": float(ce[:, :horizon].mean()), "eos_ce": float(ce[:, horizon].mean()),
            "weighted_forecast_gradient_norm": float(f2.sqrt()), "weighted_eos_gradient_norm": float(e2.sqrt()),
            "gradient_cosine": float(dot / (f2.sqrt() * e2.sqrt()).clamp_min(1e-30))}


def run(args):
    cfg = load_config(ROOT)
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    series = load_soh(ROOT, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "design.json", {
        "purpose": "Post-hoc causal diagnosis; test outcomes do not select checkpoints",
        "batteries": args.battery, "variants": args.variant, "config": asdict(cfg),
        "original_run": str(args.run), "seed": 42, "script_sha256": file_sha256(Path(__file__)),
        "checkpoint_rule": "Final original selected epoch; baseline also checked at selection-equivalent updates",
        "without_eos_rule": "Mask final label only, preserve encoder EOS and decoder conditioning, multiply forecast-only loss by 16/17 to preserve per-token weight",
        "lower_lr_rule": "Only learning rate changes from 1e-4 to 3e-5",
        "selection_prefix_rule": "Only dataset changes to original temporal training prefixes; same selected epochs",
        "all_train_tail_scope": "Tail windows are in-sample after full-data refit; not an independent validation score",
    })
    rows, detail = [], {}
    for battery in args.battery:
        fold = next(f for f in manifest["folds"] if f["test_battery"] == battery)
        train_batteries = fold["train_batteries"]
        epochs = fold["selected_epoch"]
        full = [make_windows(series[b], cfg.context_length, cfg.prediction_length) for b in train_batteries]
        split = {b: temporal_split(series[b], cfg) for b in train_batteries}
        prefix = [split[b][0] for b in train_batteries]
        matched_steps = math.ceil(sum(len(d["target"]) for d in prefix) / cfg.batch_size) * epochs
        test = make_windows(series[battery], cfg.context_length, cfg.prediction_length)
        for variant in args.variant:
            case_dir = args.output / f"{battery}_{variant}"
            if (case_dir / "complete.json").exists():
                print(f"Already completed: {case_dir}", flush=True)
                continue
            case_dir.mkdir(parents=True, exist_ok=True)
            local_cfg = replace(cfg, learning_rate=3e-5) if variant == "lower_lr" else cfg
            datasets = prefix if variant == "selection_prefix" else full
            pipeline = fresh_pipeline(local_cfg, 42, lora=True)
            loader = _training_loader(pipeline, datasets, local_cfg, 42)
            optimizer = _optimizer(pipeline, local_cfg)
            model = pipeline.model.model
            device = next(model.parameters()).device
            parameters = [p for p in model.parameters() if p.requires_grad]
            fixed_batch = tuple(t[:16] for t in loader.dataset.tensors)
            with preserve_rng():
                before = gradient_parts(pipeline, fixed_batch, cfg.prediction_length)
            detail[f"{battery}_{variant}_initial_gradient"] = before
            write_json(args.output / "details.json", detail)
            print(f"START {battery} {variant}: epochs={epochs}, batches={len(loader)}, matched_steps={matched_steps}", flush=True)

            def evaluate(label, steps):
                with preserve_rng():
                    tokens, samples = raw_predict(pipeline, test["context"], cfg, seed=42)
                    row = {"battery": battery, "variant": variant, "checkpoint": label, "steps": steps,
                           **distribution_metrics(test["target"], samples)}
                    if variant == "baseline" and label == "final":
                        with np.load(args.run / "predictions" / f"{battery}_train42_infer42.npz") as previous:
                            row["original_samples_identical"] = bool(np.array_equal(samples, previous["samples"]))
                    rows.append(row)
                    tf = teacher_forced(pipeline, test, cfg)
                    token_info = token_diagnostics(tokens, samples)
                    detail[f"{battery}_{variant}_{label}"] = {"teacher_forced_test": tf, "tokens": token_info}
                    np.savez_compressed(case_dir / f"{label}.npz", **test, tokens=tokens, samples=samples)
                    pd.DataFrame(rows).to_csv(args.output / "training_controls.csv", index=False)
                    write_json(args.output / "details.json", detail)
                    print(json.dumps(row), flush=True)
                model.train()

            steps, losses = 0, []
            for epoch in range(1, epochs + 1):
                seed_everything(42 + epoch)
                model.train()
                total, count = 0., 0
                for ids, mask, labels in loader:
                    ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                    if variant == "without_eos":
                        labels = labels.clone()
                        assert (labels[:, -1] == 1).all()
                        labels[:, -1] = -100
                    optimizer.zero_grad(set_to_none=True)
                    output = model(input_ids=ids, attention_mask=mask, labels=labels)
                    loss = output.loss
                    if variant == "without_eos":
                        loss = loss * cfg.prediction_length / (cfg.prediction_length + 1)
                    assert torch.isfinite(loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    steps += 1
                    total += float(loss) * len(ids)
                    count += len(ids)
                    if variant == "baseline" and steps == matched_steps:
                        evaluate("matched_updates", steps)
                losses.append(total / count)
                print(f"  {battery} {variant} epoch={epoch}/{epochs} loss={losses[-1]:.5f} steps={steps}", flush=True)
            evaluate("final", steps)
            with preserve_rng():
                detail[f"{battery}_{variant}_final_gradient"] = gradient_parts(pipeline, fixed_batch, cfg.prediction_length)
            # Distribution and forecast/EOS loss on the training-battery tails.
            tail_rows = []
            for b in train_batteries:
                tail = split[b][1]
                with preserve_rng():
                    _, samples = raw_predict(pipeline, tail["context"], replace(cfg, num_samples=32), seed=31415)
                    tf = teacher_forced(pipeline, tail, cfg)
                tail_rows.append({"battery": b, **distribution_metrics(tail["target"], samples), **tf})
            pd.DataFrame(tail_rows).to_csv(case_dir / "train_battery_tails.csv", index=False)
            model.save_pretrained(case_dir / "adapter", save_embedding_layers=False)
            write_json(case_dir / "complete.json", {"variant": variant, "test_battery": battery,
                       "train_batteries": train_batteries, "epochs": epochs, "steps": steps,
                       "losses": losses, "learning_rate": local_cfg.learning_rate,
                       "test_metrics": rows[-1], "final_gradient": detail[f"{battery}_{variant}_final_gradient"]})
            write_json(args.output / "details.json", detail)
            del optimizer, loader, model, pipeline, parameters, output, loss
            clear_device_cache()
    print("TRAINING DIAGNOSTICS COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "results/runs/20260923T063628Z_lora_2df6cb")
    parser.add_argument("--output", type=Path, default=ROOT / "results/diagnostics/20260929/training")
    parser.add_argument("--battery", nargs="+", default=["B0005"])
    parser.add_argument("--variant", nargs="+", choices=["baseline", "lower_lr", "without_eos", "selection_prefix"],
                        default=["baseline", "lower_lr", "without_eos", "selection_prefix"])
    run(parser.parse_args())
