"""Controlled diagnostics; does not alter the original experiments or checkpoints."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import GenerationConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_experiment import (load_config, load_soh, make_windows, reload_adapter,
    fresh_pipeline, seed_everything, clear_device_cache, write_json, file_sha256,
    point_metrics, temporal_split, _training_loader, _optimizer, _train_epoch)


def distribution_metrics(y, samples):
    s = np.asarray(samples, dtype=np.float64)
    median = np.quantile(s, .5, axis=1)
    mean = s.mean(axis=1)
    order = np.arange(1, s.shape[1] + 1)[None, :, None]
    crps = np.abs(s - y[:, None, :]).mean(axis=1) - (
        np.sort(s, axis=1) * (2 * order - s.shape[1] - 1)).sum(axis=1) / s.shape[1] ** 2
    q10, q90 = np.quantile(s, [.1, .9], axis=1)
    return {"median_rmse": point_metrics(y, median)["rmse"],
            "median_mae": point_metrics(y, median)["mae"],
            "mean_rmse": point_metrics(y, mean)["rmse"],
            "crps": float(crps.mean()), "negative_rate": float((s < 0).mean()),
            "above2_rate": float((s > 2).mean()),
            "coverage80": float(((y >= q10) & (y <= q90)).mean()),
            "width80": float((q90 - q10).mean()), "minimum": float(s.min()), "maximum": float(s.max())}


def raw_predict(pipeline, x, cfg, seed=42, use_cache=True, route="peft", suppress=None):
    seed_everything(seed)
    model = pipeline.model.model
    model.eval()
    rows = []
    samples = []
    device = next(model.parameters()).device
    for start in range(0, len(x), cfg.inference_batch_size):
        context = torch.tensor(x[start:start + cfg.inference_batch_size], dtype=torch.float32)
        ids, mask, scale = pipeline.tokenizer.context_input_transform(context)
        gc = GenerationConfig(min_new_tokens=cfg.prediction_length, max_new_tokens=cfg.prediction_length,
            do_sample=True, num_return_sequences=cfg.num_samples, eos_token_id=pipeline.model.config.eos_token_id,
            pad_token_id=pipeline.model.config.pad_token_id, temperature=cfg.temperature,
            top_k=cfg.top_k, top_p=cfg.top_p, use_cache=use_cache, suppress_tokens=suppress)
        generator = model.base_model.model if route == "direct" else model
        with torch.inference_mode():
            generated = generator.generate(input_ids=ids.to(device), attention_mask=mask.to(device), generation_config=gc)
        tokens = generated[:, 1:].reshape(len(context), cfg.num_samples, cfg.prediction_length).cpu()
        values = pipeline.tokenizer.output_transform(tokens, scale).numpy()
        rows.append(tokens.numpy())
        samples.append(values)
    return np.concatenate(rows), np.concatenate(samples)


def token_diagnostics(tokens, samples, vocabulary_size=4096):
    bad = (samples < 0) | (samples > 2)
    unique, counts = np.unique(tokens[bad], return_counts=True)
    order = np.argsort(counts)[::-1]
    return {"all_token_min": int(tokens.min()), "all_token_max": int(tokens.max()),
            "outside_vocab_rate": float(((tokens < 0) | (tokens >= vocabulary_size)).mean()),
            "eos_rate": float((tokens == 1).mean()), "pad_rate": float((tokens == 0).mean()),
            "max_bin_rate": float((tokens == vocabulary_size - 1).mean()),
            "bad_by_horizon": bad.mean(axis=(0, 1)).tolist(),
            "top_bad_tokens": [{"token": int(unique[i]), "count": int(counts[i])} for i in order[:15]]}


def teacher_forced(pipeline, windows, cfg):
    model = pipeline.model.model
    model.eval()
    device = next(model.parameters()).device
    tokenizer = replace(pipeline.model.config, context_length=cfg.context_length,
                        prediction_length=cfg.prediction_length).create_tokenizer()
    all_stats = []
    for start in range(0, len(windows["context"]), 16):
        x = torch.tensor(windows["context"][start:start+16], dtype=torch.float32)
        y = torch.tensor(windows["target"][start:start+16], dtype=torch.float32)
        ids, mask, scale = tokenizer.context_input_transform(x)
        labels, _ = tokenizer.label_input_transform(y, scale)
        with torch.inference_mode():
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)).logits.float()
            p = logits.softmax(-1)
            nll = -p.gather(-1, labels.to(device)[..., None]).clamp_min(1e-30).log().squeeze(-1)
        for i in range(len(x)):
            all_stats.append({"forecast_ce": float(nll[i, :cfg.prediction_length].mean()),
                              "eos_ce": float(nll[i, -1]),
                              "forecast_eos_probability": float(p[i, :cfg.prediction_length, 1].mean()),
                              "forecast_maxbin_probability": float(p[i, :cfg.prediction_length, -1].mean()),
                              "vocab_size": logits.shape[-1]})
    return {k: float(np.mean([r[k] for r in all_stats])) for k in all_stats[0]}


def inference_controls(args):
    cfg = load_config(ROOT)
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    series = load_soh(ROOT, cfg)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "design.json", {"purpose": "Post-hoc causal diagnosis, not unbiased hyperparameter selection",
        "config": asdict(cfg), "original_run": str(args.run), "script_sha256": file_sha256(Path(__file__)),
        "controls": args.variant,
        "battery_scope": args.battery, "seed": 42, "window_limit": args.limit})
    rows, details = [], {}
    for battery in args.battery:
        fold = next(f for f in manifest["folds"] if f["test_battery"] == battery)
        windows = make_windows(series[battery], cfg.context_length, cfg.prediction_length)
        if args.limit:
            windows = {k: v[:args.limit] for k, v in windows.items()}
        pipeline = reload_adapter(cfg, Path(fold["adapter_path"]))
        original_tokens = None
        for variant in args.variant:
            begin = time.monotonic()
            if variant == "float32":
                pipeline.model.model.float()
            scope = pipeline.model.model.disable_adapter() if variant == "adapter_disabled" else nullcontext()
            with scope:
                tokens, samples = raw_predict(pipeline, windows["context"], cfg,
                    use_cache=variant != "no_cache", route="direct" if variant == "direct_generate" else "peft",
                    suppress=[4095] if variant == "suppress_maxbin" else None)
                tf = teacher_forced(pipeline, windows, cfg)
            row = {"battery": battery, "variant": variant, **distribution_metrics(windows["target"], samples),
                   "seconds": time.monotonic() - begin}
            if variant == "original":
                original_tokens = tokens.copy()
                with np.load(args.run / "predictions" / f"{battery}_train42_infer42.npz") as saved:
                    row["saved_samples_identical"] = bool(np.array_equal(samples, saved["samples"][:len(samples)]))
            if variant == "adapter_disabled":
                with np.load(Path(manifest["zeroshot_run"]) / "predictions" / f"{battery}_infer42.npz") as saved:
                    row["saved_samples_identical"] = bool(np.array_equal(samples, saved["samples"][:len(samples)]))
            if original_tokens is not None:
                row["token_agreement_original"] = float((tokens == original_tokens).mean())
            rows.append(row)
            details[f"{battery}_{variant}"] = {**token_diagnostics(tokens, samples), "teacher_forced": tf}
            np.savez_compressed(output / f"{battery}_{variant}.npz", **windows, tokens=tokens, samples=samples)
            pd.DataFrame(rows).to_csv(output / "inference_controls.csv", index=False)
            write_json(output / "token_diagnostics.json", details)
            print(json.dumps(row), flush=True)
        del pipeline
        clear_device_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "results/runs/20260923T063628Z_lora_2df6cb")
    parser.add_argument("--output", type=Path, default=ROOT / "results/diagnostics/20260929/inference")
    parser.add_argument("--battery", nargs="+", default=["B0005"])
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--variant", nargs="+", choices=["original", "adapter_disabled", "direct_generate", "float32", "no_cache", "suppress_maxbin"],
                        default=["original", "adapter_disabled", "direct_generate", "float32"])
    inference_controls(parser.parse_args())
