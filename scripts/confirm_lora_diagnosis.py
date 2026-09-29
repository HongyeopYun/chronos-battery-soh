"""Evaluate fixed diagnostic adapters across the original inference seeds.

No training or checkpoint selection takes place here. Seed 42 is reused from
the controlled experiment; seeds 43/44 check sampling variability only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_experiment import load_config, reload_adapter, clear_device_cache, write_json, file_sha256
from scripts.diagnose_lora import raw_predict, distribution_metrics


def run(args):
    cfg = load_config(ROOT)
    output = args.root / "seed_confirmation"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "design.json", {
        "purpose": "Fixed without-EOS-loss adapters; sampling robustness, not training-seed robustness",
        "training_seed": 42, "inference_seeds": [42, 43, 44],
        "script_sha256": file_sha256(Path(__file__)),
    })
    rows = []
    sources = {
        "zero": ROOT / "results/runs/20260923T063348Z_zeroshot_b57856/predictions",
        "original_lora": ROOT / "results/runs/20260923T063628Z_lora_2df6cb/predictions",
    }
    for battery in cfg.battery_ids:
        case = args.root / ("training" if battery == "B0005" else "confirm_eos") / f"{battery}_without_eos"
        assert (case / "complete.json").exists(), case
        with np.load(case / "final.npz") as data:
            context, target, start = data["context"], data["target"], data["forecast_start"]
            first_samples = data["samples"]
        pipeline = reload_adapter(cfg, case / "adapter")
        for seed in (42, 43, 44):
            for name, source in sources.items():
                pattern = f"{battery}_*infer{seed}.npz"
                paths = list(source.glob(pattern))
                assert len(paths) == 1, (source, pattern, paths)
                with np.load(paths[0]) as data:
                    assert np.array_equal(data["target"], target)
                    assert np.array_equal(data["context"], context)
                    rows.append({"model": name, "battery": battery, "inference_seed": seed,
                                 **distribution_metrics(target, data["samples"])})
            path = output / f"{battery}_without_eos_infer{seed}.npz"
            if seed == 42:
                samples = first_samples
            elif path.exists():
                with np.load(path) as data:
                    assert np.array_equal(data["target"], target)
                    samples = data["samples"]
            else:
                tokens, samples = raw_predict(pipeline, context, cfg, seed=seed)
                np.savez_compressed(path, context=context, target=target, forecast_start=start,
                                    tokens=tokens, samples=samples)
            row = {"model": "without_eos", "battery": battery, "inference_seed": seed,
                   **distribution_metrics(target, samples)}
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
            print(f"{battery} seed={seed} median={row['median_rmse']:.7f} "
                  f"mean={row['mean_rmse']:.7f} CRPS={row['crps']:.7f}", flush=True)
        del pipeline
        clear_device_cache()
    frame = pd.DataFrame(rows)
    metrics = list(distribution_metrics(target, samples))
    frame.groupby(["model", "battery"])[metrics].mean().to_csv(output / "battery_summary.csv")
    frame.groupby(["model", "inference_seed"])[metrics].mean().to_csv(output / "macro_by_seed.csv")
    frame.groupby("model")[metrics].mean().to_csv(output / "macro_summary.csv")
    print("SEED CONFIRMATION COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/diagnostics/20260929")
    run(parser.parse_args())
