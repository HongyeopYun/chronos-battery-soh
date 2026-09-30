"""Evaluate frozen checkpoints with exact weight checks and measured replay drift.

Training, candidate scores, and selections remain immutable. Sample equality is
reported, while CRPS drift must remain below the original selection min_delta.
This alternative evaluation path records its own implementation hash.
"""
import argparse
import json
from pathlib import Path
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.soh_checkpoint as checkpoint
from src.soh_experiment import (
    file_sha256, write_json, load_soh, make_windows, reload_adapter, predict_samples,
    save_forecast, summarize_samples, metric_rows, clear_device_cache,
    verify_zeroshot, _write_tables, _paired_comparison, _finish_run,
)


def resume_verified(directory):
    directory = Path(directory).resolve()
    cfg, policy = checkpoint.load_checkpoint_config(ROOT)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    checkpoint._verify_identity(manifest, ROOT, cfg, policy)
    expected_trials = len(cfg.battery_ids) * len(cfg.train_seeds) * len(policy.learning_rates)
    cases = list((directory / "trials").glob("*/complete.json"))
    if len(cases) != expected_trials or not (directory / "selection_frozen.json").exists():
        raise ValueError("This entrypoint requires an already finished, frozen candidate search")
    if manifest["status"] == "complete":
        return checkpoint.verify_checkpoint_run(directory, ROOT, cfg, policy)
    for case in cases:
        checkpoint.verify_trial(case.parent)
    frozen = json.loads((directory / "selection_frozen.json").read_text(encoding="utf-8"))
    if frozen["folds"] != manifest["folds"]:
        raise ValueError("Manifest choices differ from frozen choices")
    baseline = Path(manifest["zeroshot_run"])
    verify_zeroshot(baseline, ROOT, cfg)
    series = load_soh(ROOT, cfg)
    checks = {"entrypoint_sha256": file_sha256(Path(__file__)), "checks": [],
              "note": "Exact loaded tensors; record sample differences; CRPS drift below original selection min_delta."}
    manifest.update(status="running", evaluation_entrypoint=str(Path(__file__).relative_to(ROOT)),
                    evaluation_entrypoint_sha256=checks["entrypoint_sha256"],
                    replay_acceptance={"adapter_tensors": "exact", "samples": "differences recorded",
                                       "max_abs_macro_crps_delta": policy.min_delta,
                                       "selection_changed": False})
    manifest.pop("error", None)
    write_json(directory / "manifest.json", manifest)

    def reload_verified(config, adapter):
        pipeline = reload_adapter(config, adapter)
        disk = load_file(str(Path(adapter) / "adapter_model.safetensors"))
        live = get_peft_model_state_dict(pipeline.model.model, save_embedding_layers=False)
        if set(disk) != set(live):
            raise ValueError("Loaded adapter tensor keys differ from saved checkpoint")
        for key in disk:
            torch.testing.assert_close(live[key].cpu(), disk[key], rtol=0, atol=0, check_dtype=True)
        checks["checks"].append({"adapter": str(adapter), "tensor_count": len(disk), "exact": True})
        write_json(directory / "loaded_adapter_checks.json", checks)
        print(f"Exact loaded adapter tensors: {Path(adapter).name} ({len(disk)})", flush=True)
        return pipeline

    rows, horizons, window_rows, distributions, replays = [], [], [], [], []
    try:
        for fold in frozen["folds"]:
            battery, seed = fold["test_battery"], fold["train_seed"]
            config = replace(cfg, learning_rate=fold["learning_rate"])
            pipeline = reload_verified(config, Path(fold["adapter_path"]))
            try:
                _, validation, _ = checkpoint._split_training(series, fold["train_batteries"], config)
                score, forecasts = checkpoint.validation_metrics(pipeline, validation, config, policy)
                case = directory / "trials" / Path(fold["adapter_path"]).name
                differences = []
                for name, (windows, samples) in forecasts.items():
                    with np.load(case / "selected_validation" / f"{name}.npz", allow_pickle=False) as saved:
                        for key in ("context", "target", "forecast_start"):
                            np.testing.assert_array_equal(windows[key], saved[key])
                        differences.append({"array": name, "values": samples.size,
                                            "different_values": int(np.count_nonzero(samples != saved["samples"])),
                                            "max_abs_sample_delta": float(np.abs(samples - saved["samples"]).max())})
                    save_forecast(directory / "replayed_validation" / f"{battery}_{name}.npz", windows, samples)
                crps_delta = score["macro"]["crps"] - fold["selected_validation"]["macro"]["crps"]
                if abs(crps_delta) >= policy.min_delta:
                    raise AssertionError(f"Replay CRPS drift exceeds original selection threshold: {crps_delta}")
                replays.append({"test_battery": battery, "train_seed": seed,
                                "selected_step": fold["selected_step"], "validation_arrays_checked": len(differences),
                                "validation_arrays_identical": sum(d["different_values"] == 0 for d in differences),
                                "macro_crps_delta": crps_delta, "sample_differences": differences,
                                "replayed_validation": score})
                write_json(directory / "checkpoint_replay.json", replays)
                print(f"REPLAY {battery}: CRPS delta={crps_delta:.10g}; different values="
                      f"{sum(d['different_values'] for d in differences)}", flush=True)
                windows = make_windows(series[battery], config.context_length, config.prediction_length)
                for inference_seed in config.inference_seeds:
                    samples = predict_samples(pipeline, windows["context"], config, inference_seed)
                    save_forecast(directory / "predictions" / f"{battery}_train{seed}_infer{inference_seed}.npz", windows, samples)
                    r, h, w = metric_rows(windows, summarize_samples(samples), "LoRA", battery, seed, inference_seed)
                    rows.extend(r); horizons.extend(h); window_rows.extend(w)
                    tags = {"battery": battery, "train_seed": seed, "inference_seed": inference_seed}
                    distributions.append({"model": "SelectedLoRA", **tags,
                                          **checkpoint.distribution_scores(windows["target"], samples)})
                    with np.load(baseline / "predictions" / f"{battery}_infer{inference_seed}.npz", allow_pickle=False) as zero:
                        for key in ("context", "target", "forecast_start"):
                            np.testing.assert_array_equal(zero[key], windows[key])
                        distributions.append({"model": "ZeroShot", **tags,
                                              **checkpoint.distribution_scores(windows["target"], zero["samples"])})
                    print(f"TEST {battery} infer={inference_seed} median_RMSE={r[0]['rmse']:.7f}", flush=True)
                _write_tables(directory, rows, horizons, window_rows)
                pd.DataFrame(distributions).to_csv(directory / "distribution_metrics.csv", index=False)
            finally:
                del pipeline
                clear_device_cache()
        _paired_comparison(directory, baseline)
        _finish_run(directory, manifest)
    except Exception as error:
        _finish_run(directory, manifest, error)
        raise
    return checkpoint.verify_checkpoint_run(directory, ROOT, cfg, policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    print(resume_verified(parser.parse_args().run), flush=True)
