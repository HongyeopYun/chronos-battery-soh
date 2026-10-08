"""Inference-only timing on the presentation's 90 fixed XJTU 2C windows."""
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
import torch
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows
from src.battery_model_zeroshot import load_model
from src.battery_fewshot import history_bank, isolated_inputs
from src.nasa_covariates import FEATURES
from src.soh_evaluation import window_r2_values

OUT = Path(__file__).resolve().parent
G = ROOT / "results/c2-battery"
old = json.loads((G / "covariates/xjtu_2c/c2-battery_xjtu_2c_vi_run.json").read_text())
original = json.loads((G / "full/xjtu_2c/c2-battery_run.json").read_text())
config = ZeroShotConfig(**old["config"])
records, _, data, _ = prepare_battery_windows(original["bundle"], config)
features = pd.read_csv(G / "covariates/xjtu_2c/c2-battery_xjtu_2c_vi_cycle_features.csv")
support = pd.read_csv(G / "fewshot/xjtu_2c/c2-battery_xjtu_2c_fewshot_support_features.csv")
banks = [history_bank(data[data.battery_id == b], support[support.battery_id == b])
         for b in ["batch1_2C_battery-4", "batch1_2C_battery-5"]]
inputs = {k: [] for k in ("SOH", "VI", "cross2")}
targets = []
with np.load(G / "fewshot/xjtu_2c/c2-battery_xjtu_2c_fewshot_predictions.npz") as cache:
    for record in records:
        b, w = record["battery_id"], record["windows"]
        indices = [int(np.flatnonzero(w["forecast_start"] == s)[0]) for s in cache[f"{b}_forecast_start"]]
        np.testing.assert_array_equal(w["target"][indices], cache[f"{b}_target"])
        for i in indices:
            args = (b, w["context"][i], w["context_cycle"][i], int(w["forecast_start"][i]), features[features.battery_id == b])
            for key, refs, names in [("SOH", [], ()), ("VI", [], FEATURES), ("cross2", banks, ())]:
                tasks, _ = isolated_inputs(*args, refs, names)
                inputs[key].append(tasks)
            targets.append(w["target"][i])
targets = np.asarray(targets)
assert targets.shape == (90, 16)
pipeline = load_model(config)
device = pipeline.model.device
def sync():
    if device.type == "xpu": torch.xpu.synchronize()
    elif device.type == "cuda": torch.cuda.synchronize()

def predict(tasks, cross=False):
    q, _ = pipeline.predict_quantiles(inputs=tasks, prediction_length=16, context_length=64,
        quantile_levels=[i/10 for i in range(1, 10)],
        batch_size=sum(1 + len(t.get("past_covariates", {})) for t in tasks), cross_learning=cross)
    return q[0].detach().cpu().numpy()[0]

results = []
def measure(label, key, cross=False):
    pipeline.model.eval()
    with torch.inference_mode():
        for tasks in inputs[key][:5]: predict(tasks, cross)
        sync()
        times, predictions = [], []
        for tasks in inputs[key]:
            sync()
            tick = time.perf_counter()
            q = predict(tasks, cross)
            sync()
            times.append(time.perf_counter() - tick)
            predictions.append(q[:, 4])
    r2 = window_r2_values(targets, np.asarray(predictions))
    row = dict(method=label, windows=90, total_seconds=sum(times),
               mean_ms_per_window=1000*np.mean(times), median_ms_per_window=1000*np.median(times),
               r2_mean=float(np.mean(r2)), r2_median=float(np.median(r2)))
    results.append(row)
    print(json.dumps(row), flush=True)

measure("C2 zero-shot", "SOH")
measure("SOH + V + I", "VI")
measure("Cross-learning 2-shot", "cross2", True)
head = ROOT / "models/c2-fewshot-partial/xjtu_2c/SOH_head.pt"
weights = torch.load(head, map_location="cpu", weights_only=True)
parameters = dict(pipeline.model.named_parameters())
with torch.no_grad():
    for name, value in weights.items(): parameters[name].copy_(value)
measure("Head-only", "SOH")
del parameters, weights, pipeline
if device.type == "xpu": torch.xpu.empty_cache()
from chronos import Chronos2Pipeline
pipeline = Chronos2Pipeline.from_pretrained(str(ROOT / "models/c2-fewshot-full/xjtu_2c/SOH/finetuned-ckpt"),
    device_map=str(device), dtype=torch.float32, local_files_only=True)
measure("Full tuning", "SOH")
report = dict(measured_at=datetime.now(timezone.utc).isoformat(), device=str(device),
              device_name=torch.xpu.get_device_name() if device.type == "xpu" else str(device),
              context=64, horizon=16, dtype="float32", warmup_calls=5,
              protocol="One forecast origin per call; all 90 presentation windows once; device synchronized. Model loading, feature extraction, input preparation, training and scoring excluded.",
              results=results)
(OUT / "timing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
pd.DataFrame(results).to_csv(OUT / "timing.csv", index=False)
print("TIMING_COMPLETE", flush=True)
