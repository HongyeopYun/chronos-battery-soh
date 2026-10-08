"""Re-infer saved zero-shot settings: C2 all windows, T5 phase representatives.

No fitting. Original reports are unchanged. Store compact window R2 values so
distribution summaries do not require another model run in the future.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, sha256, write_json
from src.battery_model_zeroshot import load_model, predict_quantiles, phase_labels, PHASES, quantile_scores
from src.soh_experiment import predict_samples, clear_device_cache, resolve_device
from src.soh_evaluation import window_r2_values
from scripts.report_r2_distribution import GROUPS, OUT


def location(tag, group):
    dataset = group.replace("_legacy32", "")
    if dataset == "calce_cs2_1c" and not group.endswith("legacy32"):
        return ROOT / f"results/{tag}-battery/calce64-16/{dataset}", f"{tag}-battery_"
    return ROOT / f"results/{tag}-battery" / (f"full/{dataset}" if tag == "c2" else dataset), "c2-battery_" if tag == "c2" else ""


def recover(tag, group, pipeline):
    directory, prefix = location(tag, group)
    manifest = json.loads((directory / f"{prefix}run.json").read_text(encoding="utf-8"))
    cfg = ZeroShotConfig(**manifest["config"])
    output = OUT / "windows" / f"{tag}_{group}.csv.gz"
    meta_path = output.with_suffix(".json")
    scope = "all_windows" if tag == "c2" else "representative_windows_all_metrics"
    if output.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("status") == "complete":
            if old["config"] != manifest["config"] or old["scope"] != scope or old["bundle_fingerprint"] != manifest["bundle_fingerprint"]:
                raise ValueError("Recovery output has incompatible configuration")
            print(f"SKIP {tag}/{group}", flush=True)
            return
    if sha256(Path(manifest["bundle"]) / "manifest.json") != manifest["bundle_fingerprint"]:
        raise ValueError("Input bundle differs from original zero-shot run")
    records, _, data, _ = prepare_battery_windows(manifest["bundle"], cfg)
    selected = []
    for record in records:
        b, w = record["battery_id"], record["windows"]
        cycles = data[(data.battery_id == b) & data.is_observed].cycle.to_numpy()
        phases = phase_labels(w["forecast_start"], cycles)
        if tag == "t5":
            centers = cycles.min() + (np.arange(3) + .5) * (cycles.max() - cycles.min() + 1) / 3
            indices = []
            for phase, center in zip(PHASES, centers):
                possible = np.flatnonzero(phases == phase)
                if len(possible):
                    indices.append(possible[np.abs(w["forecast_start"][possible] - center).argmin()])
            indices = np.asarray(indices, dtype=int)
            w = {key: value[indices] for key, value in w.items()}
            phases = phases[indices]
        selected.append((b, w, phases))
    meta = dict(status="running", model=cfg.model_id, config=manifest["config"], scope=scope,
                bundle_fingerprint=manifest["bundle_fingerprint"], source=str(directory.relative_to(ROOT)),
                started_at=datetime.now(timezone.utc).isoformat(), completed_batteries=0,
                n_batteries=len(selected), n_windows=sum(len(w['target']) for _, w, _ in selected))
    write_json(meta_path, meta)
    print(f"START {tag}/{group}: {meta['n_batteries']} cells, {meta['n_windows']} windows, device={resolve_device(cfg.device)}", flush=True)
    original_metrics = pd.read_csv(directory / f"{prefix}battery_metrics.csv")
    source_scores = original_metrics[original_metrics.model.isin(["Chronos", "c2-battery", "t5-battery"])]
    rows, metrics = [], []
    quantiles = None
    tick = time.perf_counter()
    if tag == "t5":
        contexts = np.concatenate([w["context"] for _, w, _ in selected])
        samples = predict_samples(pipeline, contexts, cfg, cfg.inference_seeds[0])
        quantiles = np.quantile(samples.astype(np.float64), np.arange(1, 10) / 10, axis=1).transpose(1, 2, 0)
        del samples
    cursor = 0
    for b, w, phases in selected:
        n = len(w["target"])
        if tag == "c2":
            _, q = predict_quantiles(pipeline, w["context"], cfg)
        else:
            q = quantiles[cursor:cursor+n]
            cursor += n
        values = window_r2_values(w["target"], q[..., 4])
        rows.extend(dict(battery_id=b, seed=cfg.inference_seeds[0], forecast_start=int(start), phase=phase, r2=value)
                    for start, phase, value in zip(w["forecast_start"], phases, values))
        for phase in ["all", *PHASES]:
            mask = np.ones(n, dtype=bool) if phase == "all" else phases == phase
            if not mask.any():
                continue
            scores = quantile_scores(w["target"][mask], q[mask, :, 4], q[mask])
            metrics.append(dict(battery_id=b, phase=phase, n_windows=int(mask.sum()), **scores))
            if tag == "c2":
                old = source_scores[(source_scores.battery_id == b) & (source_scores.phase == phase)].iloc[0]
                np.testing.assert_allclose(scores["rmse"], old.rmse, rtol=2e-5, atol=5e-8)
                np.testing.assert_allclose(scores["r2"], old.r2, rtol=2e-5, atol=2e-6)
        meta["completed_batteries"] += 1
        write_json(meta_path, meta)
        if tag == "c2" and (meta["completed_batteries"] % 5 == 0 or meta["completed_batteries"] == len(selected)):
            print(f"PROGRESS {tag}/{group}: {meta['completed_batteries']}/{len(selected)} elapsed={time.perf_counter()-tick:.1f}s", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False, compression="gzip")
    summary = pd.DataFrame(metrics).groupby("phase", sort=False, as_index=False)[["rmse", "r2", "wql", "picp80"]].mean()
    meta.update(status="complete", summary=summary.to_dict("records"),
                original_metrics_cross_checked=tag == "c2", elapsed_seconds=time.perf_counter()-tick,
                finished_at=datetime.now(timezone.utc).isoformat())
    write_json(meta_path, meta)
    print(f"DONE {tag}/{group}: {len(frame)} windows, {meta['elapsed_seconds']:.1f}s", flush=True)


def main():
    (OUT / "windows").mkdir(parents=True, exist_ok=True)
    status = OUT / "recovery_status.json"
    state = dict(status="running", mode="C2 full / T5 representative", completed=[])
    write_json(status, state)
    pipeline = None
    try:
        for tag in ["c2", "t5"]:
            directory, prefix = location(tag, GROUPS[0])
            manifest = json.loads((directory / f"{prefix}run.json").read_text(encoding="utf-8"))
            pipeline = load_model(ZeroShotConfig(**manifest["config"]))
            for group in GROUPS + ["calce_cs2_1c_legacy32"]:
                state["current"] = f"{tag}/{group}"
                write_json(status, state)
                recover(tag, group, pipeline)
                state["completed"].append(state["current"])
                write_json(status, state)
            del pipeline
            pipeline = None
            clear_device_cache()
        from scripts.report_r2_distribution import main as report
        report()
        state.update(status="complete", current=None)
    except Exception as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(status, state)


if __name__ == "__main__":
    main()
