"""Validation-only, one-factor LoRA ablation; test forecasts are a separate stage."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from . import soh_experiment as e


def replay_selection(history, config):
    """Replay a stopping policy without consulting any outer-test observations."""
    baseline = history[0]["validation"]
    best, best_epoch, stale, stopped = baseline, 0, 0, 0
    for row in history[1:]:
        if row["epoch"] > config.max_epochs:
            break
        stopped = row["epoch"]
        if e.validation_candidate_improves(row["validation"], best, baseline, config):
            best, best_epoch, stale = row["validation"], row["epoch"], 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    return {"selected_epoch": best_epoch, "selected_validation": best, "stopped_epoch": stopped}


def choose_candidate(records, min_delta):
    """Stable order is the predeclared tie-break, including all-epoch-zero folds."""
    winner = next(iter(records))
    for name, record in records.items():
        if record["selected_validation"]["crps"] < records[winner]["selected_validation"]["crps"] - min_delta:
            winner = name
    return winner


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_configs(base, candidates):
    """Vary training factors while preserving the shared evaluation/selection protocol."""
    if base.refit_full_data or base.selection_metric != "guarded_crps":
        raise ValueError("Ablation requires guarded_crps and validated checkpoints")
    allowed = {"learning_rate", "kl_weight", "patience", "max_epochs",
               "batch_size", "weight_decay", "lora_rank", "lora_alpha", "lora_dropout",
               "battery_balance"}
    result = {}
    for item in candidates:
        name, overrides = item["name"], item["overrides"]
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("Candidate names must be unique and nonempty")
        unsupported = set(overrides) - allowed
        if unsupported:
            raise ValueError(f"Candidate {name} cannot override shared protocol fields: {sorted(unsupported)}")
        result[name] = replace(base, **overrides)
    if not result:
        raise ValueError("Candidate names must be unique and nonempty")
    return result


def _selected_training(root, directory, series, batteries, config, seed, label):
    output = directory / "selection" / f"{label}.json"
    adapter = root / "models" / directory.name / label
    receipt = directory / "receipts" / f"{label}.json"
    expected = json.loads(json.dumps({"config": asdict(config), "train_batteries": batteries, "seed": seed}))
    if receipt.exists():
        saved = _read(receipt)
        if saved["request"] != expected or saved["selection_sha256"] != e.file_sha256(output):
            raise ValueError(f"Changed selection request or artifact: {label}")
        for name, digest in saved["adapter_sha256"].items():
            if e.file_sha256(adapter / name) != digest:
                raise ValueError(f"Changed adapter: {label}/{name}")
        return _read(output), adapter
    e.select_epochs(series, batteries, config, seed, output, checkpoint=adapter)
    record = _read(output)
    if record["status"] != "complete":
        raise ValueError(f"Incomplete selection: {label}")
    e.write_json(receipt, {"request": expected, "selection_sha256": e.file_sha256(output),
                           "adapter_sha256": {p.name: e.file_sha256(p) for p in adapter.iterdir() if p.is_file()}})
    return record, adapter


def run_ablation(root: Path, protocol_path: Path, resume: Path | None = None):
    root = root.resolve()
    protocol = _read(root / protocol_path)
    base = e.load_config(root, protocol["base_config"])
    candidates = candidate_configs(base, protocol["candidates"])
    search_seed = protocol["search_train_seed"]
    if search_seed not in base.train_seeds:
        raise ValueError("Search seed must be included in final training seeds")
    frozen = {"protocol": protocol, "base_config": asdict(base),
              "candidate_configs": {name: asdict(config) for name, config in candidates.items()},
              "implementation_sha256": e.file_sha256(Path(__file__)),
              "core_sha256": e.file_sha256(root / "src" / "soh_experiment.py")}
    frozen = json.loads(json.dumps(frozen))
    if resume:
        directory = e.resolve_reference(root, resume)
        manifest = _read(directory / "manifest.json")
        if manifest["kind"] != "lora_ablation":
            raise ValueError("Resume requires a lora_ablation run")
        if _read(directory / "protocol.json") != frozen:
            raise ValueError("Protocol/code changed; start a new run instead of resuming")
        if manifest["evaluation_signature"] != e.evaluation_signature(root, base):
            raise ValueError("Data/runtime changed; cannot resume")
        if manifest["status"] == "complete":
            e.audit_saved_run(directory, root, verify_adapters=True)
            return directory
        baseline = e.resolve_reference(root, manifest["zeroshot_run"])
    else:
        baseline = e.find_zeroshot(root, base)
        if baseline is None:
            baseline = e.run_zeroshot(root, base)
        directory, manifest = e._start_run(root, base, "lora_ablation")
        manifest["zeroshot_run"] = baseline.relative_to(root).as_posix()
        e.write_json(directory / "protocol.json", frozen)
    e.verify_zeroshot(baseline, root, base)
    manifest["status"] = "running"
    e.write_json(directory / "manifest.json", manifest)
    print(f"ABLATION_RUN={directory}", flush=True)
    series = e.load_soh(root, base)
    pipeline = None
    try:
        choices, search = {}, {}
        for heldout in base.battery_ids:
            batteries = [b for b in base.battery_ids if b != heldout]
            # Pass only training series to selection, even though the loader read all data.
            training_series = {b: series[b] for b in batteries}
            records = {}
            for name, config in candidates.items():
                print(f"SEARCH heldout={heldout} candidate={name} seed={search_seed}", flush=True)
                record, _ = _selected_training(root, directory, training_series, batteries, config,
                                               search_seed, f"{heldout}_{name}_train{search_seed}")
                records[name] = record
            choices[heldout] = choose_candidate(records, base.min_delta)
            search[heldout] = {
                name: {"selected_epoch": r["selected_epoch"], "selected_validation": r["selected_validation"],
                       "baseline_validation": r["baseline_validation"],
                       "short_policy": replay_selection(r["history"], replace(candidates[name], patience=base.patience)),
                       "median_only_best": min(r["history"], key=lambda row: row["validation"]["median_rmse"])}
                for name, r in records.items()}
            print(f"CHOSEN heldout={heldout} candidate={choices[heldout]}", flush=True)
        # Persist all choices before creating or reading any held-out forecast.
        decisions = {"choices": choices, "validation_search": search,
                     "rule": protocol["candidate_selection"], "test_results_used": False}
        choice_path = directory / "choices.json"
        if choice_path.exists() and _read(choice_path) != decisions:
            raise ValueError("Frozen choices changed during resume")
        e.write_json(choice_path, decisions)
        manifest["choices_sha256"] = e.file_sha256(choice_path)
        e.write_json(directory / "manifest.json", manifest)
        folds = []
        for heldout, name in choices.items():
            config = candidates[name]
            batteries = [b for b in base.battery_ids if b != heldout]
            for seed in base.train_seeds:
                print(f"REPEAT heldout={heldout} candidate={name} seed={seed}", flush=True)
                record, adapter = _selected_training(root, directory, {b: series[b] for b in batteries},
                                                     batteries, config, seed, f"{heldout}_{name}_train{seed}")
                folds.append({"test_battery": heldout, "train_batteries": batteries, "train_seed": seed,
                              "candidate": name, "config": asdict(config), "selected_epoch": record["selected_epoch"],
                              "fit_protocol": "validated_temporal_checkpoint", "adapter_path": adapter.relative_to(root).as_posix(),
                              "adapter_sha256": {p.name: e.file_sha256(p) for p in adapter.iterdir() if p.is_file()}})
        manifest["folds"] = folds
        e.write_json(directory / "manifest.json", manifest)
        rows, horizons, window_rows, reload_checks = [], [], [], []
        for fold in folds:
            config = e.ExperimentConfig(**fold["config"])
            heldout, seed = fold["test_battery"], fold["train_seed"]
            adapter = e.resolve_reference(root, fold["adapter_path"])
            pipeline = e.reload_adapter(config, adapter)
            windows = e.make_windows(series[heldout], config.context_length, config.prediction_length)
            expected = None
            for infer_seed in config.inference_seeds:
                samples = e.predict_samples(pipeline, windows["context"], config, infer_seed)
                e.save_forecast(directory / "predictions" / f"{heldout}_train{seed}_infer{infer_seed}.npz", windows, samples)
                r, h, w = e.metric_rows(windows, e.summarize_samples(samples), "LoRA", heldout, seed, infer_seed, samples=samples)
                rows.extend(r); horizons.extend(h); window_rows.extend(w)
                if infer_seed == config.inference_seeds[0]:
                    expected = samples[:config.inference_batch_size].copy()
                print(f"TEST heldout={heldout} train={seed} infer={infer_seed} median_RMSE={r[0]['rmse']:.6f}", flush=True)
            del pipeline
            pipeline = None
            e.clear_device_cache()
            pipeline = e.reload_adapter(config, adapter)
            actual = e.predict_samples(pipeline, windows["context"][:config.inference_batch_size], config, config.inference_seeds[0])
            np.testing.assert_array_equal(actual, expected)
            reload_checks.append({"battery": heldout, "train_seed": seed, "windows": len(actual), "identical": True})
            del pipeline
            pipeline = None
            e.clear_device_cache()
            e._write_tables(directory, rows, horizons, window_rows)
        e.write_json(directory / "reload_checks.json", reload_checks)
        e._paired_comparison(directory, baseline)
        e._finish_run(directory, manifest)
        print(e.audit_saved_run(directory, root, verify_adapters=True), flush=True)
    except Exception as error:
        e._finish_run(directory, manifest, error)
        raise
    finally:
        del pipeline
        e.clear_device_cache()
    return directory
