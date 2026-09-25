"""Controlled follow-up: five validation seeds, longer training, balanced loss."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import numpy as np

from . import soh_experiment as e


# This revision differs only in input validation and completed-cache handling.
# Its completed runs remain compatible when protocol/config/core/runtime match.
# Never use this exception to resume unfinished training from older code.
_COMPATIBLE_COMPLETED_RUNNERS = frozenset({
    "e0e592fe7981dd577d9a099c369b62c3b08ba723265c5aea3fa85f58f788442c",
})


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _json(value):
    return json.loads(json.dumps(value))


def policy_configs(base):
    if base.max_epochs != 20 or base.battery_balance:
        raise ValueError("Robust controls require max_epochs=20 and battery_balance=false")
    if len(base.validation_seeds) < 2 or base.refit_full_data or base.selection_metric != "guarded_crps":
        raise ValueError("Robust experiment requires multiple validation seeds, guarded_crps and preserved checkpoints")
    return {
        "single_seed10": replace(base, max_epochs=10, validation_seeds=(base.validation_seeds[0],)),
        "multi_seed10": replace(base, max_epochs=10),
        "multi_seed20": replace(base, max_epochs=20),
        "balanced20": replace(base, max_epochs=20, battery_balance=True),
    }


def load_robust_config(root, protocol_path=Path("configs/lora_robust.json")):
    protocol = _read(Path(root) / protocol_path)
    base = replace(e.load_config(Path(root), protocol["base_config"]), **protocol["overrides"])
    policy_configs(base)
    return base


def _frozen(root, protocol_path):
    protocol = _read(Path(root) / protocol_path)
    base = load_robust_config(root, protocol_path)
    return _json({"protocol": protocol, "config": asdict(base),
                  "source_sha256": {"core": e.file_sha256(Path(e.__file__)), "runner": e.file_sha256(Path(__file__))}})


def _completed_protocol_matches(saved, current):
    if saved == current:
        return True
    if saved.get("source_sha256", {}).get("runner") not in _COMPATIBLE_COMPLETED_RUNNERS:
        return False
    normalized = _json(saved)
    normalized["source_sha256"]["runner"] = current["source_sha256"]["runner"]
    return normalized == current


def verify_robust(directory, root, protocol_path=Path("configs/lora_robust.json")):
    directory = e.resolve_reference(Path(root), directory)
    manifest = _read(directory / "manifest.json")
    if manifest["kind"] != "lora_robust" or manifest["status"] != "complete":
        raise ValueError("Expected a complete robust LoRA experiment")
    if not _completed_protocol_matches(_read(directory / "protocol.json"), _frozen(root, protocol_path)):
        raise ValueError("Robust protocol or implementation changed")
    base = load_robust_config(root, protocol_path)
    if manifest["evaluation_signature"] != e.evaluation_signature(Path(root), base):
        raise ValueError("Robust evaluation signature incompatible")
    if e.file_sha256(directory / "search_decisions.json") != manifest["decisions_sha256"]:
        raise ValueError("Frozen robust decisions changed")
    e.verify_zeroshot(e.resolve_reference(Path(root), manifest["zeroshot_run"]), Path(root), base)
    return e.audit_saved_run(directory, Path(root), verify_adapters=True)


def find_robust(root, protocol_path=Path("configs/lora_robust.json")):
    for path in sorted((Path(root) / "results/runs/ablations").glob("*_lora_robust_*/manifest.json"), reverse=True):
        try:
            verify_robust(path.parent, root, protocol_path)
            return path.parent
        except (ValueError, KeyError, FileNotFoundError, AssertionError):
            continue
    return None


def _trajectory(root, directory, series, heldout, seed, config, policies, label):
    output = directory / "training" / f"{label}.json"
    request = _json({"config": asdict(config), "policies": {k: asdict(v) for k, v in policies.items()},
                     "heldout": heldout, "seed": seed})
    if output.exists():
        previous = _read(output)
        if previous.get("status") == "complete":
            if previous["request"] != request:
                raise ValueError(f"Changed trajectory request: {label}")
            for selected in previous["policies"].values():
                adapter = e.resolve_reference(root, selected["adapter_path"])
                for name, digest in selected["adapter_sha256"].items():
                    if e.file_sha256(adapter / name) != digest:
                        raise ValueError(f"Changed adapter: {adapter}/{name}")
            return previous
    batteries = [b for b in config.battery_ids if b != heldout]
    training, validation, split_info = [], {}, {}
    for b in batteries:
        train, valid, cutoff = e.temporal_split(series[b], config)
        training.append(train)
        validation[b] = valid
        split_info[b] = {"cutoff": cutoff, "training_windows": len(train["target"]),
                         "validation_windows": len(valid["target"])}
    pipeline, optimizer, loader = None, None, None
    record = {"status": "running", "request": request, "train_batteries": batteries,
              "splits": split_info, "history": [], "policies": {}}
    try:
        pipeline = e.fresh_pipeline(config, seed, lora=True)
        loader = e._training_loader(pipeline, training, config, seed)
        optimizer = e._optimizer(pipeline, config)
        for epoch in range(config.max_epochs + 1):
            loss = None if epoch == 0 else e._train_epoch(pipeline, loader, optimizer, seed + epoch, config.kl_weight)
            raw = e.validation_seed_scores(pipeline, validation, config)
            score, by_battery = e.aggregate_validation_scores(raw)
            record["history"].append({"epoch": epoch, "train_loss": loss, "validation": score,
                                       "by_battery": by_battery, "by_seed": raw})
            for name, policy in policies.items():
                if epoch > policy.max_epochs:
                    continue
                subset = {s: raw[s] for s in policy.validation_seeds}
                candidate, _ = e.aggregate_validation_scores(subset)
                best = record["policies"].get(name)
                if best is None or e.validation_candidate_improves(candidate, best["selected_validation"], best["baseline_validation"], policy):
                    adapter = root / "models" / directory.name / label / name
                    pipeline.model.model.save_pretrained(adapter)
                    record["policies"][name] = {
                        "selected_epoch": epoch, "selected_validation": candidate,
                        "baseline_validation": candidate if best is None else best["baseline_validation"],
                        "config": asdict(policy), "adapter_path": adapter.relative_to(root).as_posix(),
                        "training_record": output.relative_to(root).as_posix(),
                    }
            print(f"TRAIN {label} epoch={epoch}/{config.max_epochs} loss={loss} median={score['median_rmse']:.7f} "
                  f"mean={score['mean_rmse']:.7f} CRPS={score['crps']:.7f} "
                  f"selected={ {k:v['selected_epoch'] for k,v in record['policies'].items()} }", flush=True)
            e.write_json(output, record)
        for selected in record["policies"].values():
            adapter = e.resolve_reference(root, selected["adapter_path"])
            selected["adapter_sha256"] = {p.name: e.file_sha256(p) for p in adapter.iterdir() if p.is_file()}
        record["status"] = "complete"
        e.write_json(output, record)
        return _json(record)
    finally:
        del optimizer, loader, pipeline
        e.clear_device_cache()


def _fold(heldout, seed, policy, selected, base):
    return {"test_battery": heldout, "train_seed": seed, "selected_policy": policy,
            "train_batteries": [b for b in base.battery_ids if b != heldout],
            "selected_epoch": selected["selected_epoch"], "config": selected["config"],
            "adapter_path": selected["adapter_path"], "adapter_sha256": selected["adapter_sha256"],
            "training_record": selected["training_record"],
            "fit_protocol": "validated_temporal_checkpoint"}


def _evaluate(root, destination, series, fold):
    """Forecast and verify a reloaded checkpoint; no training or selection here."""
    config = e.ExperimentConfig(**fold["config"])
    heldout, seed = fold["test_battery"], fold["train_seed"]
    windows = e.make_windows(series[heldout], config.context_length, config.prediction_length)
    pipeline = None
    rows, horizons, window_rows = [], [], []
    try:
        pipeline = e.reload_adapter(config, e.resolve_reference(root, fold["adapter_path"]))
        expected = None
        for infer_seed in config.inference_seeds:
            samples = e.predict_samples(pipeline, windows["context"], config, infer_seed)
            e.save_forecast(destination / "predictions" / f"{heldout}_train{seed}_infer{infer_seed}.npz", windows, samples)
            r, h, w = e.metric_rows(windows, e.summarize_samples(samples), "LoRA", heldout, seed, infer_seed, samples=samples)
            rows.extend(r); horizons.extend(h); window_rows.extend(w)
            if infer_seed == config.inference_seeds[0]:
                expected = samples[:config.inference_batch_size].copy()
            print(f"TEST {destination.name} {heldout} train={seed} infer={infer_seed} RMSE={r[0]['rmse']:.7f}", flush=True)
        del pipeline
        pipeline = None
        e.clear_device_cache()
        pipeline = e.reload_adapter(config, e.resolve_reference(root, fold["adapter_path"]))
        actual = e.predict_samples(pipeline, windows["context"][:config.inference_batch_size], config, config.inference_seeds[0])
        np.testing.assert_array_equal(actual, expected)
        return rows, horizons, window_rows
    finally:
        del pipeline
        e.clear_device_cache()


def _copy_evaluation(source, destination, heldout, seed, config):
    import pandas as pd
    (destination / "predictions").mkdir(parents=True, exist_ok=True)
    for infer_seed in config.inference_seeds:
        name = f"{heldout}_train{seed}_infer{infer_seed}.npz"
        shutil.copy2(source / "predictions" / name, destination / "predictions" / name)
    tables = []
    for name in ("metrics.csv", "horizon_metrics.csv", "window_metrics.csv"):
        frame = pd.read_csv(source / name)
        tables.append(frame[(frame.battery == heldout) & (frame.train_seed == seed)].to_dict(orient="records"))
    return tables


def run_robust(root, protocol_path=Path("configs/lora_robust.json"), resume=None):
    root = Path(root).resolve()
    protocol = _read(root / protocol_path)
    base = load_robust_config(root, protocol_path)
    policies = policy_configs(base)
    seed = protocol["search_train_seed"]
    order = protocol["candidate_order"]
    if seed not in base.train_seeds or order != ["multi_seed10", "multi_seed20", "balanced20"]:
        raise ValueError("Unsupported search seed or candidate order")
    frozen = _frozen(root, protocol_path)
    if resume:
        directory = e.resolve_reference(root, resume)
        manifest = _read(directory / "manifest.json")
        if manifest["status"] == "complete":
            verify_robust(directory, root, protocol_path)
            return directory
        if manifest["kind"] != "lora_robust" or _read(directory / "protocol.json") != frozen:
            raise ValueError("Cannot resume a different kind, protocol or implementation")
        if manifest["evaluation_signature"] != e.evaluation_signature(root, base):
            raise ValueError("Cannot resume with changed data/runtime")
        baseline = e.resolve_reference(root, manifest["zeroshot_run"])
    else:
        baseline = e.find_zeroshot(root, base) or e.run_zeroshot(root, base)
        directory, manifest = e._start_run(root, base, "lora_robust")
        manifest["zeroshot_run"] = baseline.relative_to(root).as_posix()
        e.write_json(directory / "protocol.json", frozen)
    e.verify_zeroshot(baseline, root, base)
    manifest["status"] = "running"
    e.write_json(directory / "manifest.json", manifest)
    print(f"ROBUST_RUN={directory}", flush=True)
    series = e.load_soh(root, base)
    try:
        choices, selected_policies = {}, {}
        for heldout in base.battery_ids:
            training_series = {b: values for b, values in series.items() if b != heldout}
            window = _trajectory(root, directory, training_series, heldout, seed, base,
                                 {k: policies[k] for k in ("single_seed10", "multi_seed10", "multi_seed20")}, f"{heldout}_window_train{seed}")
            sizes = [v["training_windows"] for v in window["splits"].values()]
            if len(set(sizes)) == 1:
                # Weights N/(B*n_b) are exactly 1, and core uses identical loss reduction.
                balanced = {**window["policies"]["multi_seed20"], "config": asdict(policies["balanced20"]),
                            "equivalent_to": "multi_seed20", "equivalence": "All battery weights are exactly one; identical shuffle, loss and updates"}
            else:
                balanced = _trajectory(root, directory, training_series, heldout, seed, policies["balanced20"],
                                       {"balanced20": policies["balanced20"]}, f"{heldout}_balanced_train{seed}")["policies"]["balanced20"]
            records = {**window["policies"], "balanced20": balanced}
            if records["multi_seed20"]["baseline_validation"] != balanced["baseline_validation"]:
                raise ValueError("Balanced and unbalanced epoch-zero validation must match")
            winner = order[0]
            for name in order[1:]:
                if records[name]["selected_validation"]["crps"] < records[winner]["selected_validation"]["crps"] - base.min_delta:
                    winner = name
            choices[heldout], selected_policies[heldout] = winner, records
            print(f"CHOSEN {heldout} {winner} epoch={records[winner]['selected_epoch']}", flush=True)
        decisions = _json({"choices": choices, "policies": selected_policies, "test_results_used": False})
        decision_path = directory / "search_decisions.json"
        if decision_path.exists() and _read(decision_path) != decisions:
            raise ValueError("Frozen choices changed")
        e.write_json(decision_path, decisions)
        manifest["decisions_sha256"] = e.file_sha256(decision_path)
        e.write_json(directory / "manifest.json", manifest)
        folds = []
        for heldout, winner in choices.items():
            for train_seed in base.train_seeds:
                selected = selected_policies[heldout][winner]
                if train_seed != seed:
                    config = policies[winner]
                    selected = _trajectory(root, directory, {b: s for b, s in series.items() if b != heldout},
                                           heldout, train_seed, config, {winner: config},
                                           f"{heldout}_{winner}_train{train_seed}")["policies"][winner]
                folds.append(_fold(heldout, train_seed, winner, selected, base))
                history = _read(e.resolve_reference(root, selected["training_record"]))["history"]
                history = [{**row, "validation_macro_rmse": row["validation"]["median_rmse"]}
                           for row in history if row["epoch"] <= selected["config"]["max_epochs"]]
                e.write_json(directory / "selection" / f"{heldout}_train{train_seed}.json",
                             {"status": "complete", "train_batteries": [b for b in base.battery_ids if b != heldout],
                              "history": history, "selected_epoch": selected["selected_epoch"],
                              "selected_policy": winner, "selected_validation": selected["selected_validation"],
                              "baseline_validation": selected["baseline_validation"]})
        manifest["folds"] = folds
        e.write_json(directory / "manifest.json", manifest)
        # All choices and all repeated-seed checkpoints exist before any heldout forecast.
        evaluated = {}
        for name in policies:
            destination = directory / "policy_evaluation" / name
            destination.mkdir(parents=True, exist_ok=True)
            tables = [[], [], []]
            for heldout in base.battery_ids:
                selected = selected_policies[heldout][name]
                fold = _fold(heldout, seed, name, selected, base)
                # Adapter weights identify equal checkpoints even when paths differ.
                key = (heldout, selected["adapter_sha256"]["adapter_model.safetensors"])
                values = (_copy_evaluation(evaluated[key], destination, heldout, seed, base) if key in evaluated else
                          _evaluate(root, destination, series, fold))
                for target, value in zip(tables, values):
                    target.extend(value)
                e._write_tables(destination, *tables)
                evaluated[key] = destination
        tables, reload_checks = [[], [], []], []
        for fold in folds:
            if fold["train_seed"] == seed:
                values = _copy_evaluation(directory / "policy_evaluation" / fold["selected_policy"], directory,
                                          fold["test_battery"], seed, base)
            else:
                values = _evaluate(root, directory, series, fold)
            for target, value in zip(tables, values):
                target.extend(value)
            reload_checks.append({"battery": fold["test_battery"], "train_seed": fold["train_seed"],
                                  "identical": True, "scope": "First inference batch after adapter reload"})
            e._write_tables(directory, *tables)
        e.write_json(directory / "reload_checks.json", reload_checks)
        e._paired_comparison(directory, baseline)
        e._finish_run(directory, manifest)
        print(e.audit_saved_run(directory, root, verify_adapters=True), flush=True)
    except Exception as error:
        e._finish_run(directory, manifest, error)
        raise
    return directory
