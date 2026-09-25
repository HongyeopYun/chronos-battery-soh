"""Diagnose validation sampling sensitivity without retraining or changing choices."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src import soh_experiment as e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/lora_validation_sensitivity.json"))
    args = parser.parse_args()
    directory = args.run.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    audit = e.audit_saved_run(directory, ROOT, verify_adapters=True)
    base = e.ExperimentConfig(**manifest["config"])
    all_series = e.load_soh(ROOT, base)
    search_seed = protocol["protocol"]["search_train_seed"]
    folds = [f for f in manifest["folds"] if f["train_seed"] == search_seed and f["selected_epoch"] > 0]
    validation_seeds = [base.validation_seed + i for i in range(5)]
    record = {"status": "running", "source_run": directory.relative_to(ROOT).as_posix(),
              "scope": "Previously selected nonzero checkpoints at search training seed; validation only, no model reselection or new test forecasts.",
              "validation_seeds": validation_seeds, "validation_samples": base.validation_samples,
              "frozen_choices_sha256": e.file_sha256(directory / "choices.json"), "audit": audit,
              "core_sha256": e.file_sha256(ROOT / "src" / "soh_experiment.py"),
              "script_sha256": e.file_sha256(Path(__file__)), "folds": []}
    e.write_json(args.output, record)
    csv_rows = []
    for fold in folds:
        config = e.ExperimentConfig(**fold["config"])
        validation = {b: e.temporal_split(all_series[b], config)[1] for b in fold["train_batteries"]}
        assert fold["test_battery"] not in validation
        rows = {seed: {} for seed in validation_seeds}
        pipeline = None
        try:
            for label in ("zero", "selected"):
                pipeline = (e.fresh_pipeline(config, search_seed) if label == "zero" else
                            e.reload_adapter(config, e.resolve_reference(ROOT, fold["adapter_path"])))
                for seed in validation_seeds:
                    score, by_battery = e._validation_score(pipeline, validation, replace(config, validation_seed=seed))
                    rows[seed][label] = score
                    csv_rows.append({"heldout": fold["test_battery"], "model": label, "validation_seed": seed, **score})
                    print(f"{fold['test_battery']} {label} validation_seed={seed} median={score['median_rmse']:.8f} "
                          f"mean={score['mean_rmse']:.8f} crps={score['crps']:.8f}", flush=True)
                del pipeline
                pipeline = None
                e.clear_device_cache()
            original = json.loads((directory / "selection" /
                f"{fold['test_battery']}_{fold['candidate']}_train{search_seed}.json").read_text(encoding="utf-8"))
            for label, key in (("zero", "baseline_validation"), ("selected", "selected_validation")):
                for metric, value in original[key].items():
                    np.testing.assert_allclose(rows[base.validation_seed][label][metric], value, rtol=0, atol=1e-12)
            for seed, row in rows.items():
                row["passes_original_rule"] = e.validation_candidate_improves(row["selected"], row["zero"], row["zero"], config)
                row["crps_improvement"] = row["zero"]["crps"] - row["selected"]["crps"]
                row["median_rmse_improvement"] = row["zero"]["median_rmse"] - row["selected"]["median_rmse"]
                row["mean_rmse_improvement"] = row["zero"]["mean_rmse"] - row["selected"]["mean_rmse"]
            averages = {label: {metric: float(np.mean([row[label][metric] for row in rows.values()]))
                               for metric in rows[base.validation_seed][label]}
                        for label in ("zero", "selected")}
            record["folds"].append({"heldout": fold["test_battery"], "train_batteries": fold["train_batteries"],
                "candidate": fold["candidate"], "selected_epoch": fold["selected_epoch"],
                "rows": rows, "pass_count": sum(row["passes_original_rule"] for row in rows.values()),
                "seed_average": averages, "average_passes_original_rule": e.validation_candidate_improves(
                    averages["selected"], averages["zero"], averages["zero"], config),
                "original_seed_exactly_reproduced": True})
            e.write_json(args.output, record)
        finally:
            del pipeline
            e.clear_device_cache()
    if e.file_sha256(directory / "choices.json") != record["frozen_choices_sha256"]:
        raise ValueError("Original choices changed")
    record["status"] = "complete"
    e.write_json(args.output, record)
    pd.DataFrame(csv_rows).to_csv(args.output.with_suffix(".csv"), index=False)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
