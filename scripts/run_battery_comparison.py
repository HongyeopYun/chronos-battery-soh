"""T5 / Chronos-2 zero-shot pilot or full C2 suite; no training or cross-learning."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows
from src.battery_model_zeroshot import PILOT, load_model, run_model_battery, write_comparison, write_model_suite
from src.battery_zeroshot_minimal import GROUPS
from src.battery_zeroshot import write_json
from src.soh_experiment import clear_device_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["t5", "c2", "both"], default="both")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--all", action="store_true", help="Evaluate all seven cleaned groups with --model c2")
    parser.add_argument("--output-root", type=Path, help="Separate output directory for a new full-suite protocol")
    args = parser.parse_args()
    if args.all and args.model != "c2":
        parser.error("Full C2 evaluation uses --model c2 --all; the previous full T5 suite remains available")
    if args.output_root is not None and not args.all:
        parser.error("--output-root is supported for the full suite; pilot uses its comparison directory")
    # Fast C2 first, then T5; each model is released before loading the next one.
    families = ["c2", "t5"] if args.model == "both" else [args.model]
    for family in families:
        tag = family + "-battery"
        config = ZeroShotConfig(**json.loads((ROOT / "configs" / f"{tag}.json").read_text(encoding="utf-8")))
        settings = {group: config for group in GROUPS} if args.all else {
                        group: replace(config, battery_ids=ids) for group, ids in PILOT.items()}
        if args.all and (config.battery_ids is not None or config.max_windows_per_battery is not None or config.stride != 1):
            parser.error("The full C2 suite requires all batteries, all windows and stride 1")
        if args.prepare_only:
            for group, setting in settings.items():
                _, coverage, _, _ = prepare_battery_windows(ROOT / "data/processed/clean" / group, setting)
                print(tag, group, f"{setting.context_length}->{setting.prediction_length}",
                      f"evaluated={int((coverage.evaluated_windows>0).sum())} skipped={int((coverage.evaluated_windows==0).sum())} windows={int(coverage.evaluated_windows.sum())}", flush=True)
            continue
        scope = "full" if args.all else "pilot"
        output_root = args.output_root or ROOT / "results" / tag / scope
        state_path = output_root / ".runtime" / f"{tag}_status.json"
        state = {"status": "running", "groups_completed": [], "groups_total": len(settings)}
        if args.all:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(state_path, state)
        pipeline = None
        try:
            pipeline = load_model(config)
            for group, setting in settings.items():
                directory = run_model_battery(ROOT / "data/processed/clean" / group, setting,
                                              output_root=output_root, pipeline=pipeline, scope=scope)
                print("REPORT=" + str(directory / f"{tag}_report.md"), flush=True)
                if args.all:
                    state["groups_completed"].append(group)
                    write_model_suite(output_root, tag, state["groups_completed"])
                    write_json(state_path, state)
                else:
                    write_comparison()
            if args.all:
                state["status"] = "complete"
                write_json(state_path, state)
        except Exception as error:
            if args.all:
                state.update(status="failed", error=f"{type(error).__name__}: {error}")
                write_json(state_path, state)
            raise
        finally:
            del pipeline
            clear_device_cache()
    if not args.prepare_only:
        print("ALL_C2_COMPLETE" if args.all else "COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
