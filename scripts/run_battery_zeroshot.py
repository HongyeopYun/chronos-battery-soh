"""Run Chronos zero-shot by swapping the path of a common SOH bundle."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.battery_zeroshot import (battery_zeroshot, load_zeroshot_config, prepare_battery_windows,
                                 refresh_battery_zeroshot, verify_battery_zeroshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, nargs="?")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/battery_zeroshot.json")
    parser.add_argument("--battery-ids", nargs="+")
    parser.add_argument("--max-windows", type=int, help="Smoke check: evenly spaced window limit per battery")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/battery_zeroshot")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Check data/window coverage without loading the model")
    parser.add_argument("--audit", type=Path, help="Verify an existing run and recompute its metrics without inference")
    parser.add_argument("--refresh-metrics", type=Path, help="Update metric tables/report from unchanged saved forecasts")
    args = parser.parse_args()
    if args.audit and args.refresh_metrics:
        parser.error("Choose either --audit or --refresh-metrics")
    if args.audit or args.refresh_metrics:
        if args.prepare_only or args.force or args.battery_ids or args.max_windows is not None:
            parser.error("Saved-run commands use the saved configuration; do not combine them with run overrides")
        action = refresh_battery_zeroshot if args.refresh_metrics else verify_battery_zeroshot
        print(action(args.refresh_metrics or args.audit, args.bundle), flush=True)
        return
    if args.bundle is None:
        parser.error("Provide a common SOH bundle directory")
    config = load_zeroshot_config(args.config)
    overrides = {}
    if args.battery_ids is not None:
        overrides["battery_ids"] = tuple(args.battery_ids)
    if args.max_windows is not None:
        overrides["max_windows_per_battery"] = args.max_windows
    config = replace(config, **overrides)
    if args.prepare_only:
        _, coverage, _, _ = prepare_battery_windows(args.bundle, config)
        print(coverage.to_string(index=False), flush=True)
        print(f"WINDOWS={coverage.evaluated_windows.sum()} ELIGIBLE={coverage.eligible_windows.sum()}", flush=True)
        return
    directory = battery_zeroshot(args.bundle, config, args.output_root, args.force)
    print("REPORT=" + str(directory / "report.md"), flush=True)
    print(verify_battery_zeroshot(directory, args.bundle, config), flush=True)


if __name__ == "__main__":
    main()
