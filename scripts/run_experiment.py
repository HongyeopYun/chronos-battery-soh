"""Run the exact same experiment functions used by notebooks 02 and 03."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.soh_experiment import (audit_saved_run, find_lobo, find_zeroshot, load_config, run_lobo,
                                resolve_reference, run_zeroshot, verify_saved_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["zeroshot", "lora", "all", "audit"])
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--force", action="store_true", help="Start a new run even if a compatible complete run exists")
    parser.add_argument("--run", type=Path, help="Run directory for audit")
    parser.add_argument("--verify-adapters", action="store_true", help="Also require and hash adapter files during audit")
    args = parser.parse_args()
    config = load_config(ROOT, args.config)
    if args.mode == "audit":
        if not args.run:
            parser.error("audit requires --run")
        print(audit_saved_run(args.run.resolve(), ROOT, args.verify_adapters), flush=True)
        manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
        if manifest["kind"] in ("lora", "lora_ablation", "lora_robust"):
            print(audit_saved_run(resolve_reference(ROOT, manifest["zeroshot_run"]), ROOT), flush=True)
        return
    zero = None if args.force else find_zeroshot(ROOT, config)
    if args.mode in ("zeroshot", "all"):
        zero = zero or run_zeroshot(ROOT, config)
        print("ZERO_RUN=" + str(zero), flush=True)
        print(verify_saved_run(zero, ROOT, config), flush=True)
    if args.mode in ("lora", "all"):
        lora = None if args.force else find_lobo(ROOT, config)
        lora = lora or run_lobo(ROOT, config, zero)
        print("LORA_RUN=" + str(lora), flush=True)
        print(verify_saved_run(lora, ROOT, config), flush=True)
        manifest = json.loads((lora / "manifest.json").read_text(encoding="utf-8"))
        print(verify_saved_run(resolve_reference(ROOT, manifest["zeroshot_run"]), ROOT, config), flush=True)


if __name__ == "__main__":
    main()
