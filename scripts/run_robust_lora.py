"""Run or resume the controlled robust-LoRA follow-up."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_robust import find_robust, run_robust

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/lora_robust.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path)
    mode.add_argument("--force", action="store_true", help="Start a new run instead of reusing a verified completed run")
    args = parser.parse_args()
    cached = None if args.resume or args.force else find_robust(ROOT, args.protocol)
    print(cached or run_robust(ROOT, args.protocol, args.resume), flush=True)
