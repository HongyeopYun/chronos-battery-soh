"""Three-learning-rate search, best-checkpoint evaluation, and saved-result audit."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_checkpoint import load_checkpoint_config, find_checkpoint_run, run_checkpoint_search, verify_checkpoint_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_checkpoint.json")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg, policy = load_checkpoint_config(ROOT, args.config)
    if args.audit:
        print(verify_checkpoint_run(args.audit, ROOT, cfg, policy), flush=True)
    else:
        directory = None if args.force or args.resume else find_checkpoint_run(ROOT, cfg, policy)
        directory = directory or run_checkpoint_search(ROOT, cfg, policy, resume=args.resume)
        print("CHECKPOINT_RUN=" + str(directory), flush=True)
        print(verify_checkpoint_run(directory, ROOT, cfg, policy), flush=True)
