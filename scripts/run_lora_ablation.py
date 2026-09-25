"""Run or resume the predeclared validation-only LoRA ablation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_ablation import run_ablation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("configs/lora_ablation.json"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    directory = run_ablation(ROOT, args.protocol, args.resume)
    # Keep exploratory runs outside the ordinary *_lora_* cache glob.
    # The basename stays unchanged so model/receipt paths remain resumable.
    run_root = (ROOT / "results" / "runs").resolve()
    if directory.parent == run_root:
        archive = run_root / "ablations" / directory.name
        if not archive.resolve().is_relative_to(run_root):
            raise ValueError("Ablation destination escapes the results directory")
        archive.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(archive)
        directory = archive
    print(directory, flush=True)
