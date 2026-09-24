"""Check actual adapter initialization and checkpoint reload on the current device."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_experiment import (clear_device_cache, fresh_pipeline, load_config,
                                predict_samples, reload_adapter, verify_saved_run, write_json)


def initialization_digest(config, seed):
    pipeline = fresh_pipeline(config, seed, lora=True)
    digest = hashlib.sha256()
    for name, parameter in pipeline.model.model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    del pipeline
    clear_device_cache()
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--output", type=Path, default=Path("results/reproducibility_checks.json"))
    args = parser.parse_args()
    config = load_config(ROOT, args.config)
    directory = args.run.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    checks = {"lora_artifacts": verify_saved_run(directory, ROOT, config),
              "zeroshot_artifacts": verify_saved_run(Path(manifest["zeroshot_run"]), ROOT, config)}
    seed = config.train_seeds[0]
    first, second = initialization_digest(config, seed), initialization_digest(config, seed)
    assert first == second, "Repeated initialization with the same seed differs"
    checks["adapter_initialization"] = {"seed": seed, "sha256": first, "identical": True}
    print("Identical adapter initialization confirmed", flush=True)
    infer_seed = config.inference_seeds[0]
    # Check each trained fold, rather than just assuming the first saved adapter works.
    checks["adapter_reload"] = []
    for fold in manifest["folds"]:
        filename = f"{fold['test_battery']}_train{fold['train_seed']}_infer{infer_seed}.npz"
        with np.load(directory / "predictions" / filename, allow_pickle=False) as saved:
            context = saved["context"][:config.inference_batch_size].copy()
            expected = saved["samples"][:config.inference_batch_size].copy()
        pipeline = reload_adapter(config, Path(fold["adapter_path"]))
        actual = predict_samples(pipeline, context, config, infer_seed)
        np.testing.assert_array_equal(actual, expected)
        checks["adapter_reload"].append({"battery": fold["test_battery"], "train_seed": fold["train_seed"],
                                          "inference_seed": infer_seed, "windows": len(context), "identical": True})
        print(f"{fold['test_battery']}: reloaded adapter reproduces saved samples exactly", flush=True)
        del pipeline
        clear_device_cache()
    zero = Path(manifest["zeroshot_run"])
    battery = config.battery_ids[0]
    with np.load(zero / "predictions" / f"{battery}_infer{infer_seed}.npz", allow_pickle=False) as saved:
        context = saved["context"][:config.inference_batch_size].copy()
        expected = saved["samples"][:config.inference_batch_size].copy()
    pipeline = fresh_pipeline(config, infer_seed)
    actual = predict_samples(pipeline, context, config, infer_seed)
    np.testing.assert_array_equal(actual, expected)
    checks["zeroshot_repeat"] = {"battery": battery, "inference_seed": infer_seed, "windows": len(context), "identical": True}
    del pipeline
    clear_device_cache()
    checks["scope"] = "Exact checks on current environment and first inference batch of each fold; no cross-device guarantee."
    write_json(args.output, checks)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
