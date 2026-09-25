"""Check actual adapter initialization and checkpoint reload on the current device."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.soh_experiment import (ExperimentConfig, audit_saved_run, clear_device_cache,
                                evaluation_signature, fresh_pipeline, load_config, predict_samples,
                                reload_adapter, resolve_reference, verify_saved_run, write_json)


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


def prepare_run(directory, root, config_path=None):
    """Check stored inputs for reproduction, independently of training-cache reuse."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["kind"] not in ("lora", "lora_ablation", "lora_robust"):
        raise ValueError("Expected a LoRA, ablation or robust LoRA run")
    config = ExperimentConfig(**manifest["config"])
    if config_path is not None and load_config(root, config_path) != config:
        raise ValueError("Explicit --config differs from the run's saved configuration")
    if manifest["evaluation_signature"] != evaluation_signature(root, config):
        raise ValueError("Evaluation signature incompatible with the current environment/code")
    # Reproducing saved checkpoints does not select or resume training policies.
    # All run kinds share the artifact audit; special kinds must not enter the
    # ordinary LoRA training-cache verifier.
    checks = {"lora_artifacts": audit_saved_run(directory, root, verify_adapters=True)}
    zero = resolve_reference(root, manifest["zeroshot_run"])
    zero_manifest = json.loads((zero / "manifest.json").read_text(encoding="utf-8"))
    if zero_manifest["kind"] != "zeroshot" or zero_manifest["evaluation_signature"] != manifest["evaluation_signature"]:
        raise ValueError("Linked baseline must be a compatible zero-shot run")
    zero_config = ExperimentConfig(**zero_manifest["config"])
    checks["zeroshot_artifacts"] = verify_saved_run(zero, root, zero_config)
    if not manifest.get("folds"):
        raise ValueError("No trained folds to reproduce")
    return manifest, config, zero, zero_config, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", help="Optional consistency check against the saved effective configuration")
    parser.add_argument("--output", type=Path, default=Path("results/reproducibility_checks.json"))
    args = parser.parse_args()
    directory = args.run.resolve()
    manifest, config, zero, zero_config, checks = prepare_run(directory, ROOT, args.config)
    first_fold = manifest["folds"][0]
    initialization_config = ExperimentConfig(**first_fold["config"]) if "config" in first_fold else config
    seed = first_fold["train_seed"]
    first, second = initialization_digest(initialization_config, seed), initialization_digest(initialization_config, seed)
    assert first == second, "Repeated initialization with the same seed differs"
    checks["adapter_initialization"] = {"seed": seed, "sha256": first, "identical": True}
    print("Identical adapter initialization confirmed", flush=True)
    # Check each trained fold, rather than just assuming the first saved adapter works.
    checks["adapter_reload"] = []
    for fold in manifest["folds"]:
        fold_config = ExperimentConfig(**fold["config"]) if "config" in fold else config
        infer_seed = fold_config.inference_seeds[0]
        filename = f"{fold['test_battery']}_train{fold['train_seed']}_infer{infer_seed}.npz"
        with np.load(directory / "predictions" / filename, allow_pickle=False) as saved:
            context = saved["context"][:fold_config.inference_batch_size].copy()
            expected = saved["samples"][:fold_config.inference_batch_size].copy()
        pipeline = reload_adapter(fold_config, resolve_reference(ROOT, fold["adapter_path"]))
        actual = predict_samples(pipeline, context, fold_config, infer_seed)
        np.testing.assert_array_equal(actual, expected)
        checks["adapter_reload"].append({"battery": fold["test_battery"], "train_seed": fold["train_seed"],
                                          "inference_seed": infer_seed, "windows": len(context), "identical": True})
        print(f"{fold['test_battery']}: reloaded adapter reproduces saved samples exactly", flush=True)
        del pipeline
        clear_device_cache()
    battery = zero_config.battery_ids[0]
    infer_seed = zero_config.inference_seeds[0]
    with np.load(zero / "predictions" / f"{battery}_infer{infer_seed}.npz", allow_pickle=False) as saved:
        context = saved["context"][:zero_config.inference_batch_size].copy()
        expected = saved["samples"][:zero_config.inference_batch_size].copy()
    pipeline = fresh_pipeline(zero_config, infer_seed)
    actual = predict_samples(pipeline, context, zero_config, infer_seed)
    np.testing.assert_array_equal(actual, expected)
    checks["zeroshot_repeat"] = {"battery": battery, "inference_seed": infer_seed, "windows": len(context), "identical": True}
    del pipeline
    clear_device_cache()
    checks["scope"] = "Exact checks on current environment and first inference batch of each fold; no cross-device guarantee."
    write_json(args.output, checks)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
