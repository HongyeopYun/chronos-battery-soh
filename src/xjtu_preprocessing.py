"""Audit and extract XJTU Batch 1 (2C charging) into the common SOH format."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.soh_data import (SCHEMA_VERSION, SOH_COLUMNS, load_bundle, observed_windows,
                          sha256, validate_tables, write_json)

INITIAL_DESCRIPTION = "0.5C charge and 0.2C discharge [test capacity]"
NORMAL_DESCRIPTION = "2C charge and 1C discharge"


def extract_battery(path, policy):
    path = Path(path)
    if not re.fullmatch(r"2C_battery-[1-8]\.mat", path.name):
        raise ValueError(f"Not an XJTU Batch-1 source: {path.name}")
    raw = loadmat(path, simplify_cells=True, variable_names=["summary", "data"])
    summary, data = raw["summary"], raw["data"]
    data = [data] if isinstance(data, dict) else list(data)
    q = np.asarray(summary["discharge_capacity_Ah"], dtype=np.float64).reshape(-1)
    if len(q) != len(data) or not len(q):
        raise ValueError("Capacity and raw cycle counts do not match")
    if not np.isfinite(q[0]) or q[0] <= 0:
        raise ValueError("First-cycle reference capacity is invalid; no fallback normalization is allowed")
    if str(summary["description"]).strip() != "2C charge experiment":
        raise ValueError("Unexpected summary experiment description")
    battery = f"{policy['batch']}_{path.stem}"
    descriptions = [str(c["description"]).strip() for c in data]
    soh = q / q[0]
    rows = []
    for i, (capacity, value, description) in enumerate(zip(q, soh, descriptions)):
        flags = []
        expected = INITIAL_DESCRIPTION if i == 0 else NORMAL_DESCRIPTION
        if i == 0:
            flags.append("initial_capacity_test")
        if description != expected:
            flags.append("unexpected_protocol")
        if not np.isfinite(capacity):
            flags.append("nonfinite_capacity")
        elif capacity <= 0:
            flags.append("nonpositive_capacity")
        if i and np.isfinite(capacity) and np.isfinite(q[i-1]) and capacity > q[i-1]:
            flags.append("capacity_increase")
        if i and description != descriptions[i-1]:
            flags.append("protocol_transition")
        if (i and description == descriptions[i-1] and np.isfinite(value) and np.isfinite(soh[i-1])
                and abs(value - soh[i-1]) > policy["large_soh_change_threshold"]):
            flags.append("large_soh_change")
        if np.isfinite(value) and value > 1:
            flags.append("soh_above_one")
        rows.append({"dataset": policy["dataset"], "battery_id": battery, "cycle": i + 1,
                     "capacity_ah": capacity, "soh": value,
                     "capacity_kind": "unknown" if description != expected else ("reference_test" if i == 0 else "full_discharge"),
                     "is_observed": True, "quality_flags": ";".join(flags),
                     "protocol_id": "unknown" if description != expected else ("initial_capacity_test" if i == 0 else "cccv_2c_charge_1c_discharge"),
                     "cycle_description": description})
    frame = pd.DataFrame(rows, columns=SOH_COLUMNS)
    metadata = {"dataset": policy["dataset"], "battery_id": battery, "batch": policy["batch"],
                "raw_battery_name": path.stem, "reference_method": "first_cycle", "reference_cycle": 1,
                "reference_capacity_ah": float(q[0]), "nominal_capacity_ah": policy["nominal_capacity_ah"],
                "charge_rate_c": policy["normal_charge_rate_c"], "discharge_rate_c": policy["normal_discharge_rate_c"],
                "initial_charge_rate_c": policy["initial_charge_rate_c"],
                "initial_discharge_rate_c": policy["initial_discharge_rate_c"],
                "chemistry": "NCM", "temperature_condition": "room_temperature", "temperature_c": None,
                "source_file": path.name, "source_sha256": sha256(path),
                "cycle_number_source": "data_array_position_1based",
                "source_summary_cycle_life": int(summary["cycle_life"]),
                "source_description": str(summary["description"])}
    validate_tables(frame, pd.DataFrame([metadata]))
    windows = observed_windows(frame, policy["context_length"], policy["prediction_length"],
                               policy["prediction_excluded_flags"])
    n_invalid = int((~np.isfinite(q) | (q <= 0)).sum())
    n_protocol = sum(d != (INITIAL_DESCRIPTION if i == 0 else NORMAL_DESCRIPTION)
                     for i, d in enumerate(descriptions))
    notes = ["initial_cycle_protocol_change", "cycle_numbers_from_array_position"]
    if np.any(np.diff(q) > 0):
        notes.append("capacity_increases_preserved")
    if np.any(soh > 1):
        notes.append("soh_above_one_preserved")
    n_large_changes = int(frame.quality_flags.str.contains("large_soh_change", regex=False).sum())
    if n_large_changes:
        notes.append("large_soh_changes_review_only_preserved")
    if metadata["source_summary_cycle_life"] != len(q):
        notes.append("cycle_life_differs_from_record_count_no_truncation")
    audit = {"dataset": policy["dataset"], "battery_id": battery, "source_file": path.name,
             "status": "failed" if n_invalid or n_protocol else "passed_with_notes",
             "n_source_cycles": len(data), "n_extracted_cycles": len(frame), "first_cycle": 1, "last_cycle": len(q),
             "n_nonfinite_capacity": int((~np.isfinite(q)).sum()), "n_nonpositive_capacity": int((q <= 0).sum()),
             "n_duplicate_cycles": 0, "n_cycle_gaps_in_stored_index": 0,
             "cycle_number_source": metadata["cycle_number_source"],
             "n_unexpected_protocol": n_protocol, "n_capacity_increases": int((np.diff(q) > 0).sum()),
             "n_large_soh_changes": n_large_changes,
             "n_soh_above_one": int((soh > 1).sum()), "n_initial_capacity_test": 1,
             "reference_capacity_ah": float(q[0]), "last_capacity_ah": float(q[-1]), "last_soh": float(soh[-1]),
             "first_step_soh_change": float(soh[1] - soh[0]) if len(soh) > 1 else None,
             "prediction_context": policy["context_length"], "prediction_horizon": policy["prediction_length"],
             "n_prediction_windows": len(windows["target"]),
             "prediction_ready": bool(len(windows["target"])) and not (n_invalid or n_protocol),
             "first_forecast_cycle": int(windows["forecast_start"][0]) if len(windows["target"]) else None,
             "notes": ";".join(notes), "error": ""}
    del raw, data
    gc.collect()
    return frame, metadata, audit


def preprocess(raw_directory, destination, policy):
    raw_directory, destination = Path(raw_directory).resolve(), Path(destination).resolve()
    if policy["schema_version"] != SCHEMA_VERSION or policy["reference_method"] != "first_cycle" or policy["reference_cycle"] != 1:
        raise ValueError("Unsupported schema or normalization policy")
    if policy["dataset"] != "xjtu" or policy["batch"] != "batch1":
        raise ValueError("This extractor supports XJTU Batch 1 only")
    if not np.isfinite(policy["large_soh_change_threshold"]) or policy["large_soh_change_threshold"] <= 0:
        raise ValueError("Large-change audit threshold must be positive and finite")
    paths = sorted(raw_directory.glob(policy["source_pattern"]), key=lambda p: int(p.stem.split("-")[-1]))
    if len(paths) != policy["expected_batteries"] or {p.name for p in paths} != {f"2C_battery-{i}.mat" for i in range(1, 9)}:
        raise ValueError("Expected exactly 2C_battery-1.mat through 2C_battery-8.mat")
    if raw_directory == destination or raw_directory in destination.parents:
        raise ValueError("Processed output must be outside the raw directory")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": SCHEMA_VERSION, "status": "running", "selection": policy,
                "source_files": {p.name: sha256(p) for p in paths},
                "implementation_sha256": {"src/soh_data.py": sha256(Path(__file__).with_name("soh_data.py")),
                                           "src/xjtu_preprocessing.py": sha256(Path(__file__))},
                "scope": "XJTU Batch 1 only; no prediction or training performed",
                "cycle_semantics": "1-based positions in the published MAT data array; acquisition gaps cannot be independently inferred",
                "interpolation": "none", "smoothing": "none", "rows_dropped": 0}
    write_json(destination / "manifest.json", manifest)
    frames, metadata, audits = [], [], []
    for path in paths:
        print("EXTRACT", path.name, flush=True)
        try:
            frame, battery, audit = extract_battery(path, policy)
            if sha256(path) != manifest["source_files"][path.name]:
                raise ValueError("Raw source changed during extraction")
            frames.append(frame)
            metadata.append(battery)
            audits.append(audit)
        except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
            audits.append({"dataset": "xjtu", "battery_id": f"batch1_{path.stem}", "source_file": path.name,
                           "status": "failed", "error": f"{type(error).__name__}: {error}"})
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SOH_COLUMNS)
    batteries = pd.DataFrame(metadata)
    audit = pd.DataFrame(audits)
    if frames:
        validate_tables(data, batteries)
    for name, frame in (("soh.csv", data), ("batteries.csv", batteries), ("audit.csv", audit)):
        frame.to_csv(destination / name, index=False, float_format="%.17g", lineterminator="\n")
    schema = {"version": SCHEMA_VERSION, "key": ["dataset", "battery_id", "cycle"],
              "soh_columns": {"dataset": "string", "battery_id": "string", "cycle": "positive_integer",
                              "capacity_ah": "float64, Ah", "soh": "float64, fraction Q/Q_ref",
                              "capacity_kind": "reference_test|full_discharge|partial_discharge|unknown",
                              "is_observed": "boolean", "quality_flags": "semicolon-delimited strings; empty means none",
                              "protocol_id": "string", "cycle_description": "source string"},
              "battery_required_columns": ["dataset", "battery_id", "batch", "reference_method", "reference_cycle",
                                           "reference_capacity_ah", "source_file", "source_sha256", "cycle_number_source"]}
    write_json(destination / "schema.json", schema)
    manifest.update(status="complete" if len(metadata) == len(paths) and not (audit.status == "failed").any() else "audit_failed",
                    n_batteries=len(metadata), n_rows=len(data), n_failed_files=int((audit.status == "failed").sum()),
                    artifact_sha256={name: sha256(destination / name) for name in ["soh.csv", "batteries.csv", "audit.csv", "schema.json"]})
    write_json(destination / "manifest.json", manifest)
    if manifest["status"] == "complete":
        loaded, _, _ = load_bundle(destination)
        np.testing.assert_allclose(loaded.soh, data.soh, rtol=1e-12, atol=1e-12)
    return data, batteries, audit, manifest
