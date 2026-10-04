"""Conservative capacity-spike exclusion using only earlier measurements."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.soh_data import (EXCLUDED_FLAGS, SCHEMA_VERSION, load_bundle, observed_windows,
                          sha256, validate_tables, write_json)

OUTLIER_FLAG = "capacity_outlier"
REMOVED_COLUMNS = ["dataset", "battery_id", "cycle", "capacity_ah", "soh",
                   "past_median_soh", "history_count", "rule"]


def validate_policy(policy):
    if policy["schema_version"] != SCHEMA_VERSION or policy["rule"] != "large_upward_capacity_spike":
        raise ValueError("Unsupported cleaning rule or schema")
    for name in ["history_length", "minimum_history", "context_length", "prediction_length"]:
        value = policy[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if policy["minimum_history"] > policy["history_length"]:
        raise ValueError("Minimum history exceeds history length")
    for name in ["minimum_soh", "minimum_excess_soh"]:
        if not np.isfinite(policy[name]) or policy[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")


def flag_capacity_outliers(data, policy):
    """Mark only large upward spikes; never inspect future rows or change measured values."""
    validate_policy(policy)
    flagged = data.reset_index(drop=True).copy()
    if flagged.duplicated(["dataset", "battery_id", "cycle"]).any():
        raise ValueError("Duplicate cycle identity")
    removed = []
    for _, rows in flagged.groupby(["dataset", "battery_id"], sort=False):
        if not rows.cycle.is_monotonic_increasing:
            raise ValueError("Unordered source cycles")
        history = deque(maxlen=policy["history_length"])
        previous_key, previous_cycle = None, None
        for row in rows.itertuples():
            key = (getattr(row, "segment_id", row.battery_id), row.protocol_id)
            if key != previous_key or (previous_cycle is not None and row.cycle != previous_cycle + 1):
                history.clear()
            previous_key, previous_cycle = key, row.cycle
            flags = set(str(row.quality_flags).split(";"))
            if (not row.is_observed or not np.isfinite(row.capacity_ah) or row.capacity_ah <= 0
                    or not np.isfinite(row.soh) or row.capacity_kind not in {"full_discharge", "reference_test"}
                    or flags & EXCLUDED_FLAGS):
                continue
            baseline = float(np.median(history)) if len(history) >= policy["minimum_history"] else None
            if (baseline is not None and row.soh > policy["minimum_soh"]
                    and row.soh - baseline > policy["minimum_excess_soh"]):
                flagged.at[row.Index, "quality_flags"] = ";".join(filter(None, [row.quality_flags, OUTLIER_FLAG]))
                removed.append({"dataset": row.dataset, "battery_id": row.battery_id, "cycle": int(row.cycle),
                    "capacity_ah": float(row.capacity_ah), "soh": float(row.soh), "past_median_soh": baseline,
                    "history_count": len(history), "rule": policy["rule"]})
            else:
                history.append(float(row.soh))
    return flagged, pd.DataFrame(removed, columns=REMOVED_COLUMNS)


def clean_bundle(source, destination, policy):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Cleaned output must be separate from the original bundle")
    data, batteries, parent = load_bundle(source)
    input_hashes = {p.name: sha256(p) for p in source.iterdir() if p.is_file()}
    flagged, removed = flag_capacity_outliers(data, policy)
    mask = flagged.quality_flags.map(lambda f: OUTLIER_FLAG in str(f).split(";"))
    cleaned = flagged.loc[~mask].copy()
    validate_tables(cleaned, batteries)
    audit_rows = []
    for battery_id, original in data.groupby("battery_id", sort=False):
        after = cleaned[cleaned.battery_id == battery_id]
        before_windows = observed_windows(original, policy["context_length"], policy["prediction_length"])
        after_windows = observed_windows(after, policy["context_length"], policy["prediction_length"])
        audit_rows.append({"battery_id": battery_id, "rows_before": len(original), "rows_after": len(after),
            "outliers_removed": len(original) - len(after), "windows_before": len(before_windows["target"]),
            "windows_after": len(after_windows["target"])})
    audit = pd.DataFrame(audit_rows)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": SCHEMA_VERSION, "status": "running", "stage": "outlier_cleaning",
                "parent_manifest_sha256": input_hashes["manifest.json"], "source_files": parent["source_files"],
                "selection": policy, "n_batteries": len(batteries), "n_rows_before": len(data), "n_rows": len(cleaned),
                "n_removed_outliers": int(mask.sum()), "interpolation": "none", "smoothing": "none",
                "cycle_number_policy": "original cycle numbers retained; forecast windows never bridge removed cycles",
                "implementation_sha256": {"src/soh_cleaning.py": sha256(Path(__file__)),
                                           "src/soh_data.py": sha256(Path(__file__).with_name("soh_data.py"))}}
    write_json(destination / "manifest.json", manifest)
    for name in ["batteries.csv", "schema.json"]:
        shutil.copy2(source / name, destination / name)
    for name, table in [("soh.csv", cleaned), ("soh_flagged.csv", flagged), ("audit.csv", audit)]:
        table.to_csv(destination / name, index=False, float_format="%.17g", lineterminator="\n")
    manifest.update(status="complete", artifact_sha256={name: sha256(destination / name)
                    for name in ["soh.csv", "soh_flagged.csv", "batteries.csv", "schema.json", "audit.csv"]})
    write_json(destination / "manifest.json", manifest)
    loaded, _, _ = load_bundle(destination)
    np.testing.assert_allclose(loaded.soh, cleaned.soh, rtol=1e-12, atol=1e-12)
    if input_hashes != {p.name: sha256(p) for p in source.iterdir() if p.is_file()}:
        raise ValueError("Original preprocessing bundle changed during cleaning")
    return cleaned, removed, audit, manifest
