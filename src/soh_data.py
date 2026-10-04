"""Common, cycle-level SOH tables; no model or NASA experiment dependencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
SOH_COLUMNS = ["dataset", "battery_id", "cycle", "capacity_ah", "soh", "capacity_kind",
               "is_observed", "quality_flags", "protocol_id", "cycle_description"]
BATTERY_REQUIRED = ["dataset", "battery_id", "batch", "reference_method", "reference_cycle",
                    "reference_capacity_ah", "source_file", "source_sha256", "cycle_number_source"]
EXCLUDED_FLAGS = {"initial_capacity_test", "nonfinite_capacity", "nonpositive_capacity",
                  "unexpected_protocol", "known_collection_issue", "capacity_outlier"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def validate_tables(soh, batteries):
    """Validate identity, order and normalization; invalid measurements must be flagged."""
    for frame, columns in ((soh, SOH_COLUMNS), (batteries, BATTERY_REQUIRED)):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing common SOH columns: {sorted(missing)}")
    keys = ["dataset", "battery_id"]
    if batteries.duplicated(keys).any() or soh.duplicated(keys + ["cycle"]).any():
        raise ValueError("Duplicate battery or cycle identity")
    identities = set(map(tuple, batteries[keys].itertuples(index=False, name=None)))
    if set(map(tuple, soh[keys].itertuples(index=False, name=None))) != identities:
        raise ValueError("Cycle and battery identities differ")
    for identity, rows in soh.groupby(keys, sort=False):
        cycles = rows.cycle.to_numpy(dtype=float)
        if not np.isfinite(cycles).all() or np.any(cycles < 1) or np.any(cycles != np.floor(cycles)):
            raise ValueError("Cycle numbers must be positive integers")
        if np.any(np.diff(cycles) <= 0):
            raise ValueError("Cycle order is not strictly increasing")
        if not rows.is_observed.isin([True, False]).all():
            raise ValueError("is_observed must contain booleans")
        metadata = batteries[(batteries.dataset == identity[0]) & (batteries.battery_id == identity[1])].iloc[0]
        reference = float(metadata.reference_capacity_ah)
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("Invalid reference capacity")
        ref = rows[rows.cycle == metadata.reference_cycle]
        if metadata.reference_method == "first_cycle":
            if len(ref) != 1 or metadata.reference_cycle != rows.cycle.iloc[0]:
                raise ValueError("First-cycle reference missing or inconsistent")
            np.testing.assert_allclose(ref.capacity_ah.iloc[0], reference, rtol=0, atol=1e-12)
        q = rows.capacity_ah.to_numpy(dtype=float)
        np.testing.assert_allclose(rows.soh.to_numpy(dtype=float), q / reference,
                                   rtol=1e-12, atol=1e-12, equal_nan=True)
        for value, flags in zip(q, rows.quality_flags):
            flags = set(str(flags).split(";"))
            if not np.isfinite(value) and "nonfinite_capacity" not in flags:
                raise ValueError("Unflagged nonfinite capacity")
            if np.isfinite(value) and value <= 0 and "nonpositive_capacity" not in flags:
                raise ValueError("Unflagged nonpositive capacity")


def observed_windows(rows, context_length=64, prediction_length=16, excluded_flags=None):
    """Use contiguous, observed cycles only; retain source cycle numbers and never interpolate."""
    if context_length < 1 or prediction_length < 1:
        raise ValueError("Window lengths must be positive")
    if len(rows[["dataset", "battery_id"]].drop_duplicates()) != 1:
        raise ValueError("Pass exactly one battery to observed_windows")
    if rows.cycle.duplicated().any() or not rows.cycle.is_monotonic_increasing:
        raise ValueError("Duplicate or unordered cycle numbers")
    excluded = set(EXCLUDED_FLAGS if excluded_flags is None else excluded_flags)
    excluded.add("capacity_outlier")
    cycles = rows.cycle.to_numpy(dtype=np.int64)
    values = rows.soh.to_numpy(dtype=np.float64)
    valid = (rows.is_observed.to_numpy(dtype=bool) & np.isfinite(values) &
             np.isfinite(rows.capacity_ah.to_numpy(dtype=float)) & (rows.capacity_ah.to_numpy(dtype=float) > 0) &
             rows.capacity_kind.isin(["full_discharge", "reference_test"]).to_numpy())
    valid &= np.array([not (set(str(f).split(";")) & excluded) for f in rows.quality_flags])
    total = context_length + prediction_length
    if len(rows) < total:
        return {"context": np.empty((0, context_length)), "target": np.empty((0, prediction_length)),
                "forecast_start": np.empty(0, dtype=np.int64),
                "target_cycle": np.empty((0, prediction_length), dtype=np.int64)}
    v = np.lib.stride_tricks.sliding_window_view(values, total)
    c = np.lib.stride_tricks.sliding_window_view(cycles, total)
    mask = np.lib.stride_tricks.sliding_window_view(valid, total).all(axis=1)
    mask &= (np.diff(c, axis=1) == 1).all(axis=1)
    if "segment_id" in rows:
        segments = np.lib.stride_tricks.sliding_window_view(rows.segment_id.to_numpy(), total)
        mask &= (segments == segments[:, :1]).all(axis=1)
    return {"context": v[mask, :context_length].copy(), "target": v[mask, context_length:].copy(),
            "forecast_start": c[mask, context_length].copy(), "target_cycle": c[mask, context_length:].copy()}


def load_bundle(directory):
    """Read a completed, checksummed bundle before passing it to a forecasting model."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["status"] != "complete":
        raise ValueError("Unsupported or incomplete SOH bundle")
    for name, expected in manifest["artifact_sha256"].items():
        path = directory / name
        if path.resolve().parent != directory.resolve() or sha256(path) != expected:
            raise ValueError(f"SOH artifact changed: {name}")
    soh = pd.read_csv(directory / "soh.csv", keep_default_na=False,
                      na_values={"capacity_ah": [""], "soh": [""]},
                      dtype={"cycle": "int64", "capacity_ah": "float64", "soh": "float64", "is_observed": "bool"})
    batteries = pd.read_csv(directory / "batteries.csv", keep_default_na=False)
    validate_tables(soh, batteries)
    return soh, batteries, manifest
