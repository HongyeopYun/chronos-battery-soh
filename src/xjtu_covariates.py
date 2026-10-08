"""Past-cycle XJTU discharge features for SOH forecasting."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.nasa_covariates import FEATURES, fixed_time_mean


def discharge_features(record):
    time = np.asarray(record["relative_time_min"], dtype=float).ravel() * 60
    voltage = np.asarray(record["voltage_V"], dtype=float).ravel()
    current = np.asarray(record["current_A"], dtype=float).ravel()
    if time.shape != voltage.shape or time.shape != current.shape or not np.isfinite(current).all():
        raise ValueError("Invalid XJTU sensor alignment")
    indices = np.flatnonzero(current < -.1)
    if not len(indices):
        raise ValueError("No discharge segment")
    segments = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    if len(segments) != 1:
        raise ValueError("Expected exactly one contiguous discharge segment")
    ids = segments[0]
    delta = np.diff(time[ids])
    if np.any(delta < 0):
        raise ValueError("Discharge time moves backward")
    duplicates = int(np.sum(delta == 0))
    # Repeated timestamps have zero duration; retain the final sample at that time.
    ids = ids[np.r_[delta > 0, True]]
    return {FEATURES[0]: fixed_time_mean(time[ids], voltage[ids]),
            FEATURES[1]: fixed_time_mean(time[ids], current[ids]),
            "duplicate_discharge_timestamps": duplicates}


def extract_xjtu_features(raw_directory, data):
    rows = []
    for battery_id, battery in data.groupby("battery_id", sort=False):
        filename = battery_id.split("_", 1)[1] + ".mat"
        print(f"EXTRACT {filename}", flush=True)
        raw = loadmat(Path(raw_directory) / filename, simplify_cells=True, variable_names=["summary", "data"])
        records = raw["data"]
        records = [records] if isinstance(records, dict) else list(records)
        capacity = np.asarray(raw["summary"]["discharge_capacity_Ah"], dtype=float).ravel()
        if len(records) != len(capacity) or len(records) != len(battery):
            raise ValueError(f"XJTU cycle counts differ: {battery_id}")
        for cycle, record in enumerate(records, 1):
            observed = battery[battery.cycle == cycle]
            if len(observed) != 1 or str(observed.cycle_description.iloc[0]) != str(record["description"]).strip():
                raise ValueError(f"XJTU cycle/protocol alignment differs: {battery_id}/{cycle}")
            np.testing.assert_allclose(float(observed.soh.iloc[0]), capacity[cycle-1] / capacity[0],
                                       rtol=1e-10, atol=1e-12)
            rows.append({"battery_id": battery_id, "cycle": cycle, **discharge_features(record)})
    return pd.DataFrame(rows)
