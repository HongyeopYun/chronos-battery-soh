"""Cycle-aligned NASA voltage/current features available before forecast origin."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

FEATURES = ("voltage_mean_600s", "current_mean_600s")


def fixed_time_mean(time, values, seconds=600.):
    """Time-weighted mean over a fixed initial interval, not full discharge."""
    time, values = np.asarray(time, dtype=float).ravel(), np.asarray(values, dtype=float).ravel()
    if (time.shape != values.shape or len(time) < 2 or not np.isfinite(time).all()
            or not np.isfinite(values).all() or np.any(np.diff(time) <= 0)
            or not np.isfinite(seconds) or seconds <= 0):
        raise ValueError("Expected finite, aligned sensor samples and increasing time")
    elapsed = time - time[0]
    if elapsed[-1] < seconds:
        raise ValueError("Discharge does not cover the fixed feature interval")
    inside = elapsed < seconds
    x = np.append(elapsed[inside], seconds)
    y = np.append(values[inside], np.interp(seconds, elapsed, values))
    return float(np.trapezoid(y, x) / seconds)


def extract_nasa_features(raw_directory, data):
    rows = []
    for battery_id, battery in data.groupby("battery_id", sort=False):
        raw = loadmat(Path(raw_directory) / f"{battery_id}.mat", simplify_cells=True)[battery_id]["cycle"]
        raw = [raw] if isinstance(raw, dict) else list(raw)
        discharged = [(i + 1, c) for i, c in enumerate(raw) if str(c["type"]) == "discharge"]
        if len(discharged) != len(battery):
            raise ValueError(f"NASA discharge count differs: {battery_id}")
        reference = float(discharged[0][1]["data"]["Capacity"])
        for cycle, (source_index, record) in enumerate(discharged, 1):
            observed = battery[battery.cycle == cycle]
            if len(observed) != 1 or int(observed.source_record_index.iloc[0]) != source_index:
                raise ValueError(f"NASA source alignment differs: {battery_id}/{cycle}")
            sensor = record["data"]
            np.testing.assert_allclose(float(observed.soh.iloc[0]), float(sensor["Capacity"]) / reference,
                                       rtol=1e-10, atol=1e-12)
            rows.append({"battery_id": battery_id, "cycle": cycle,
                         FEATURES[0]: fixed_time_mean(sensor["Time"], sensor["Voltage_measured"]),
                         FEATURES[1]: fixed_time_mean(sensor["Time"], sensor["Current_measured"])})
    return pd.DataFrame(rows)


def covariate_inputs(windows, battery_features, names):
    """Select only past cycle IDs; do not expose targets or future sensor values."""
    features = battery_features.set_index("cycle")
    if not features.index.is_unique:
        raise ValueError("Duplicate sensor feature cycle")
    tasks = []
    for context, cycles, start in zip(windows["context"], windows["context_cycle"], windows["forecast_start"]):
        if cycles.max() >= start or len(context) != len(cycles):
            raise ValueError("Covariate context crosses forecast origin")
        task = {"target": np.asarray(context, dtype=np.float32)}
        if names:
            past = features.loc[cycles, list(names)]
            if not np.isfinite(past.to_numpy()).all():
                raise ValueError("Missing/nonfinite past covariates")
            task["past_covariates"] = {name: past[name].to_numpy(dtype=np.float32) for name in names}
        tasks.append(task)
    return tasks
