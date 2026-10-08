"""CALCE windows in chronological observation order, across recording files."""
import numpy as np
import pandas as pd


def observation_segments(rows):
    """File splits alone do not break a series; gaps/protocol changes still do.

    The cycle coordinate is the existing chronological record ordinal. Calendar
    rests are retained in timestamps but are not modeled as extra cycle steps.
    Invalid measurements stay in place and are excluded by observed_windows.
    """
    required = {"dataset", "battery_id", "cycle", "segment_id", "source_cycle",
                "source_start_time", "source_end_time", "protocol_id"}
    if not required.issubset(rows.columns):
        raise ValueError("CALCE observation-order windows require source chronology")
    if len(rows) == 0 or set(rows.dataset) != {"calce"} or rows.battery_id.nunique() != 1:
        raise ValueError("Pass one CALCE battery record")
    starts = pd.to_datetime(rows.source_start_time, errors="raise")
    ends = pd.to_datetime(rows.source_end_time, errors="raise")
    if (starts.isna().any() or ends.isna().any() or (ends < starts).any()
            or not starts.is_monotonic_increasing
            or (starts.iloc[1:].to_numpy() < ends.iloc[:-1].to_numpy()).any()):
        raise ValueError("CALCE recording times are missing, unordered or overlapping")
    cycle = rows.cycle.to_numpy()
    source_cycle = rows.source_cycle.to_numpy(dtype=float)
    if (not np.isfinite(source_cycle).all() or (source_cycle < 1).any()
            or (source_cycle != np.floor(source_cycle)).any()
            or (np.diff(cycle) <= 0).any()):
        raise ValueError("Invalid CALCE source cycle order")
    # A preprocessed bundle may already have observation-order segment IDs.
    # Use the preserved acquisition-file identity when applying this twice.
    acquisition = rows["source_segment_id"] if "source_segment_id" in rows else rows.segment_id
    same_file = acquisition.to_numpy()[1:] == acquisition.to_numpy()[:-1]
    breaks = (np.diff(cycle) != 1) | (same_file & (np.diff(source_cycle) != 1))
    breaks |= rows.protocol_id.to_numpy()[1:] != rows.protocol_id.to_numpy()[:-1]
    groups = np.r_[0, np.cumsum(breaks)]
    return np.asarray([f"{rows.battery_id.iloc[0]}:observation:{g}" for g in groups])
