"""Essential SOH extraction for the NASA, MIT and CALCE files in data/raw."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.soh_data import (BATTERY_REQUIRED, SCHEMA_VERSION, SOH_COLUMNS, load_bundle,
                          observed_windows, sha256, validate_tables, write_json)

MIT_SOURCE = "https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation"
MIT_CONTINUATIONS = {7: "b1c0", 8: "b1c1", 9: "b1c2", 15: "b1c3", 16: "b1c4"}
MIT_NOISY = {2, 23, 32, 37, 42, 43}
CALCE_SOURCE = "https://web.calce.umd.edu/batteries/data/"
NASA_SOURCE = "https://data.nasa.gov/dataset/li-ion-battery-aging-datasets"


def capacity_flags(value):
    if not np.isfinite(value):
        return "nonfinite_capacity"
    return "nonpositive_capacity" if value <= 0 else ""


def make_frame(dataset, battery_id, cycles, capacity, reference, protocol, **extra):
    q = np.asarray(capacity, dtype=np.float64)
    cycles = np.asarray(cycles)
    if not len(q) or len(q) != len(cycles):
        raise ValueError(f"Empty or mismatched capacity/cycle arrays: {battery_id}")
    if not np.isfinite(cycles).all() or np.any(cycles < 1) or np.any(cycles != np.floor(cycles)) or np.any(np.diff(cycles) <= 0):
        raise ValueError(f"Invalid, duplicate or unordered source cycles: {battery_id}")
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError(f"Invalid explicit reference capacity: {battery_id}")
    return pd.DataFrame({"dataset": dataset, "battery_id": battery_id, "cycle": cycles.astype(np.int64),
                         "capacity_ah": q, "soh": q / reference, "capacity_kind": "full_discharge",
                         "is_observed": True, "quality_flags": [capacity_flags(v) for v in q],
                         "protocol_id": protocol, "cycle_description": protocol, **extra})


def metadata(dataset, battery_id, batch, reference, reference_cycle, method, source_files, hashes, cycle_source, **extra):
    return {"dataset": dataset, "battery_id": battery_id, "batch": batch,
            "reference_method": method, "reference_cycle": int(reference_cycle),
            "reference_capacity_ah": float(reference), "source_file": ";".join(source_files),
            "source_sha256": ";".join(hashes[name] for name in source_files),
            "cycle_number_source": cycle_source, **extra}


def read_mit(raw_directory):
    """Keep all 140 source records; link the five continued physical cells explicitly."""
    groups = []
    originals = {}
    for batch_number, expected in [(1, 46), (2, 48), (3, 46)]:
        path = Path(raw_directory) / f"MIT_Batch_{batch_number}.mat"
        print("READ", path.name, flush=True)
        hashes = {path.name: sha256(path)}
        frames, batteries = [], []
        with h5py.File(path, "r") as f:
            batch = f["batch"]
            if batch["summary"].shape != (expected, 1):
                raise ValueError(f"Unexpected MIT source count: {path.name}")
            for index in range(expected):
                summary = f[batch["summary"][index, 0]]
                q = np.asarray(summary["QDischarge"], dtype=np.float64).ravel()
                cycles = np.asarray(summary["cycle"], dtype=np.float64).ravel()
                policy = "".join(chr(int(c)) for c in np.asarray(f[batch["policy_readable"][index, 0]]).ravel())
                battery_id = f"b{batch_number}c{index}"
                continuation = MIT_CONTINUATIONS.get(index) if batch_number == 2 else None
                if batch_number == 1:
                    # All 46 published Batch-1 summaries contain a zero placeholder at cycle 1.
                    if len(q) < 2 or q[0] != 0 or cycles[0] != 1 or cycles[1] != 2:
                        raise ValueError(f"Unexpected Batch-1 initial measurement: {battery_id}")
                    reference, ref_cycle, method = float(q[1]), 2, "first_observed_cycle"
                    originals[battery_id] = (reference, len(q))
                elif continuation:
                    reference, _ = originals[continuation]
                    ref_cycle, method = 2, "original_battery_initial_capacity"
                else:
                    reference, ref_cycle, method = float(q[0]), int(cycles[0]), "first_cycle"
                physical = continuation or battery_id
                offset = originals[continuation][1] if continuation else 0
                frame = make_frame("mit", battery_id, cycles + offset, q, reference, policy,
                                   source_cycle=cycles, source_file=path.name, segment_id=battery_id)
                if batch_number == 1:
                    frame.loc[0, "is_observed"] = False
                    frame.loc[0, "quality_flags"] += ";missing_initial_measurement"
                collection_issue = batch_number == 3 and index in MIT_NOISY
                if collection_issue:
                    frame["quality_flags"] = frame.quality_flags.map(lambda flags: ";".join(filter(None, [flags, "known_collection_issue"])))
                battery = metadata("mit", battery_id, f"batch{batch_number}", reference, ref_cycle, method,
                    [path.name], hashes, "summary.cycle_with_documented_continuation_offset" if continuation else "summary.cycle",
                    physical_battery_id=physical, continuation_of=continuation or "", source_cell_index_zero_based=index,
                    nominal_capacity_ah=1.1, chemistry="LFP", source_charge_policy=policy,
                    known_collection_issue=collection_issue, source_documentation=MIT_SOURCE)
                if continuation:
                    battery["reference_source_file"] = "MIT_Batch_1.mat"
                    battery["reference_source_sha256"] = groups[0][3]["MIT_Batch_1.mat"]
                frames.append(frame)
                batteries.append(battery)
        groups.append((f"mit_batch{batch_number}", pd.concat(frames, ignore_index=True), pd.DataFrame(batteries), hashes))
    return groups


def read_nasa(raw_directory):
    frames, batteries, hashes = [], [], {}
    for battery_id in ["B0005", "B0006", "B0007", "B0018"]:
        path = Path(raw_directory) / f"{battery_id}.mat"
        print("READ", path.name, flush=True)
        hashes[path.name] = sha256(path)
        raw = loadmat(path, simplify_cells=True)[battery_id]["cycle"]
        raw = [raw] if isinstance(raw, dict) else list(raw)
        discharged = [(i + 1, c) for i, c in enumerate(raw) if str(c["type"]) == "discharge"]
        q = [float(c["data"]["Capacity"]) for _, c in discharged]
        reference = q[0]
        protocol = "nasa_reported_discharge_capacity_to_2.7v"
        frame = make_frame("nasa", battery_id, np.arange(1, len(q) + 1), q, reference, protocol,
                           source_record_index=[i for i, _ in discharged], source_file=path.name, segment_id=battery_id)
        frames.append(frame)
        batteries.append(metadata("nasa", battery_id, "aging", reference, 1, "first_cycle", [path.name], hashes,
            "discharge_record_ordinal_1based", nominal_capacity_ah=2., capacity_field="cycle.data.Capacity",
            capacity_measurement_cutoff_v=2.7, source_documentation=NASA_SOURCE))
    return [("nasa_aging", pd.concat(frames, ignore_index=True), pd.DataFrame(batteries), hashes)]


def calce_file_date(path):
    month, day, year = map(int, path.stem.rsplit("_", 3)[1:])
    return datetime(2000 + year, month, day)


def calce_cycle_record(rows, path, sheet):
    """Use the increase of the logged cumulative discharge counter, including its first increment."""
    current = rows["Current(A)"].to_numpy(dtype=float)
    counter = rows["Discharge_Capacity(Ah)"].to_numpy(dtype=float)
    positions = np.flatnonzero(current < -.1)
    flags = []
    kind = "full_discharge"
    if len(positions):
        first, last = positions[0], positions[-1]
        baseline = counter[first - 1] if first else counter[first]
        q = counter[last] - baseline
        end_voltage = float(rows["Voltage(V)"].iloc[last])
        if first == 0 or not np.isfinite(end_voltage) or end_voltage > 2.75:
            flags.append("incomplete_discharge")
            kind = "partial_discharge"
        if np.any(np.diff(counter[positions]) < -1e-6) or q < 0:
            flags.append("discharge_counter_reset")
            kind = "unknown"
        if not np.isfinite(current[positions]).all() or abs(np.median(current[positions]) + 1.1) > .15:
            flags.append("unexpected_protocol")
            kind = "unknown"
    else:
        q, end_voltage = 0., np.nan
        flags.append("no_discharge_record")
        kind = "unknown"
    if capacity_flags(q):
        flags.append(capacity_flags(q))
    timestamps = pd.to_datetime(rows["Date_Time"])
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"Invalid acquisition times: {path.name}")
    return {"capacity_ah": q, "capacity_kind": kind, "quality_flags": ";".join(flags),
            "is_observed": bool(len(positions)),
            "source_file": path.name, "source_sheet": sheet, "source_cycle": int(rows["Cycle_Index"].iloc[0]),
            "segment_id": path.stem, "source_start_time": timestamps.iloc[0].isoformat(),
            "source_end_time": timestamps.iloc[-1].isoformat(), "discharge_end_voltage_v": end_voltage}


def deduplicate_calce_records(records):
    unique = {}
    duplicates = 0
    for record in sorted(records, key=lambda r: r["source_start_time"]):
        key = (record["source_start_time"], record["source_end_time"], record["source_cycle"])
        if key in unique:
            original = unique[key]
            for field in ["capacity_ah", "discharge_end_voltage_v"]:
                if not np.isclose(original[field], record[field], rtol=0, atol=1e-12, equal_nan=True):
                    raise ValueError("Conflicting duplicate CALCE capacity records")
            for field in ["capacity_kind", "quality_flags", "is_observed"]:
                if original[field] != record[field]:
                    raise ValueError("Conflicting duplicate CALCE discharge records")
            original["source_file"] += ";" + record["source_file"]
            duplicates += 1
        else:
            unique[key] = dict(record)
    result = list(unique.values())
    if any(a["source_end_time"] > b["source_start_time"] for a, b in zip(result, result[1:])):
        raise ValueError("Overlapping CALCE acquisition records are not identical")
    return result, duplicates


def read_calce(raw_directory):
    frames, batteries, hashes = [], [], {}
    columns = ["Date_Time", "Cycle_Index", "Current(A)", "Voltage(V)", "Discharge_Capacity(Ah)"]
    for battery_id in ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]:
        directory = Path(raw_directory) / battery_id
        paths = sorted(directory.glob("*.xlsx"), key=calce_file_date)
        if not paths:
            raise ValueError(f"Missing CALCE sources: {battery_id}")
        records = []
        source_names = []
        for path in paths:
            print("READ", battery_id, path.name, flush=True)
            name = f"{battery_id}/{path.name}"
            source_names.append(name)
            hashes[name] = sha256(path)
            with pd.ExcelFile(path, engine="openpyxl") as workbook:
                sheets = [s for s in workbook.sheet_names if s.startswith("Channel_")]
                if len(sheets) != 1:
                    raise ValueError(f"Expected one CALCE acquisition channel: {path.name}")
                sheet = sheets[0]
                data = pd.read_excel(workbook, sheet_name=sheet, usecols=columns)
            cycles = data["Cycle_Index"].to_numpy(dtype=float)
            if not len(cycles) or not np.isfinite(cycles).all() or np.any(cycles < 1) or np.any(cycles != np.floor(cycles)) or np.any(np.diff(cycles) < 0):
                raise ValueError(f"Invalid CALCE cycle index: {path.name}")
            for _, rows in data.groupby("Cycle_Index", sort=False):
                records.append(calce_cycle_record(rows, path, sheet))
        records, duplicates = deduplicate_calce_records(records)
        usable = [i for i, r in enumerate(records) if r["capacity_kind"] == "full_discharge" and np.isfinite(r["capacity_ah"]) and r["capacity_ah"] > 0 and not r["quality_flags"]]
        if not usable:
            raise ValueError(f"No complete CALCE reference discharge: {battery_id}")
        ref_index = usable[0]
        reference = records[ref_index]["capacity_ah"]
        protocol = "cccv_0.5c_charge_1c_discharge_2.7v"
        frame = make_frame("calce", battery_id, np.arange(1, len(records) + 1),
                           [r["capacity_ah"] for r in records], reference, protocol)
        for key in records[0]:
            if key != "capacity_ah":
                frame[key] = [r[key] for r in records]
        frames.append(frame)
        batteries.append(metadata("calce", battery_id, "cs2_1c", reference, ref_index + 1,
            "first_complete_observed_discharge", source_names, hashes, "chronological_file_cycle_ordinal_1based",
            nominal_capacity_ah=1.1, charge_rate_c=.5, discharge_rate_c=1., discharge_cutoff_v=2.7,
            chemistry="LCO", source_documentation=CALCE_SOURCE,
            duplicate_cycle_exports_removed=duplicates,
            recording_boundaries="forecast_windows_do_not_cross_source_files"))
    return [("calce_cs2_1c", pd.concat(frames, ignore_index=True), pd.DataFrame(batteries), hashes)]


def save_bundle(group, data, batteries, hashes, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validate_tables(data, batteries)
    audit_rows = []
    for battery_id, rows in data.groupby("battery_id", sort=False):
        q = rows.capacity_ah.to_numpy(dtype=float)
        windows = observed_windows(rows)
        valid = rows.is_observed & np.isfinite(q) & (q > 0) & rows.capacity_kind.eq("full_discharge")
        valid &= ~rows.quality_flags.str.contains("unexpected_protocol|known_collection_issue", regex=True)
        audit_rows.append({"battery_id": battery_id, "n_cycles": len(rows),
            "n_nonfinite_capacity": int((~np.isfinite(q)).sum()), "n_nonpositive_capacity": int((q <= 0).sum()),
            "n_excluded_measurements": int((~valid).sum()), "n_prediction_windows": len(windows["target"]),
            "first_cycle": int(rows.cycle.iloc[0]), "last_cycle": int(rows.cycle.iloc[-1]),
            "last_soh": float(rows.soh.iloc[-1]), "status": "passed_with_exclusions" if not valid.all() else "passed"})
    audit = pd.DataFrame(audit_rows)
    manifest = {"schema_version": SCHEMA_VERSION, "status": "running", "group": group,
                "source_files": hashes, "n_batteries": len(batteries), "n_rows": len(data),
                "scope": "capacity extraction and essential checks; no model forecasting or training",
                "reference_policy": "explicit per-dataset initial observed reference; see batteries.csv",
                "interpolation": "none", "smoothing": "none", "unique_observations_dropped": 0,
                "duplicate_cycle_exports_removed": int(batteries.get("duplicate_cycle_exports_removed", pd.Series(dtype=int)).sum()),
                "implementation_sha256": {"src/battery_preprocessing.py": sha256(Path(__file__)),
                                          "src/soh_data.py": sha256(Path(__file__).with_name("soh_data.py"))}}
    write_json(output / "manifest.json", manifest)
    for name, frame in [("soh.csv", data), ("batteries.csv", batteries), ("audit.csv", audit)]:
        frame.to_csv(output / name, index=False, float_format="%.17g", lineterminator="\n")
    write_json(output / "schema.json", {"version": SCHEMA_VERSION, "key": ["dataset", "battery_id", "cycle"],
        "required_soh_columns": SOH_COLUMNS, "battery_required_columns": BATTERY_REQUIRED,
        "optional_columns": [c for c in data if c not in SOH_COLUMNS],
        "segment_id": "windows may not cross segment boundaries", "capacity_ah": "Ah", "soh": "Q/Q_ref fraction"})
    manifest.update(status="complete", artifact_sha256={name: sha256(output / name) for name in ["soh.csv", "batteries.csv", "audit.csv", "schema.json"]})
    write_json(output / "manifest.json", manifest)
    loaded, _, _ = load_bundle(output)
    np.testing.assert_allclose(loaded.soh, data.soh, rtol=1e-12, atol=1e-12)
    return audit, manifest
