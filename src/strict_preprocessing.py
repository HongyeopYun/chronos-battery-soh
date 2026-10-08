"""Raw-trace quality checks for SOH; retain measurements and original cycle IDs.

No model predictions, future SOH values, smoothing, or interpolation are used.
Dataset-specific decisions are applied after extracting auditable raw evidence.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.soh_data import EXCLUDED_FLAGS

BUNDLES = {
    "nasa_aging": "Nasa/nasa_aging",
    "xjtu_2c": "XJTU/2C", "xjtu_3c": "XJTU/3C",
    "calce_cs2_1c": "Calce/calce_cs2_1c",
    "mit_batch1": "MIT/mit_batch1", "mit_batch2": "MIT/mit_batch2",
    "mit_batch3": "MIT/mit_batch3",
}
RAW_FOLDERS = {"nasa": "Nasa", "xjtu": "XJTU", "calce": "Calce", "mit": "MIT"}
FEATURE_VERSION = 1
# These are verification failures, not proof that the capacity itself is wrong.
UNVERIFIED_REASONS = {"capacity_integral_unverified", "preceding_charge_unverified",
                      "charge_trace_invalid", "trace_missing", "trace_nonfinite", "trace_length_mismatch"}


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def trace_features(current, voltage, seconds, *, charge=None, counter=None,
                   temperature=None, cutoff=None, first_step_seconds=None):
    """Summarize the principal discharge and its preceding charge.

    XJTU time resets at step transitions: only time within the discharge is
    required to be monotone. Tiny diagnostic pulses do not define a discharge.
    Counter conventions and unit conversion are handled by dataset readers.
    """
    i, v, t = [np.asarray(x, dtype=float).ravel() for x in (current, voltage, seconds)]
    out = {"n_samples": len(i), "trace_missing": len(i) < 3,
           "trace_nonfinite": False, "trace_length_mismatch": False}
    if len(i) != len(v) or len(i) != len(t):
        out["trace_length_mismatch"] = True
        return out
    out["trace_nonfinite"] = not (np.isfinite(i).all() and np.isfinite(v).all() and np.isfinite(t).all())
    if out["trace_missing"] or out["trace_nonfinite"]:
        return out
    out.update(voltage_min_v=float(v.min()), voltage_max_v=float(v.max()))
    # At least one substantial load sample is necessary; extend through its CV tail.
    candidates = [(a, b) for a, b in runs(i < -.005) if np.min(i[a:b]) < -.1]
    out["discharge_runs"] = len(candidates)
    if not candidates:
        out["no_discharge_trace"] = True
        return out
    start, stop = max(candidates, key=lambda ab: ab[1] - ab[0])
    end = stop - 1
    di, dv, dt = i[start:stop], v[start:stop], t[start:stop]
    delta = np.diff(dt)
    out.update(discharge_samples=len(di), discharge_start_voltage_v=float(dv[0]),
               discharge_end_voltage_v=float(dv[-1]), discharge_min_voltage_v=float(dv.min()),
               discharge_end_current_a=float(di[-1]), discharge_cc_current_a=float(np.quantile(di, .1)),
               discharge_duration_s=float(np.maximum(delta, 0).sum()),
               discharge_time_backsteps=int((delta < -1e-6).sum()),
               discharge_max_gap_s=float(delta.max()) if len(delta) else 0.,
               discharge_median_gap_s=float(np.median(delta[delta > 0])) if (delta > 0).any() else 0.)
    last = end
    if cutoff is not None:
        crossing = np.flatnonzero(dv <= cutoff)
        out["capacity_cutoff_reached"] = bool(len(crossing))
        if len(crossing):
            last = start + int(crossing[0])
    # Trapezoidal integral, plus the recorded elapsed portion before the first
    # sample for CALCE. Other datasets tolerate the small unsampled edge.
    integrand = -i[start:last + 1]
    q = float(np.sum(.5 * (integrand[1:] + integrand[:-1]) * np.maximum(np.diff(t[start:last + 1]), 0)) / 3600)
    if first_step_seconds is not None:
        elapsed = float(np.asarray(first_step_seconds)[start])
        if np.isfinite(elapsed) and elapsed >= 0:
            q += -float(i[start]) * elapsed / 3600
    out["discharge_integral_ah"] = q
    if counter is not None:
        c = np.asarray(counter, dtype=float).ravel()
        if len(c) != len(i) or not np.isfinite(c).all():
            out["counter_invalid"] = True
        else:
            baseline = c[start - 1] if start else c[start]
            out.update(discharge_counter_ah=float(c[end] - baseline),
                       discharge_counter_backsteps=int((np.diff(c[start:stop]) < -1e-5).sum()))
    if temperature is not None:
        temp = np.asarray(temperature, dtype=float).ravel()
        finite = temp[np.isfinite(temp)]
        if len(finite):
            out.update(temperature_min_c=float(finite.min()), temperature_max_c=float(finite.max()))
    ci, cv = (i[:start], v[:start]) if charge is None else charge
    ci, cv = np.asarray(ci, dtype=float).ravel(), np.asarray(cv, dtype=float).ravel()
    if len(ci) != len(cv) or not (np.isfinite(ci).all() and np.isfinite(cv).all()):
        out["charge_trace_invalid"] = True
        out["charge_samples"] = 0
        return out
    pos = np.flatnonzero(ci > .005)
    out["charge_samples"] = len(pos)
    if len(pos):
        out.update(charge_end_voltage_v=float(cv[pos[-1]]), charge_end_current_a=float(ci[pos[-1]]),
                   charge_max_voltage_v=float(cv[pos].max()), charge_peak_current_a=float(ci[pos].max()))
    return out


def extract_nasa(raw, data, batteries):
    records = []
    for meta in batteries.itertuples():
        cycles = loadmat(raw / meta.source_file, simplify_cells=True)[meta.battery_id]["cycle"]
        lookup = {int(r.source_record_index): r for r in data[data.battery_id == meta.battery_id].itertuples()}
        previous_charge, previous_end, charge_invalid = None, None, False
        for ordinal, cycle in enumerate(cycles, 1):
            d = cycle["data"]
            if cycle["type"] == "charge":
                previous_charge = (d["Current_measured"], d["Voltage_measured"])
                ct = np.asarray(d["Time"], dtype=float)
                charge_invalid = not np.isfinite(ct).all() or bool((np.diff(ct) < 0).any())
            if ordinal in lookup:
                row = lookup[ordinal]
                f = trace_features(d["Current_measured"], d["Voltage_measured"], d["Time"],
                                   charge=previous_charge if previous_charge is not None else ([], []),
                                   temperature=d["Temperature_measured"], cutoff=2.7)
                f["charge_trace_invalid"] = f.get("charge_trace_invalid", False) or charge_invalid
                stamp = np.asarray(cycle["time"]).ravel()
                start = pd.Timestamp(year=int(stamp[0]), month=int(stamp[1]), day=int(stamp[2]),
                                     hour=int(stamp[3]), minute=int(stamp[4])) + pd.Timedelta(seconds=float(stamp[5]))
                f.update(battery_id=row.battery_id, cycle=int(row.cycle),
                         raw_capacity_ah=float(d["Capacity"]), source_record_index=ordinal,
                         source_file=meta.source_file, discharge_start_time=start.isoformat(),
                         inter_discharge_rest_h=(start - previous_end).total_seconds() / 3600 if previous_end is not None else np.nan)
                previous_end = start + pd.Timedelta(seconds=float(np.asarray(d["Time"])[-1]))
                previous_charge = None  # A charge may not silently serve two discharges.
                charge_invalid = False
                records.append(f)
        print(f"TRACE {meta.battery_id}: {len(lookup)} cycles", flush=True)
    return pd.DataFrame(records)


def extract_xjtu(raw, data, batteries):
    records = []
    for meta in batteries.itertuples():
        raw_data = loadmat(raw / meta.source_file, simplify_cells=True)
        cycles = raw_data["data"]
        q = np.asarray(raw_data["summary"]["discharge_capacity_Ah"]).ravel()
        rows = data[data.battery_id == meta.battery_id]
        if len(rows) != len(cycles) or len(q) != len(rows):
            raise ValueError(f"Raw cycle mapping changed: {meta.battery_id}")
        for row in rows.itertuples():
            d = cycles[int(row.cycle) - 1]
            f = trace_features(d["current_A"], d["voltage_V"], np.asarray(d["relative_time_min"]) * 60,
                               counter=d["capacity_Ah"], temperature=d["temperature_C"])
            # The XJTU capacity counter resets at the start of each step.
            ix = np.flatnonzero(np.asarray(d["current_A"]) < -.1)
            if len(ix):
                f["discharge_counter_ah"] = float(np.asarray(d["capacity_Ah"])[ix[-1]])
            f.update(battery_id=row.battery_id, cycle=int(row.cycle), source_file=meta.source_file,
                     raw_capacity_ah=float(q[int(row.cycle) - 1]))
            records.append(f)
        print(f"TRACE {meta.battery_id}: {len(rows)} cycles", flush=True)
    return pd.DataFrame(records)


def extract_mit(raw, data, batteries):
    records = []
    with h5py.File(raw / batteries.source_file.iloc[0], "r") as file:
        batch = file["batch"]
        for meta in batteries.itertuples():
            cell = int(meta.source_cell_index_zero_based)
            cycles = file[batch["cycles"][cell, 0]]
            summary = file[batch["summary"][cell, 0]]
            raw_cycles = np.asarray(summary["cycle"][()]).ravel()
            q = np.asarray(summary["QDischarge"][()]).ravel()
            refs = {k: cycles[k][()].ravel() for k in ["I", "V", "t", "Qd", "T"]}
            rows = data[data.battery_id == meta.battery_id]
            if len(rows) != len(raw_cycles) or any(len(r) != len(rows) for r in refs.values()):
                raise ValueError(f"Raw cycle mapping changed: {meta.battery_id}")
            for idx, row in enumerate(rows.itertuples()):
                if row.source_cycle != raw_cycles[idx]:
                    raise ValueError("MIT summary/source cycle mismatch")
                d = {k: file[r[idx]][()].ravel() for k, r in refs.items()}
                # These release files store I in nominal-capacity units: the
                # CC discharge plateau is 4, and Qd/(integral I dt) is ~1.1.
                # Retain this explicitly; never overwrite the native QDischarge.
                f = trace_features(d["I"] * 1.1, d["V"], d["t"] * 60, counter=d["Qd"], temperature=d["T"])
                f.update(battery_id=row.battery_id, cycle=int(row.cycle), source_cycle=int(row.source_cycle),
                         source_file=meta.source_file, raw_capacity_ah=float(q[idx]), current_scale_to_a=1.1)
                if len(d["Qd"]):
                    f["discharge_counter_ah"] = float(np.max(d["Qd"]))
                records.append(f)
            print(f"TRACE {meta.battery_id}: {len(rows)} cycles", flush=True)
    return pd.DataFrame(records)


def extract_calce(raw, data, batteries):
    records = []
    for meta in batteries.itertuples():
        rows = data[data.battery_id == meta.battery_id].copy()
        rows["primary_source_file"] = rows.source_file.str.split(";")
        exported = {}
        for name, wanted in rows.explode("primary_source_file").groupby("primary_source_file", sort=False):
            with pd.ExcelFile(raw / meta.battery_id / name) as excel:
                sheets = [s for s in excel.sheet_names if s.startswith("Channel")]
                if len(sheets) != 1:
                    raise ValueError(f"Unexpected channel layout: {name}")
                frame = pd.read_excel(excel, sheet_name=sheets[0])
            by_cycle = dict(tuple(frame.groupby("Cycle_Index", sort=False)))
            if set(by_cycle) != set(wanted.source_cycle):
                raise ValueError(f"Raw CALCE cycles not fully covered: {name}")
            for row in wanted.itertuples():
                d = by_cycle[int(row.source_cycle)]
                if (pd.Timestamp(d["Date_Time"].iloc[0]) != pd.Timestamp(row.source_start_time)
                        or pd.Timestamp(d["Date_Time"].iloc[-1]) != pd.Timestamp(row.source_end_time)):
                    raise ValueError(f"CALCE timestamp mapping mismatch: {name}/{row.source_cycle}")
                f = trace_features(d["Current(A)"], d["Voltage(V)"], d["Test_Time(s)"],
                                   counter=d["Discharge_Capacity(Ah)"], first_step_seconds=d["Step_Time(s)"])
                dis = np.flatnonzero(d["Current(A)"].to_numpy() < -.1)
                before = d.iloc[:dis[0]] if len(dis) else d.iloc[:0]
                cv = before[(before["Step_Index"] == 4) & (before["Current(A)"] > .005)]
                f.update(battery_id=row.battery_id, cycle=int(row.cycle), source_cycle=int(row.source_cycle),
                         source_file=name, charge_cv_step_present=bool(len(cv)), charge_cv_samples=len(cv),
                         raw_capacity_ah=float(f.get("discharge_counter_ah", np.nan)),
                         raw_time_backsteps=int((np.diff(d["Test_Time(s)"]) < -1e-6).sum()),
                         raw_start_time=str(d["Date_Time"].iloc[0]), raw_end_time=str(d["Date_Time"].iloc[-1]))
                key = int(row.cycle)
                if key in exported:
                    previous = exported[key]
                    # A duplicate export must also agree on charge and waveform
                    # diagnostics, not merely on the reported capacity.
                    numeric = [k for k, val in f.items() if isinstance(val, (int, float, np.number, bool))]
                    same = all(np.isclose(f[k], previous.get(k, np.nan), rtol=1e-8, atol=1e-6, equal_nan=True) for k in numeric)
                    previous["duplicate_trace_conflict"] = not same or previous.get("duplicate_trace_conflict", False)
                    previous["duplicate_exports_checked"] = previous.get("duplicate_exports_checked", 0) + 1
                else:
                    exported[key] = f
            print(f"TRACE {meta.battery_id}/{name}: {len(wanted)} cycles", flush=True)
        records.extend(exported.values())
    return pd.DataFrame(records)


def truth(row, key):
    value = row.get(key, False)
    return bool(value) if pd.notna(value) else False


def classify_trace(dataset, row, cycle):
    """Return objective exclusion reasons and non-excluding review notes.

    Tolerances are measurement tolerances, not fitted to forecast scores.
    An unverifiable or nonstandard test is not asserted to be permanent aging.
    """
    bad, review = [], []
    for key in ["trace_missing", "trace_length_mismatch", "trace_nonfinite", "no_discharge_trace", "counter_invalid", "duplicate_trace_conflict", "charge_trace_invalid"]:
        if truth(row, key):
            bad.append(key)
    if bad:
        return bad, review
    get = lambda key: float(row.get(key, np.nan))
    if get("discharge_samples") < 3:
        bad.append("discharge_trace_too_short")
    if get("discharge_time_backsteps") > 0 or get("raw_time_backsteps") > 0:
        bad.append("nonmonotone_discharge_time")
    if get("discharge_counter_backsteps") > 0:
        bad.append("discharge_counter_reset")
    if get("voltage_min_v") < .5 or get("voltage_max_v") > (3.9 if dataset == "mit" else 4.5):
        bad.append("voltage_out_of_range")
    if get("discharge_max_gap_s") > max(60., 10 * get("discharge_median_gap_s")):
        review.append("sparse_discharge_sampling")
    if get("temperature_min_c") < -10 or get("temperature_max_c") > 80:
        review.append("temperature_trace_review")
    cutoff = {"nasa": 2.7, "calce": 2.7, "xjtu": 2.5, "mit": 2.0}[dataset]
    if get("discharge_min_voltage_v") > cutoff + .05:
        bad.append("discharge_cutoff_not_reached")
    if dataset != "nasa" and get("discharge_end_voltage_v") > cutoff + .10:
        bad.append("abnormal_discharge_endpoint")
    charge_v = 3.6 if dataset == "mit" else 4.2
    charge_i = {"nasa": .12, "calce": .075, "xjtu": .15, "mit": .10}[dataset]
    if get("charge_samples") == 0 or not np.isfinite(get("charge_end_voltage_v")):
        bad.append("preceding_charge_unverified")
    elif get("charge_end_voltage_v") < charge_v - .05 or get("charge_end_current_a") > charge_i:
        bad.append("charge_termination_not_met")
    if dataset == "calce" and not truth(row, "charge_cv_step_present"):
        bad.append("missing_cv_charge_step")
    expected_i = {"nasa": 2., "calce": 1.1, "xjtu": .4 if cycle == 1 else 2., "mit": 4.4}[dataset]
    if abs(get("discharge_cc_current_a") + expected_i) > max(.15, expected_i * .15):
        bad.append("nonstandard_discharge_current")
    # NASA native Capacity is defined to 2.7 V, not to the lower physical cutoff.
    q, integrated, counted = [get(k) for k in ["raw_capacity_ah", "discharge_integral_ah", "discharge_counter_ah"]]
    if np.isfinite(q) and np.isfinite(counted) and abs(q - counted) > max(.005, .01 * abs(q)):
        bad.append("summary_counter_mismatch")
    if np.isfinite(q) and np.isfinite(integrated) and abs(q - integrated) > max(.02, .03 * abs(q)):
        # Export sampling can make an integral approximate. Quarantine rather
        # than silently replacing reported capacity with a reconstructed value.
        bad.append("capacity_integral_unverified")
    return sorted(set(bad)), sorted(set(review))


def apply_quality(data, features):
    keys = ["battery_id", "cycle"]
    if features.duplicated(keys).any() or len(features) != len(data):
        raise ValueError("Raw features must map one-to-one to all SOH records")
    aligned = data[keys].merge(features, on=keys, how="left", validate="one_to_one", indicator=True)
    if not aligned._merge.eq("both").all():
        raise ValueError("Unverified SOH cycle")
    out = data.reset_index(drop=True).copy()
    reasons, notes, eligible, statuses = [], [], [], []
    for row, f in zip(out.itertuples(), aligned.to_dict("records")):
        bad, review = classify_trace(row.dataset, f, int(row.cycle))
        old = set(str(row.quality_flags).split(";")) & EXCLUDED_FLAGS
        if not row.is_observed or row.capacity_kind not in {"full_discharge", "reference_test"}:
            old.add("previously_unusable_measurement")
        if not np.isfinite(row.capacity_ah) or row.capacity_ah <= 0:
            old.add("invalid_capacity")
        q_raw = f.get("raw_capacity_ah", np.nan)
        if np.isfinite(q_raw) and np.isfinite(row.capacity_ah) and abs(q_raw - row.capacity_ah) > .005:
            bad.append("extracted_capacity_mismatch")
        combined = old | set(bad)
        reasons.append(";".join(sorted(combined)))
        notes.append(";".join(review))
        eligible.append(not old and not bad)
        statuses.append("excluded" if combined - UNVERIFIED_REASONS else
                        "quarantined" if combined else "accepted_with_note" if review else "accepted")
    out["quality_eligible"] = eligible
    out["quality_status"] = statuses
    out["exclusion_reasons"] = reasons
    out["review_notes"] = notes
    out["quality_flags"] = [";".join(filter(None, [str(f), "strict_quality_excluded" if not ok else ""]))
                            for f, ok in zip(out.quality_flags, eligible)]
    # Shape is a review annotation only, evaluated from earlier observations.
    out["past_soh_change"] = out.groupby("battery_id", sort=False).soh.diff()
    out["shape_review"] = out.past_soh_change.abs() > .03
    return out
