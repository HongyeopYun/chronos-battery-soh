"""Versioned cycle-level adaptations of the pinned BatteryML/BatteryLife rules.

Raw rows and cycle coordinates are retained. The centered CALCE rule is an
explicit retrospective view, never the default forecasting mask.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import medfilt

from src.soh_data import load_bundle

ML_REVISION = '2861ae3b8c79938c7fc8e6fe9986b799ca71c7dd'
LIFE_REVISION = '3b17bb53202e10711deae0448c91f84144bb3dd9'
MIT_CONTINUATIONS = dict(zip(['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16'],
                             ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']))
MIT_BAD_RAW_INDEX = {'b1c0': 11, 'b2c12': 252, 'b2c44': 247, 'b1c18': 39}
MIT_COLLECTION_EXCLUSIONS = {'b3c2', 'b3c23', 'b3c32', 'b3c37', 'b3c42', 'b3c43'}
MIT_NOT_REACHED_80 = {'b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22'}
PROFILES = ('forecast_safe', 'retrospective')


def integrate_capacity(current, seconds):
    """BatteryML's right-endpoint rectangle rule; fail on invalid chronology."""
    i, t = (np.asarray(a, dtype=float).ravel() for a in (current, seconds))
    if len(i) != len(t) or not len(i) or not np.isfinite(i).all() or not np.isfinite(t).all():
        raise ValueError('Invalid current/time measurements')
    dt = np.diff(t)
    if np.any(dt < 0):
        raise ValueError('Time must be sorted within the acquisition cycle')
    charge = np.r_[0., np.cumsum(np.maximum(i[1:], 0.) * dt / 3600.)]
    discharge = np.r_[0., np.cumsum(np.maximum(-i[1:], 0.) * dt / 3600.)]
    return charge, discharge


def centered_calce_rule(capacity):
    """Exact medfilt(21), whole-series median residual, strict '<3s' rule.

The upstream zero-MAD behavior (all rejected) is intentionally exposed rather
than silently altered. This function must not drive causal forecast inputs.
"""
    q = np.asarray(capacity, dtype=float)
    if q.ndim != 1 or len(q) < 21 or not np.isfinite(q).all():
        raise ValueError('Centered reference rule requires >=21 finite capacities')
    median = medfilt(q, kernel_size=21)
    residual = np.abs(q - median)
    scale = float(np.median(residual))
    return (residual < 3 * scale) & (q > .1), median, residual, scale


def join_reason(original, extra):
    return ';'.join(sorted(set(filter(None, str(original).split(';') + list(extra)))))


def apply_exclusions(frame, reasons):
    """Add explicit exclusions without releasing any previously rejected row."""
    out = frame.copy()
    out['exclusion_reasons'] = [join_reason(a, b) for a, b in zip(out.exclusion_reasons, reasons)]
    out['quality_eligible'] = out.quality_eligible & out.exclusion_reasons.eq('')
    out['quality_status'] = np.where(out.quality_eligible, 'accepted_with_note', 'excluded')
    out['quality_flags'] = [join_reason(f, [] if ok else ['strict_quality_excluded'])
                            for f, ok in zip(out.quality_flags, out.quality_eligible)]
    return out


def mit_exclusions(canonical_id, canonical_raw_index, *, retrospective=False):
    reasons = []
    if canonical_raw_index == 0:
        reasons.append('batterylife_initial_raw_cycle_0')
    if MIT_BAD_RAW_INDEX.get(canonical_id) == canonical_raw_index:
        reasons.append('batterylife_named_cycle_exception')
    if canonical_id in MIT_COLLECTION_EXCLUSIONS:
        reasons.append('batterylife_collection_excluded_cell')
    if retrospective and canonical_id in MIT_NOT_REACHED_80:
        reasons.append('batterylife_unreached_80_cell_selection')
    return reasons


def is_rpt(description):
    description = str(description).lower()
    return ('test capacity' in description or 'capacity test' in description
            or ('1.0c charge' in description and '0.5c discharge' in description))


def split_capacity(current, capacity, nominal_capacity=2.):
    """Split by the upstream +/-0.01C thresholds, keeping raw sample order."""
    i, q = (np.asarray(a, dtype=float).ravel() for a in (current, capacity))
    if len(i) != len(q) or not len(i) or not np.isfinite(i).all() or not np.isfinite(q).all():
        raise ValueError('Invalid XJTU current/capacity arrays')
    return np.where(i >= .01 * nominal_capacity, q, 0.), np.where(i <= -.01 * nominal_capacity, q, 0.)


def unwrap_step_seconds(minutes, min_step=1e-6):
    """Monotone elapsed coordinates, not recovered wall-clock acquisition time.

Keep raw order; reset steps add no guessed gap. No capacity is integrated on
this artificial axis. Zero/negative increments become min_step.
"""
    raw = np.asarray(minutes, dtype=float).ravel() * 60.
    if not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Invalid XJTU time')
    delta = np.diff(raw)
    fixed = np.r_[0., np.cumsum(np.where(delta > 0, delta, min_step))]
    return fixed, int((delta < 0).sum()), int((delta == 0).sum())


def xjtu_conditions(current, voltage, description, *, batch):
    i, v = (np.asarray(a, dtype=float).ravel() for a in (current, voltage))
    if len(i) != len(v) or not len(i) or not np.isfinite(i).all() or not np.isfinite(v).all():
        raise ValueError('Invalid XJTU current/voltage')
    d = i <= -.02
    charge = i >= .02
    rpt = is_rpt(description)
    rate = float(np.median(-i[d]) / 2.) if d.any() else np.nan
    cutoff = float(v[d][-1]) if d.any() else np.nan
    end_charge = float(v[charge][-1]) if charge.any() else np.nan
    reasons = []
    if not d.any():
        reasons.append('xjtu_no_discharge')
    if not rpt and (not np.isfinite(rate) or not np.isclose(rate, 1., atol=.1, rtol=0)):
        reasons.append('xjtu_non_1c_discharge')
    if not np.isfinite(cutoff) or not np.isclose(cutoff, 2.5, atol=.05, rtol=0):
        reasons.append('xjtu_different_discharge_cutoff')
    if not np.isfinite(end_charge) or end_charge < 4.15:
        reasons.append('xjtu_incomplete_charge_voltage')
    return dict(attribute='RPT' if rpt else 'Cycling',
                measured_discharge_rate_c=rate, measured_discharge_cutoff_v=cutoff,
                charge_end_voltage_v=end_charge,
                measurement_condition=f'{batch}:RPT' if rpt else f'{batch}:1C_discharge_to_2.5V',
                condition_reasons=';'.join(reasons))


def extract_calce_file(filename):
    """Read raw Excel once; retain date, test-time and original cycle lineage."""
    path = Path(filename)
    xls = pd.ExcelFile(path)
    sheets = [s for s in xls.sheet_names if s.startswith('Channel')]
    if len(sheets) != 1:
        raise ValueError(f'Expected one acquisition channel: {path}')
    fields = ['Date_Time', 'Cycle_Index', 'Test_Time(s)', 'Current(A)', 'Voltage(V)', 'Discharge_Capacity(Ah)']
    frame = xls.parse(sheets[0], usecols=fields)
    frame['Date_Time'] = pd.to_datetime(frame.Date_Time)
    frame = frame.sort_values(['Date_Time', 'Test_Time(s)'], kind='stable')
    transitions = frame.Cycle_Index.ne(frame.Cycle_Index.shift()).cumsum()
    result = []
    for _, d in frame.groupby(transitions, sort=False):
        qc, qd = integrate_capacity(d['Current(A)'], d['Test_Time(s)'])
        result.append(dict(source_file=path.name, source_sheet=sheets[0], source_cycle=int(d.Cycle_Index.iloc[0]),
            source_start_time=d.Date_Time.iloc[0].isoformat(), source_end_time=d.Date_Time.iloc[-1].isoformat(),
            integrated_discharge_ah=float(qd.max()), integrated_charge_ah=float(qc.max()),
            raw_samples=len(d), nonpositive_time_steps=int((np.diff(d['Test_Time(s)']) == 0).sum())))
    xls.close()
    return result


def extract_xjtu_file(filename):
    from scipy.io import loadmat
    path = Path(filename)
    raw = loadmat(path, simplify_cells=True, variable_names=['data', 'summary'])
    cycles = raw['data'] if not isinstance(raw['data'], dict) else [raw['data']]
    rate = int(path.stem.split('C')[0])
    batch = 'batch1' if rate == 2 else 'batch2'
    summaries = np.asarray(raw['summary']['discharge_capacity_Ah']).ravel()
    if len(summaries) != len(cycles):
        raise ValueError('XJTU summary/trace length mismatch')
    result = []
    for index, d in enumerate(cycles):
        qc, qd = split_capacity(d['current_A'], d['capacity_Ah'])
        times, resets, duplicates = unwrap_step_seconds(d['relative_time_min'])
        result.append(dict(battery_id=f'{batch}_{path.stem}', cycle=index + 1,
            source_file=path.name, description=str(d['description']).strip(),
            discharge_capacity_split_ah=float(qd.max()), charge_capacity_split_ah=float(qc.max()),
            summary_capacity_ah=float(summaries[index]), time_resets=resets, time_duplicates=duplicates,
            elapsed_span_s=float(times[-1]), raw_samples=len(qd),
            **xjtu_conditions(d['current_A'], d['voltage_V'], d['description'], batch=batch)))
    return result


def load_v4_bundle(directory, *, allow_retrospective=False):
    """Require explicit opt-in to a future-dependent retrospective QC view."""
    data, meta, manifest = load_bundle(directory)
    if manifest.get('preprocessing_version') != 'reviewed_v4':
        raise ValueError('Expected reviewed_v4 bundle')
    if manifest['profile'] == 'retrospective' and not allow_retrospective:
        raise ValueError('Retrospective QC uses future observations; not allowed for causal forecasting')
    return data, meta, manifest
