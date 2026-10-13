"""Capacity-threshold RUL labels and point-trajectory evaluation.

No model fitting, smoothing, interpolation, or extrapolation lives here.
The event is absorbing even when capacity later recovers.
"""
import numpy as np
import pandas as pd


def eligible_mask(frame):
    return (frame.quality_eligible.eq(True) & frame.is_observed.eq(True)
            & frame.capacity_kind.isin(['full_discharge', 'reference_test'])
            & np.isfinite(frame.capacity_ah) & frame.capacity_ah.gt(0)
            & np.isfinite(frame.soh))


def first_crossing(cycles, values, threshold, consecutive=1):
    cycles, values = np.asarray(cycles), np.asarray(values, dtype=float)
    if (cycles.ndim != 1 or cycles.shape != values.shape or consecutive < 1
            or not np.isfinite(threshold) or not np.isfinite(cycles).all()
            or (np.diff(cycles) <= 0).any()):
        raise ValueError('Ordered unique cycles, aligned values and positive run length required')
    below = np.isfinite(values) & (values <= threshold)
    for i in range(len(values) - consecutive + 1):
        if below[i:i+consecutive].all() and (np.diff(cycles[i:i+consecutive]) == 1).all():
            return int(cycles[i])
    return None


def make_tasks(data, metadata, context_length=64, origin_stride=16, threshold_ah=1.4):
    """Origins are fixed record indices; inclusion at t uses observations <= t only."""
    audits, tasks = [], []
    for battery, raw in data.groupby('battery_id', sort=True):
        frame = raw.sort_values('cycle').reset_index(drop=True)
        cycles = frame.cycle.to_numpy(dtype=int)
        # This pilot uses the NASA discharge-record ordinal, not arbitrary row compression.
        if not np.array_equal(cycles, np.arange(1, len(frame) + 1)):
            raise ValueError('This pilot requires the unchanged consecutive NASA record axis')
        meta = metadata.set_index('battery_id').loc[battery]
        qref = float(meta.reference_capacity_ah)
        if not np.isfinite(qref) or qref <= 0:
            raise ValueError('Invalid normalization capacity')
        np.testing.assert_allclose(frame.soh, frame.capacity_ah / qref, rtol=1e-12)
        keep = eligible_mask(frame).to_numpy()
        capacity = np.where(keep, frame.capacity_ah, np.nan)
        soh = np.where(keep, frame.soh, np.nan)
        eol = first_crossing(cycles, capacity, threshold_ah)
        last = int(cycles[keep][-1])
        audits.append(dict(battery_id=battery, reference_capacity_ah=qref,
                           threshold_ah=threshold_ah, threshold_soh=threshold_ah/qref,
                           event_observed=eol is not None, eol_cycle=eol,
                           first_3_consecutive_below_cycle=first_crossing(cycles, capacity, threshold_ah, 3),
                           last_eligible_cycle=last, n_rows=len(frame), n_eligible=int(keep.sum())))
        for origin in range(context_length, last + 1, origin_stride):
            # No event labels or future lifetime fractions determine these origins.
            if first_crossing(cycles[:origin], capacity[:origin], threshold_ah) is not None:
                continue
            context = soh[origin-context_length:origin].copy()
            if not keep[origin-1] or not np.isfinite(context).any():
                continue
            tasks.append(dict(battery_id=battery, origin_cycle=origin,
                              context_start=origin-context_length+1, context=context,
                              context_observed=int(np.isfinite(context).sum()),
                              threshold_soh=threshold_ah/qref,
                              event_observed=eol is not None, true_eol_cycle=eol,
                              true_rul=None if eol is None else eol-origin,
                              observed_rul_lower_bound=last-origin if eol is None else None))
    return pd.DataFrame(audits), tasks


def score_trajectory(task, median):
    median = np.asarray(median, dtype=float)
    if median.ndim != 1 or not len(median) or not np.isfinite(median).all():
        raise ValueError('A finite one-dimensional forecast is required')
    horizon, origin = len(median), task['origin_cycle']
    predicted = first_crossing(np.arange(1, horizon+1), median, task['threshold_soh'])
    exact = task['true_rul']
    error = None if predicted is None or exact is None else predicted-exact
    # On an integer cycle axis, a non-hit implies RUL >= H+1, not RUL == H.
    error_bound = None if exact is None else (abs(error) if error is not None else max(0, horizon+1-exact))
    censor_bound = task['observed_rul_lower_bound']
    return {k:v for k,v in task.items() if k != 'context'} | dict(
        horizon=horizon, predicted_rul=predicted,
        predicted_eol_cycle=None if predicted is None else origin+predicted,
        predicted_crossing=predicted is not None,
        predicted_rul_lower_bound=horizon if predicted is None else None,
        error_cycles=error, absolute_error_cycles=None if error is None else abs(error),
        absolute_error_lower_bound_cycles=error_bound,
        contradicts_censored_followup=bool(censor_bound is not None and predicted is not None and predicted <= censor_bound))


def summarize_rul(scores):
    """Exact MAE is conditional when some forecasts never cross the threshold."""
    observed = scores[scores.event_observed.eq(True)]
    finite = observed[observed.predicted_crossing.eq(True)]
    error = finite.error_cycles.to_numpy(dtype=float)
    result = dict(n_origins=len(scores), n_event_origins=len(observed),
                  n_event_batteries=int(observed.battery_id.nunique()),
                  n_finite_event_predictions=len(finite),
                  n_no_crossing_event_predictions=len(observed)-len(finite),
                  conditional_mae_cycles=float(np.mean(abs(error))) if len(error) else None,
                  conditional_rmse_cycles=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                  conditional_bias_cycles=float(np.mean(error)) if len(error) else None,
                  mae_lower_bound_all_event_origins=float(observed.absolute_error_lower_bound_cycles.mean()) if len(observed) else None,
                  censored_origins=int((~scores.event_observed).sum()),
                  censored_contradictions=int(scores.contradicts_censored_followup.sum()))
    result['all_event_predictions_finite'] = len(finite) == len(observed)
    return result
