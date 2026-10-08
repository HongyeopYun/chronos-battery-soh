"""Causal, isolated cross-learning groups: one query and separate support cells."""
import numpy as np

from src.soh_data import EXCLUDED_FLAGS
from src.nasa_covariates import FEATURES


def history_bank(rows, features, context_length=64):
    """Build past-only contexts, including the last observation; no forecast target."""
    if (rows.battery_id.nunique() != 1 or rows.cycle.duplicated().any()
            or not rows.cycle.is_monotonic_increasing or context_length < 2):
        raise ValueError("Expected one ordered battery and a valid context length")
    feature_rows = features.set_index("cycle")
    if not feature_rows.index.is_unique:
        raise ValueError("Duplicate feature cycles")
    cycles = rows.cycle.to_numpy(dtype=np.int64)
    soh = rows.soh.to_numpy(dtype=float)
    cov = feature_rows.loc[cycles, list(FEATURES)].to_numpy(dtype=float)
    valid = (rows.is_observed.to_numpy(dtype=bool) & np.isfinite(soh)
             & np.isfinite(rows.capacity_ah.to_numpy(dtype=float)) & (rows.capacity_ah.to_numpy() > 0)
             & rows.capacity_kind.isin(["full_discharge", "reference_test"]).to_numpy()
             & np.isfinite(cov).all(axis=1))
    valid &= np.array([not (set(str(flag).split(";")) & EXCLUDED_FLAGS) for flag in rows.quality_flags])
    if "quality_eligible" in rows:
        if not rows.quality_eligible.isin([True, False]).all():
            raise ValueError("quality_eligible must contain booleans")
        valid &= rows.quality_eligible.to_numpy(dtype=bool)
    if len(rows) < context_length:
        raise ValueError("Support battery is shorter than the context")
    c = np.lib.stride_tricks.sliding_window_view(cycles, context_length)
    y = np.lib.stride_tricks.sliding_window_view(soh, context_length)
    mask = np.lib.stride_tricks.sliding_window_view(valid, context_length).all(axis=1)
    mask &= (np.diff(c, axis=1) == 1).all(axis=1)
    if "segment_id" in rows:
        segments = np.lib.stride_tricks.sliding_window_view(rows.segment_id.to_numpy(), context_length)
        mask &= (segments == segments[:, :1]).all(axis=1)
    if not mask.any():
        raise ValueError("Support battery has no valid contiguous history")
    return {"battery_id": str(rows.battery_id.iloc[0]), "cycles": c[mask].copy(), "soh": y[mask].copy(),
            "covariates": {name: np.lib.stride_tricks.sliding_window_view(cov[:, i], context_length)[mask].copy()
                           for i, name in enumerate(FEATURES)}}


def support_index(bank, forecast_start):
    """Latest valid support context ending strictly before the query forecast."""
    index = int(np.searchsorted(bank["cycles"][:, -1], forecast_start - 1, side="right") - 1)
    if index < 0:
        raise ValueError("No support history available before forecast origin")
    return index


def isolated_inputs(query_id, context, context_cycles, forecast_start, query_features, banks, names=FEATURES):
    """One query only; reject same-cell support and any input after the cutoff."""
    context, cycles = np.asarray(context), np.asarray(context_cycles)
    if (context.ndim != 1 or cycles.shape != context.shape or not np.isfinite(context).all()
            or np.any(np.diff(cycles) != 1) or cycles[-1] != forecast_start - 1):
        raise ValueError("Invalid query history or forecast cutoff")
    if len({bank["battery_id"] for bank in banks}) != len(banks):
        raise ValueError("Duplicate support batteries")
    if any(bank["battery_id"] == query_id for bank in banks):
        raise ValueError("Support and query batteries must be disjoint")
    f = query_features.set_index("cycle")
    if not f.index.is_unique:
        raise ValueError("Duplicate query feature cycles")
    task = {"target": context.astype(np.float32)}
    if names:
        past = f.loc[cycles, list(names)]
        if not np.isfinite(past.to_numpy()).all():
            raise ValueError("Missing query covariates")
        task["past_covariates"] = {name: past[name].to_numpy(dtype=np.float32) for name in names}
    inputs, selections = [task], []
    for bank in banks:
        index = support_index(bank, forecast_start)
        support_cycles = bank["cycles"][index]
        if len(support_cycles) != len(cycles) or support_cycles[-1] >= forecast_start:
            raise ValueError("Support history crosses query forecast cutoff")
        task = {"target": bank["soh"][index].astype(np.float32)}
        if names:
            task["past_covariates"] = {name: bank["covariates"][name][index].astype(np.float32) for name in names}
        inputs.append(task)
        selections.append({"query_battery_id": query_id, "forecast_start": int(forecast_start),
                           "query_context_start": int(cycles[0]), "query_context_end": int(cycles[-1]),
                           "support_battery_id": bank["battery_id"], "support_context_start": int(support_cycles[0]),
                           "support_context_end": int(support_cycles[-1]),
                           "support_lag_cycles": int(forecast_start - 1 - support_cycles[-1])})
    return inputs, selections


def predict_isolated_group(pipeline, inputs, config):
    """Place exactly one causal query/support group in one inference batch."""
    n_variates = sum(1 + len(task.get("past_covariates", {})) for task in inputs)
    quantiles, _ = pipeline.predict_quantiles(
        inputs=inputs, prediction_length=config.prediction_length, context_length=config.context_length,
        quantile_levels=[i / 10 for i in range(1, 10)], batch_size=n_variates, cross_learning=True)
    if len(quantiles) != len(inputs):
        raise ValueError("Missing query/support forecasts")
    query = quantiles[0].detach().cpu().numpy()[0]
    if query.shape != (config.prediction_length, 9) or not np.isfinite(query).all():
        raise ValueError("Invalid query quantiles")
    # Support predictions are deliberately discarded; score the query only.
    return query[:, 4].copy(), query
