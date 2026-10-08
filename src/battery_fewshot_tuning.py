"""Fixed-budget supervised adaptation, with battery-disjoint held-out evaluation."""
import numpy as np


def adaptation_scores(target, quantiles):
    """Keep raw medians/pinball loss; expose crossed intervals instead of sorting them.

    Any invalid q10/q90 interval makes PICP and width undefined for that group.
    Macro aggregation must propagate these NaNs, rather than omit affected cells.
    """
    from src.battery_model_zeroshot import quantile_scores
    y, q = np.asarray(target, dtype=np.float64), np.asarray(quantiles, dtype=np.float64)
    if q.shape != (*y.shape, 9) or not np.isfinite(q).all():
        raise ValueError("Expected nine finite forecast quantiles")
    crossed = q[..., 0] > q[..., -1]
    if crossed.any():
        result = quantile_scores(y, q[..., 4])
        levels = np.arange(1, 10) / 10
        error = y[..., None] - q
        denominator = np.abs(y).sum()
        result["wql"] = float(2 * np.maximum(levels * error, (levels - 1) * error).sum(axis=(0, 1)).mean()
                              / denominator) if denominator else float("nan")
    else:
        result = quantile_scores(y, q[..., 4], q)
    result["invalid_interval_fraction"] = float(crossed.mean())
    result["quantile_crossing_fraction"] = float((np.diff(q, axis=-1) < 0).any(axis=-1).mean())
    return result


def check_battery_split(train_ids, validation_ids, test_ids):
    groups = [list(ids) for ids in (train_ids, validation_ids, test_ids)]
    if any(not ids or len(ids) != len(set(ids)) for ids in groups):
        raise ValueError("Each split needs unique battery IDs")
    if any(set(groups[i]) & set(groups[j]) for i, j in [(0, 1), (0, 2), (1, 2)]):
        raise ValueError("Training, validation and test batteries must be disjoint")


def select_label_windows(windows, count):
    """Uniform selection from greedily non-overlapping target intervals, using cycles only."""
    starts, cycles = windows["forecast_start"], windows["target_cycle"]
    if count < 1 or len(starts) != len(cycles) or np.any(np.diff(starts) <= 0):
        raise ValueError("Invalid window budget or cycle order")
    eligible, previous_end = [], -1
    for index, (start, target_cycles) in enumerate(zip(starts, cycles)):
        if target_cycles[0] != start or np.any(np.diff(target_cycles) != 1):
            raise ValueError("Target horizon must contain consecutive cycles")
        if start > previous_end:
            eligible.append(index)
            previous_end = int(target_cycles[-1])
    if len(eligible) < count:
        raise ValueError(f"Need {count} non-overlapping label windows; found {len(eligible)}")
    return np.asarray(eligible)[np.linspace(0, len(eligible) - 1, count, dtype=int)]


def fixed_training_inputs(windows, features, indices, names=()):
    """One 64+16 sequence per sample; future sensors absent, SOH horizon used as loss labels.

    Chronos2Dataset must be instantiated with min_past=context_length=64,
    prediction_length=16: this fixes its split at 64, rather than drawing new labels.
    """
    if windows["context"].shape[1] != 64 or windows["target"].shape[1] != 16:
        raise ValueError("This protocol requires exactly 64 -> 16")
    f = features.set_index("cycle")
    if not f.index.is_unique:
        raise ValueError("Pass features for a single battery")
    inputs = []
    for index in indices:
        history_cycles = windows["context_cycle"][index]
        label_cycles = windows["target_cycle"][index]
        if (np.any(np.diff(history_cycles) != 1) or history_cycles[-1] + 1 != label_cycles[0]
                or np.any(np.diff(label_cycles) != 1)):
            raise ValueError("History and label horizon must be consecutive")
        y = np.concatenate([windows["context"][index], windows["target"][index]]).astype(np.float32)
        if not np.isfinite(y).all():
            raise ValueError("Missing SOH in fixed training window")
        task = {"target": y}
        if names:
            past = f.loc[history_cycles, list(names)]
            if not np.isfinite(past.to_numpy()).all():
                raise ValueError("Missing historical covariates")
            task["past_covariates"] = {
                name: np.concatenate([past[name].to_numpy(dtype=np.float32), np.full(16, np.nan, dtype=np.float32)])
                for name in names}
        inputs.append(task)
    return inputs
