"""Label-position controls for a fixed-budget head-only search."""
import numpy as np

from src.battery_fewshot_tuning import select_label_windows


def alternate_label_windows(windows, count, seed=1042):
    """One random candidate per lifetime stratum, using cycle positions only."""
    select_label_windows(windows, count)  # Validate cycles and available budget.
    eligible, previous_end = [], -1
    for i, cycles in enumerate(windows["target_cycle"]):
        if cycles[0] > previous_end:
            eligible.append(i)
            previous_end = int(cycles[-1])
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(chunk) for chunk in np.array_split(eligible, count)], dtype=int)


def indices_from_budget(windows, rows):
    """Rebuild exactly the old label windows, rejecting altered cycle bounds."""
    lookup = {int(c): i for i, c in enumerate(windows["forecast_start"])}
    indices = []
    for row in rows.itertuples():
        i = lookup[int(row.label_start)]
        actual = (int(windows["context_cycle"][i, 0]), int(windows["context_cycle"][i, -1]),
                  int(windows["target_cycle"][i, 0]), int(windows["target_cycle"][i, -1]))
        if actual != (row.context_start, row.context_end, row.label_start, row.label_end):
            raise ValueError("Saved label window no longer matches the dataset")
        indices.append(i)
    selected = np.array(indices, dtype=int)
    if not len(selected) or len(set(selected)) != len(selected):
        raise ValueError("Empty or duplicate saved label windows")
    intervals = windows["target_cycle"][selected]
    if np.any(intervals[1:, 0] <= intervals[:-1, -1]):
        raise ValueError("Label intervals overlap or are not ordered")
    return selected
