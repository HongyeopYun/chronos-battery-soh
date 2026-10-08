from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from src.battery_fewshot import history_bank, isolated_inputs, predict_isolated_group
from src.nasa_covariates import FEATURES


def fixture(battery="support"):
    cycles = np.arange(1, 101)
    rows = pd.DataFrame({"battery_id": battery, "cycle": cycles, "soh": 1 - cycles / 1000,
                         "capacity_ah": 2., "capacity_kind": "full_discharge", "is_observed": True,
                         "quality_flags": "", "segment_id": battery})
    features = pd.DataFrame({"cycle": cycles, FEATURES[0]: 4 - cycles / 1000, FEATURES[1]: -2.})
    return rows, features


class FewshotTests(unittest.TestCase):
    def test_strict_quality_mask_also_excludes_support_contexts(self):
        rows, features = fixture()
        rows["quality_eligible"] = True
        rows.loc[rows.cycle == 50, "quality_eligible"] = False
        bank = history_bank(rows, features, context_length=20)
        self.assertFalse(np.any(bank["cycles"] == 50))

    def test_mutating_all_future_soh_and_sensors_cannot_change_inputs(self):
        rows, features = fixture()
        query_rows, query_features = fixture("query")
        args = ("query", query_rows.soh.to_numpy()[:64], np.arange(1, 65), 65, query_features)
        before, selections = isolated_inputs(*args, [history_bank(rows, features)])
        rows.loc[rows.cycle >= 65, "soh"] = 999
        features.loc[features.cycle >= 65, list(FEATURES)] = -999
        query_features.loc[query_features.cycle >= 65, list(FEATURES)] = 999
        after, _ = isolated_inputs(*args, [history_bank(rows, features)])
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a["target"], b["target"])
            for name in FEATURES:
                np.testing.assert_array_equal(a["past_covariates"][name], b["past_covariates"][name])
            self.assertNotIn("future_covariates", a)
        self.assertEqual(selections[0]["support_context_end"], 64)

    def test_same_battery_and_future_query_history_rejected(self):
        rows, f = fixture("query")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            isolated_inputs("query", rows.soh.to_numpy()[:64], np.arange(1, 65), 65, f, [history_bank(rows, f)])
        with self.assertRaisesRegex(ValueError, "cutoff"):
            isolated_inputs("query", rows.soh.to_numpy()[:64], np.arange(2, 66), 65, f, [])

    def test_quality_gaps_and_short_support_history(self):
        rows, f = fixture()
        rows.loc[rows.cycle == 50, "quality_flags"] = "capacity_outlier"
        bank = history_bank(rows, f, context_length=20)
        self.assertFalse(np.any((bank["cycles"][:, 0] <= 50) & (bank["cycles"][:, -1] >= 50)))
        rows, f = fixture()
        with self.assertRaisesRegex(ValueError, "available"):
            isolated_inputs("q", np.ones(20), np.arange(1, 21), 21, f, [history_bank(rows, f, context_length=64)])

    def test_one_group_stays_in_single_batch_and_only_query_is_scored(self):
        class Tensor:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value
        class Pipeline:
            def predict_quantiles(self, **kwargs):
                self.kwargs = kwargs
                return [Tensor(np.ones((1, 16, 9)) * i) for i in range(3)], []
        p = Pipeline()
        tasks = [{"target": np.ones(64), "past_covariates": {name: np.ones(64) for name in FEATURES}} for _ in range(3)]
        point, q = predict_isolated_group(p, tasks, SimpleNamespace(prediction_length=16, context_length=64))
        self.assertEqual(p.kwargs["batch_size"], 9)
        self.assertTrue(p.kwargs["cross_learning"])
        self.assertEqual(len(p.kwargs["inputs"]), 3)
        self.assertTrue(np.all(point == 0))


if __name__ == "__main__":
    unittest.main()
