import unittest
import numpy as np
import pandas as pd
from src.c2_head_search import alternate_label_windows, indices_from_budget
from src.battery_fewshot_tuning import select_label_windows


class HeadSearchTests(unittest.TestCase):
    def setUp(self):
        starts = np.arange(65, 420)
        self.w = dict(forecast_start=starts, context_cycle=starts[:, None] + np.arange(-64, 0),
                      target_cycle=starts[:, None] + np.arange(16), target=np.ones((len(starts), 16)))

    def test_alternate_selection_is_reproducible_label_independent_and_nonoverlapping(self):
        a = alternate_label_windows(self.w, 10)
        self.w["target"][:] = 999
        np.testing.assert_array_equal(a, alternate_label_windows(self.w, 10))
        self.assertFalse(np.array_equal(a, select_label_windows(self.w, 10)))
        spans = self.w["target_cycle"][a]
        self.assertTrue(np.all(spans[1:, 0] > spans[:-1, -1]))
        self.assertEqual(len(a), 10)
        with self.assertRaises(ValueError): alternate_label_windows(self.w, 100)

    def test_old_budget_roundtrip_and_changed_bounds_rejected(self):
        indices = select_label_windows(self.w, 10)
        rows = pd.DataFrame([dict(context_start=int(self.w["context_cycle"][i, 0]),
                                 context_end=int(self.w["context_cycle"][i, -1]),
                                 label_start=int(self.w["target_cycle"][i, 0]),
                                 label_end=int(self.w["target_cycle"][i, -1])) for i in indices])
        np.testing.assert_array_equal(indices, indices_from_budget(self.w, rows))
        rows.loc[0, "context_start"] += 1
        with self.assertRaises(ValueError): indices_from_budget(self.w, rows)

    def test_overlapping_saved_labels_rejected(self):
        rows = pd.DataFrame([dict(context_start=i+1, context_end=i+64,
                                 label_start=i+65, label_end=i+80) for i in range(2)])
        with self.assertRaises(ValueError): indices_from_budget(self.w, rows)


if __name__ == "__main__": unittest.main()
