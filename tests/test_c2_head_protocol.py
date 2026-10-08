import unittest
from src.c2_head_protocol import make_folds, check_physical_split


class HeadProtocolTests(unittest.TestCase):
    def test_four_folds_hold_out_each_cell_and_never_share_its_labels(self):
        records = [dict(battery_id=f"B{i}", physical_battery_id=f"B{i}") for i in (1, 2, 3, 4)]
        folds = make_folds("nasa_aging", records)
        self.assertEqual([f["test_ids"][0] for f in folds], ["B1", "B2", "B3", "B4"])
        for f in folds:
            self.assertEqual(len(f["train_ids"]), 2)
            self.assertFalse(set(f["test_ids"]) & set(f["train_ids"] + f["validation_ids"]))

    def test_mit_continuations_never_enter_sources_even_under_different_record_names(self):
        records = [dict(battery_id=f"b2c{i}", physical_battery_id=f"p{i}") for i in (0, 1, 2, 3, 4, 10)]
        f = make_folds("mit_batch2", records, repeated_physical_ids={"p0", "p2"})[0]
        self.assertEqual(f["train_ids"], ["b2c1", "b2c3"])
        self.assertEqual(f["validation_ids"], ["b2c4"])
        self.assertEqual(f["test_ids"], ["b2c0", "b2c2", "b2c10"])

    def test_distinct_record_names_do_not_bypass_physical_cell_check(self):
        fold = dict(train_ids=["a"], validation_ids=["b"], test_ids=["c"])
        with self.assertRaises(ValueError):
            check_physical_split(fold, {"a": "same", "b": "separate", "c": "same"})


if __name__ == "__main__":
    unittest.main()
