from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np
import pandas as pd
from scipy.io import savemat

from src.battery_preprocessing import (calce_cycle_record, deduplicate_calce_records, make_frame, read_calce, read_mit,
                                      read_nasa, save_bundle)
from src.soh_data import load_bundle, observed_windows


class BatteryPreprocessingTests(unittest.TestCase):
    def test_calce_cumulative_counter_includes_first_increment_and_incomplete_flag(self):
        rows = pd.DataFrame({"Current(A)": [0., 1.1, -1.1, -1.1, 0.],
                             "Discharge_Capacity(Ah)": [.8, .8, .81, 1.94, 1.94],
                             "Voltage(V)": [3.4, 4.2, 4.1, 2.7, 3.2], "Cycle_Index": [2] * 5,
                             "Date_Time": pd.date_range("2010-08-17", periods=5, freq="min")})
        path = Path("CS2_35_8_17_10.xlsx")
        record = calce_cycle_record(rows, path, "Channel_1")
        self.assertAlmostEqual(record["capacity_ah"], 1.14)
        self.assertEqual(record["capacity_kind"], "full_discharge")
        self.assertEqual(record["quality_flags"], "")
        duplicate = dict(record, source_file="CS2_35_8_18_10.xlsx")
        unique, n = deduplicate_calce_records([record, duplicate])
        self.assertEqual(n, 1)
        self.assertEqual(len(unique), 1)
        self.assertIn("CS2_35_8_18_10.xlsx", unique[0]["source_file"])
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            deduplicate_calce_records([record, dict(duplicate, capacity_ah=1.1)])
        rows.loc[3, "Voltage(V)"] = 3.4
        self.assertEqual(calce_cycle_record(rows, path, "Channel_1")["capacity_kind"], "partial_discharge")
        rows.loc[3, "Discharge_Capacity(Ah)"] = .2
        self.assertIn("discharge_counter_reset", calce_cycle_record(rows, path, "Channel_1")["quality_flags"])
        rows["Current(A)"] = 0.
        record = calce_cycle_record(rows, path, "Channel_1")
        self.assertFalse(record["is_observed"])
        self.assertEqual(record["capacity_kind"], "unknown")

    def test_calce_recording_restarts_do_not_join_forecast_windows(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for battery in ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]:
                (root / battery).mkdir()
                for day in [17, 18]:
                    rows = []
                    for cycle in range(1, 46):
                        for step, (current, counter, voltage) in enumerate([(0., cycle - 1., 4.2), (-1.1, cycle - .99, 4.), (-1.1, float(cycle), 2.7), (0., float(cycle), 3.2)]):
                            rows.append({"Date_Time": pd.Timestamp(2010, 8, day) + pd.Timedelta(minutes=cycle * 5 + step),
                                "Cycle_Index": cycle, "Current(A)": current, "Voltage(V)": voltage,
                                "Discharge_Capacity(Ah)": counter})
                    pd.DataFrame(rows).to_excel(root / battery / f"{battery}_8_{day}_10.xlsx", sheet_name="Channel_1", index=False)
            group, data, batteries, hashes = read_calce(root)[0]
            self.assertEqual(len(data), 360)
            rows = data[data.battery_id == "CS2_35"]
            np.testing.assert_allclose(rows.capacity_ah, 1.)
            self.assertEqual(rows.cycle.tolist(), list(range(1, 91)))
            self.assertEqual(rows.source_cycle.iloc[45], 1)
            self.assertEqual(len(observed_windows(rows)["target"]), 0)
            self.assertEqual(len(observed_windows(rows.drop(columns="segment_id"))["target"]), 11)
            audit, manifest = save_bundle(group, data, batteries, hashes, root / "processed")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(int(audit.n_prediction_windows.sum()), 0)
            loaded, _, _ = load_bundle(root / "processed")
            self.assertIn("segment_id", loaded)

    def test_mit_hdf5_references_continuations_and_noisy_cells_are_preserved(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for number, count in [(1, 46), (2, 48), (3, 46)]:
                with h5py.File(root / f"MIT_Batch_{number}.mat", "w") as f:
                    batch = f.create_group("batch")
                    summaries = batch.create_dataset("summary", (count, 1), dtype=h5py.ref_dtype)
                    policies = batch.create_dataset("policy_readable", (count, 1), dtype=h5py.ref_dtype)
                    for index in range(count):
                        summary = f.create_group(f"summary{index}")
                        q = np.linspace(1.05, .9, 90)
                        if number == 1:
                            q[0] = 0.
                        summary.create_dataset("QDischarge", data=q.reshape(1, -1))
                        summary.create_dataset("cycle", data=np.arange(1, 91).reshape(1, -1))
                        policy = f.create_dataset(f"policy{index}", data=np.array([ord(c) for c in "3C(50%)-4C"], dtype=np.uint16).reshape(-1, 1))
                        summaries[index, 0], policies[index, 0] = summary.ref, policy.ref
            # Hashing source files is tested by bundle tests; these tiny fixtures need no large-file cost.
            groups = read_mit(root)
            self.assertEqual([len(group[2]) for group in groups], [46, 48, 46])
            _, first, first_meta, _ = groups[0]
            first = first[first.battery_id == "b1c0"]
            self.assertEqual(len(first), 90)
            self.assertFalse(first.iloc[0].is_observed)
            self.assertEqual(first_meta.iloc[0].reference_cycle, 2)
            self.assertAlmostEqual(first.soh.iloc[1], 1.)
            _, second, second_meta, _ = groups[1]
            continuation = second[second.battery_id == "b2c7"]
            parent = second_meta[second_meta.battery_id == "b2c7"].iloc[0]
            self.assertEqual(parent.physical_battery_id, "b1c0")
            self.assertEqual(continuation.cycle.iloc[0], 91)
            self.assertEqual(continuation.source_cycle.iloc[0], 1)
            np.testing.assert_allclose(continuation.soh, continuation.capacity_ah / first_meta.iloc[0].reference_capacity_ah)
            _, third, _, _ = groups[2]
            noisy = third[third.battery_id == "b3c37"]
            self.assertEqual(len(noisy), 90)
            self.assertEqual(len(observed_windows(noisy)["target"]), 0)
            self.assertEqual(len(observed_windows(third[third.battery_id == "b3c0"])["target"]), 11)

    def test_nasa_uses_reported_discharge_not_current_integral_or_charge_rows(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for battery in ["B0005", "B0006", "B0007", "B0018"]:
                savemat(root / f"{battery}.mat", {battery: {"cycle": np.array([
                    {"type": "charge", "data": {"Capacity": 999.}},
                    {"type": "discharge", "data": {"Capacity": 1.8}},
                    {"type": "impedance", "data": {"Capacity": 888.}},
                    {"type": "discharge", "data": {"Capacity": 1.7}}], dtype=object)}})
            _, data, _, _ = read_nasa(root)[0]
            rows = data[data.battery_id == "B0005"]
            self.assertEqual(rows.source_record_index.tolist(), [2, 4])
            np.testing.assert_allclose(rows.soh, [1., 1.7 / 1.8])
            with self.assertRaisesRegex(ValueError, "duplicate or unordered"):
                make_frame("nasa", "bad", [1, 1], [1.8, 1.7], 1.8, "test")


if __name__ == "__main__":
    unittest.main()
