import unittest
from dataclasses import replace

from src.soh_ablation import choose_candidate, replay_selection
from src.soh_experiment import ExperimentConfig


def score(rmse, crps, mean_rmse=None):
    return {"median_rmse": rmse, "mean_rmse": rmse if mean_rmse is None else mean_rmse, "crps": crps}


class AblationPolicyTests(unittest.TestCase):
    def test_early_stopping_cannot_use_later_improvement(self):
        history = [{"epoch": epoch, "validation": score(1.0, 1.0)} for epoch in range(4)]
        history.append({"epoch": 4, "validation": score(0.8, 0.8)})
        config = ExperimentConfig(patience=3)
        self.assertEqual(replay_selection(history, config)["selected_epoch"], 0)
        self.assertEqual(replay_selection(history, replace(config, patience=10))["selected_epoch"], 4)

    def test_guard_rejects_better_crps_with_exploding_mean(self):
        history = [{"epoch": 0, "validation": score(1.0, 1.0)},
                   {"epoch": 1, "validation": score(0.8, 0.8, mean_rmse=4.0)}]
        self.assertEqual(replay_selection(history, ExperimentConfig())["selected_epoch"], 0)

    def test_candidate_selection_never_uses_test_metrics(self):
        records = {"first": {"selected_validation": score(1.0, 1.0), "test_rmse": 100.0},
                   "second": {"selected_validation": score(0.8, 1.0), "test_rmse": 0.0}}
        self.assertEqual(choose_candidate(records, 1e-5), "first")
        records["second"]["selected_validation"]["crps"] = 0.9
        self.assertEqual(choose_candidate(records, 1e-5), "second")


if __name__ == "__main__":
    unittest.main()
