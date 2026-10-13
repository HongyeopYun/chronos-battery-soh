import unittest
import numpy as np
import pandas as pd
from src.c2_rul import first_crossing, make_tasks, score_trajectory


class RulTests(unittest.TestCase):
    def test_first_passage_is_not_record_end(self):
        self.assertEqual(first_crossing([1,2,3,4], [1.8,1.39,1.5,1.3], 1.4), 2)
        self.assertIsNone(first_crossing([1,2,3], [1.8,1.5,1.41], 1.4))

    def test_confirmation_does_not_bridge_missing_or_excluded(self):
        self.assertIsNone(first_crossing([1,2,3], [1.3,np.nan,1.3], 1.4, 2))
        self.assertIsNone(first_crossing([1,3], [1.3,1.3], 1.4, 2))
        self.assertEqual(first_crossing([1,2,3,4], [1.5,1.3,1.2,1.1], 1.4, 3), 2)

    def fixture(self):
        q = np.array([2.,1.9,1.8,1.7,1.6,1.3,1.2,1.1])
        frame = pd.DataFrame(dict(battery_id=['a']*8,cycle=np.arange(1,9),capacity_ah=q,
                                 soh=q/2,quality_eligible=True,is_observed=True,capacity_kind='full_discharge'))
        return frame, pd.DataFrame([dict(battery_id='a',reference_capacity_ah=2.)])

    def test_context_ends_at_origin_and_future_does_not_change_it(self):
        data, meta = self.fixture()
        _, original = make_tasks(data, meta, 4, 2)
        data.loc[data.cycle>4, 'capacity_ah'] = 1.8
        data['soh'] = data.capacity_ah/2
        _, altered = make_tasks(data, meta, 4, 2)
        np.testing.assert_array_equal(original[0]['context'], altered[0]['context'])
        self.assertEqual(original[0]['true_rul'], 2)
        self.assertIsNone(altered[0]['true_rul'])
        self.assertEqual([t['origin_cycle'] for t in original], [4])

    def test_excluded_point_is_masked_without_time_compression(self):
        data, meta = self.fixture()
        data.loc[1,'quality_eligible'] = False
        _, tasks = make_tasks(data, meta, 4, 2)
        self.assertEqual(len(tasks[0]['context']), 4)
        self.assertTrue(np.isnan(tasks[0]['context'][1]))
        self.assertEqual(tasks[0]['context_start'], 1)

    def test_horizon_non_hit_is_a_bound_not_a_point(self):
        task = dict(origin_cycle=4,threshold_soh=.7,true_rul=2,observed_rul_lower_bound=None)
        score = score_trajectory(task, [.9,.8,.75])
        self.assertIsNone(score['predicted_rul'])
        self.assertIsNone(score['error_cycles'])
        self.assertEqual(score['absolute_error_lower_bound_cycles'], 2)
        self.assertEqual(score_trajectory(task,[.8,.7,.6])['predicted_rul'],2)

    def test_censored_record_has_no_exact_error(self):
        task = dict(origin_cycle=4,threshold_soh=.7,true_rul=None,observed_rul_lower_bound=4)
        score = score_trajectory(task, [.8,.6,.5])
        self.assertIsNone(score['error_cycles'])
        self.assertTrue(score['contradicts_censored_followup'])


if __name__ == '__main__':
    unittest.main()
