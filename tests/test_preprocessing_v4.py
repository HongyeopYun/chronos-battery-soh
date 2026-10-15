import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.preprocessing_v4 import (
    apply_exclusions, centered_calce_rule, integrate_capacity, is_rpt,
    load_v4_bundle, mit_exclusions, split_capacity, unwrap_step_seconds,
    xjtu_conditions,
)


class V4PreprocessingTests(unittest.TestCase):
    def test_right_endpoint_integral_and_charge_sign(self):
        charge, discharge = integrate_capacity([0, 1, -2, 0, -1], [0, 3600, 5400, 9000, 12600])
        np.testing.assert_allclose(charge, [0, 1, 1, 1, 1])
        np.testing.assert_allclose(discharge, [0, 0, 1, 1, 2])
        with self.assertRaises(ValueError):
            integrate_capacity([-1, -1], [2, 1])
        with self.assertRaises(ValueError):
            integrate_capacity([-1, np.nan], [1, 2])

    def test_integral_is_prefix_invariant(self):
        current = np.array([0, -1, -1, 0, 2, -3.])
        seconds = np.arange(6.) * 100
        before = integrate_capacity(current[:4], seconds[:4])
        after = integrate_capacity(current, seconds)
        for short, long in zip(before, after):
            np.testing.assert_array_equal(short, long[:4])

    def test_centered_rule_detects_both_directions_without_replacing_q(self):
        rng = np.random.default_rng(13)
        q = np.linspace(1.1, .6, 201) + rng.normal(0, .005, 201)
        q[50] += .3
        q[90] -= .3
        original = q.copy()
        keep, median, residual, scale = centered_calce_rule(q)
        self.assertFalse(keep[50])
        self.assertFalse(keep[90])
        np.testing.assert_array_equal(q, original)
        np.testing.assert_array_equal(keep, (abs(q-median) < 3*scale) & (q>.1))
        self.assertGreater(keep.sum(), 130)

    def test_original_zero_scale_behavior_is_visible(self):
        keep, _, _, scale = centered_calce_rule(np.ones(31))
        self.assertEqual(scale, 0)
        self.assertFalse(keep.any())

    def test_mit_raw_offsets_and_continuation_initial_cycle(self):
        for cell, idx in [('b1c0',11),('b2c12',252),('b2c44',247),('b1c18',39)]:
            self.assertIn('batterylife_named_cycle_exception', mit_exclusions(cell, idx))
            self.assertNotIn('batterylife_named_cycle_exception', mit_exclusions(cell, idx+1))
        self.assertIn('batterylife_initial_raw_cycle_0', mit_exclusions('b1c0',0))
        # Local raw index 0 of a continuation is NOT canonical raw index 0.
        self.assertEqual(mit_exclusions('b1c0',500), [])
        self.assertEqual(mit_exclusions('b1c8',10), [])
        self.assertIn('batterylife_unreached_80_cell_selection', mit_exclusions('b1c8',10,retrospective=True))
        self.assertIn('batterylife_collection_excluded_cell',mit_exclusions('b3c23',20))

    def test_exclusion_retains_row_coordinates_and_existing_qc(self):
        d = pd.DataFrame(dict(cycle=[1,2,3], capacity_ah=[1.,.2,.9], quality_eligible=[True,False,True],
                              exclusion_reasons=['','old_failure',''],quality_flags=['','strict_quality_excluded','']))
        result = apply_exclusions(d, [[],[],['named_exception']])
        np.testing.assert_array_equal(result.cycle, d.cycle)
        np.testing.assert_array_equal(result.capacity_ah, d.capacity_ah)
        self.assertEqual(result.quality_eligible.tolist(),[True,False,False])
        self.assertEqual(result.exclusion_reasons.iloc[1],'old_failure')

    def test_xjtu_current_units_and_rest_boundary(self):
        charge, discharge = split_capacity([.02,.019,0,-.019,-.02,-2], [1,2,3,4,5,6])
        np.testing.assert_array_equal(charge,[1,0,0,0,0,0])
        np.testing.assert_array_equal(discharge,[0,0,0,0,5,6])
        self.assertTrue(is_rpt('0.5C charge and 0.2C discharge [test capacity]'))
        self.assertFalse(is_rpt('2C charge and 1C discharge'))

    def test_xjtu_partial_discharge_is_not_aging_capacity(self):
        i=[4., .1, 0., -2., -2., 0.]
        full=xjtu_conditions(i,[3,4.2,4.2,4.1,2.5,2.6],'2C charge and 1C discharge',batch='batch1')
        self.assertEqual(full['condition_reasons'],'')
        partial=xjtu_conditions(i,[3,4.2,4.2,4.1,3.,3.1],'2C charge and 1C discharge',batch='batch1')
        self.assertIn('xjtu_different_discharge_cutoff',partial['condition_reasons'])
        wrong_rate=xjtu_conditions([4,.1,0,-4,-4,0],[3,4.2,4.2,4.1,2.5,2.6],
                                   '2C charge and 2C discharge',batch='batch1')
        self.assertIn('xjtu_non_1c_discharge',wrong_rate['condition_reasons'])

    def test_time_reset_keeps_order_and_does_not_invent_a_gap(self):
        result,resets,duplicates=unwrap_step_seconds([0,1,2,0,1,1,2])
        self.assertTrue((np.diff(result)>0).all())
        self.assertEqual((resets,duplicates),(1,1))
        self.assertAlmostEqual(result[-1],240.000002)

    def test_future_dependent_view_requires_explicit_opt_in(self):
        payload = (None,None,dict(preprocessing_version='reviewed_v4',profile='retrospective'))
        with patch('src.preprocessing_v4.load_bundle',return_value=payload):
            with self.assertRaisesRegex(ValueError,'future observations'):
                load_v4_bundle('unused')
            self.assertEqual(load_v4_bundle('unused',allow_retrospective=True),payload)


if __name__ == '__main__':
    unittest.main()
