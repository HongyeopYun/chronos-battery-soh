import unittest
import numpy as np
import pandas as pd
from scripts.rescore_c2_rul_soh80 import eol_audit, rethreshold

class Soh80Tests(unittest.TestCase):
    def data(self):
        d=pd.DataFrame(dict(battery_id=['A']*5,cycle=[1,2,3,4,5],soh=[1.,.79,.85,.8,.75],
            capacity_ah=np.array([1.,.79,.85,.8,.75])*3,quality_eligible=True,is_observed=True,capacity_kind='full_discharge'))
        return d,pd.DataFrame(dict(battery_id=['A'],reference_capacity_ah=[3.]))

    def test_first_crossing_is_absorbing_and_not_rated_1p6ah(self):
        d,m=self.data();r=eol_audit(d,m).iloc[0]
        self.assertEqual(r.eol_cycle,2);self.assertAlmostEqual(r.threshold_capacity_ah,2.4)
        self.assertIsNone(rethreshold(dict(origin_cycle=3),r))
        self.assertIsNone(rethreshold(dict(origin_cycle=2),r))
        self.assertEqual(rethreshold(dict(origin_cycle=1),r)['true_rul'],1)

    def test_excluded_measurement_does_not_define_eol(self):
        d,m=self.data();d.loc[d.cycle.eq(2),'quality_eligible']=False
        self.assertEqual(eol_audit(d,m).iloc[0].eol_cycle,4)

    def test_censoring_and_future_do_not_change_origin_admission(self):
        d,m=self.data();r=eol_audit(d,m).iloc[0]
        original=rethreshold(dict(origin_cycle=1),r)
        d['soh']=1.;d['capacity_ah']=3.
        changed=rethreshold(dict(origin_cycle=1),eol_audit(d,m).iloc[0])
        self.assertIsNotNone(original);self.assertIsNotNone(changed)
        self.assertIsNone(changed['true_rul']);self.assertEqual(changed['observed_rul_lower_bound'],4)

if __name__=='__main__':unittest.main()
