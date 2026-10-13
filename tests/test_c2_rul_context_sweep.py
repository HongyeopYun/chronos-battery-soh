import unittest
import numpy as np
import pandas as pd
from scripts.run_c2_rul_context_sweep import matched_inputs
from scripts.report_c2_rul_context_sweep import trajectory_diagnostics

class MatchedRulTests(unittest.TestCase):
    def frame(self):
        n=130
        d=pd.DataFrame(dict(battery_id=['A']*n,cycle=np.arange(1,n+1),capacity_ah=np.linspace(2,1.2,n),
            quality_eligible=True,is_observed=True,capacity_kind='full_discharge'))
        d['soh']=d.capacity_ah/2
        return d,pd.DataFrame(dict(battery_id=['A'],reference_capacity_ah=[2.]))

    def test_same_origin_and_future_labels_for_all_lengths(self):
        d,m=self.frame();reference=None
        for length in [16,32,64]:
            _,tasks,source=matched_inputs(d,m,length)
            self.assertEqual(tasks[0]['origin_cycle'],64)
            self.assertEqual(tasks[0]['context_start'],65-length)
            np.testing.assert_array_equal(tasks[0]['context'],d.soh.to_numpy()[64-length:64])
            if reference is None:reference=source['A']['target']
            np.testing.assert_array_equal(source['A']['target'],reference)
            self.assertAlmostEqual(source['A']['target'][0],d.soh.iloc[64],places=6)

    def test_future_changes_do_not_change_context(self):
        d,m=self.frame();_,a,_=matched_inputs(d,m,16)
        d.loc[d.cycle.gt(64),'capacity_ah']=1.;d['soh']=d.capacity_ah/2
        _,b,_=matched_inputs(d,m,16)
        np.testing.assert_array_equal(a[0]['context'],b[0]['context'])

    def test_unobserved_bands_remain_nan(self):
        task=dict(battery_id='A',true_rul=40,threshold_soh=.7)
        q=np.full((256,9),.8)
        f=pd.DataFrame(dict(horizon_step=np.arange(1,257),score_eligible=np.arange(256)<40,target_soh=.7,q50=.8))
        rows,diag=trajectory_diagnostics(task,q,f,16,'zero-shot')
        tail=[r for r in rows if r['kind']=='band' and r['start_step']==129][0]
        self.assertEqual(tail['n_scored'],0);self.assertTrue(np.isnan(tail['mae_soh_pp']))
        self.assertAlmostEqual(diag['threshold_margin_at_true_eol_pp'],10)

if __name__=='__main__':unittest.main()
