import unittest
import numpy as np
import pandas as pd
import torch

from src.reviewed_soh import masked_windows,mit_review_features,reviewed_mit_decision
from src.c2_one_step import baselines,make_dataset,one_step_loss
from tests.test_c2_one_step import tiny_model


def rows(n=168):
    c=np.arange(1,n+1)
    return pd.DataFrame(dict(dataset='nasa',battery_id='test',cycle=c,soh=1-c*.001,
                            capacity_ah=1-c*.001,is_observed=True,capacity_kind='full_discharge',
                            quality_flags='',quality_eligible=True,segment_id='test'))


class ReviewedTests(unittest.TestCase):
    def test_one_invalid_point_masks_context_without_removing_following_targets(self):
        d=rows();d.loc[d.cycle.eq(90),'quality_eligible']=False
        w=masked_windows(d)
        np.testing.assert_array_equal(w['forecast_start'],np.r_[np.arange(65,90),np.arange(91,169)])
        k=int(np.flatnonzero(w['forecast_start']==91)[0])
        self.assertTrue(np.isnan(w['context'][k,-1]))
        self.assertEqual(w['context_observed'][k],63)
        self.assertEqual(w['last_observed_lag'][k],2)
        self.assertEqual(w['context_cycle'][k,-1],90)
        self.assertEqual(w['target'][k,0],d.loc[d.cycle.eq(91),'soh'].iloc[0])
        # Future capacity and eligibility cannot alter earlier inputs or masks.
        d.loc[d.cycle.ge(150),['soh','capacity_ah']]=100
        new=masked_windows(d)
        earlier=w['forecast_start']<150
        np.testing.assert_allclose(new['context'][earlier],w['context'][earlier],equal_nan=True)

    def test_real_gaps_segments_and_excluded_targets_are_not_compressed(self):
        d=rows();d.loc[d.cycle.eq(90),'quality_flags']='strict_quality_excluded'
        self.assertNotIn(90,masked_windows(d)['forecast_start'])
        absent=masked_windows(d[d.cycle.ne(90)])['forecast_start']
        self.assertFalse(np.isin(absent,np.arange(90,155)).any())
        d=rows();d.loc[d.cycle.ge(90),'segment_id']='restart'
        self.assertFalse(np.isin(masked_windows(d)['forecast_start'],np.arange(90,154)).any())
        d=rows();d['quality_eligible']=False
        self.assertEqual(len(masked_windows(d)['target']),0)

    def test_missing_baselines_use_original_time_coordinates(self):
        predicted=baselines([[1.,np.nan,3.,np.nan]])
        np.testing.assert_allclose(predicted['LastValue'],[3.])
        np.testing.assert_allclose(predicted['Linear'],[5.])
        with self.assertRaises(ValueError):baselines([[np.nan,1.]])

    def test_native_c2_accepts_internal_and_trailing_missingness(self):
        sequence=np.linspace(1,.9,65,dtype=np.float32)
        sequence[[0,4,31,63]]=np.nan
        for training in (False,True):
            batch=next(iter(make_dataset([dict(target=sequence)],64,training=training)))
            self.assertEqual(int(torch.isnan(batch['context'][0]).sum()),4)
            self.assertEqual(batch['context'].shape[1],64)
            self.assertTrue(torch.isfinite(batch['future_target']).all())
            self.assertTrue(torch.isnan(batch['future_covariates']).all())
            model=tiny_model()
            loss,output=one_step_loss(model,batch)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(output.quantile_preds).all())
            if training:
                loss.backward()
                gradients=[p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(len(gradients)>0)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_long_gap_is_not_integrated_and_counter_reset_across_fragments_is_seen(self):
        i=[.5,.02,0,-4,-4,-4,-4,0]
        v=[3.5,3.6,3.6,3.5,3,2.5,2.,2.1]
        t=[0,1,2,3,4,404,405,406]
        q=[0,0,0,0,.073333,.073333,.146666,.146666]
        f=mit_review_features(i,v,t,q)
        self.assertEqual(f['integral_long_gap_count'],1)
        self.assertAlmostEqual(f['integral_observed_only_ah'],2*4.4/60)
        f.update(raw_capacity_ah=.146666)
        bad,notes=reviewed_mit_decision(f,10)
        self.assertEqual(bad,[])
        self.assertIn('long_sampling_gap_not_integrated',notes)
        i=[.5,.02,0,-4,-4,0,-4,-4,0]
        v=[3.5,3.6,3.6,3.5,3.,3.1,3.1,2.,2.1]
        q=[0,0,0,0,.2,.2,.1,.3,.3]
        f=mit_review_features(i,v,np.arange(9),q);f['raw_capacity_ah']=.3
        bad,_=reviewed_mit_decision(f,10)
        self.assertIn('discharge_counter_reset',bad)

    def test_integral_policy_does_not_release_hard_anomalies(self):
        f=mit_review_features([.5,.02,0,-4,-4,-4,0],[3.5,3.6,3.6,3.5,3.,2.,2.1],np.arange(7),[0,0,0,0,.1,.2,.2])
        f.update(raw_capacity_ah=.2,discharge_integral_ah=.5)
        bad,notes=reviewed_mit_decision(f,10)
        self.assertEqual(bad,[])
        self.assertIn('native_capacity_retained_integral_not_calibrated',notes)
        for change,reason in [({'intervening_charge_samples':3},'interleaved_charge_discharge'),
                              ({'discharge_counter_ah':.5},'summary_counter_mismatch')]:
            self.assertIn(reason,reviewed_mit_decision({**f,**change},10)[0])
        bad,notes=reviewed_mit_decision({**f,'discharge_counter_initial_ah':.02},10)
        self.assertEqual(bad,[])
        self.assertIn('discharge_prefix_unrecorded_native_capacity_retained',notes)


if __name__=='__main__':unittest.main()
