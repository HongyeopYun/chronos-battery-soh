import unittest
import numpy as np
import pandas as pd
import torch

from scripts.run_c2_context32 import cropped, persistence
from src.c2_context_comparison import paired_metrics, summarize_paired
from src.c2_one_step import fixed_inputs, make_dataset, one_step_loss
from tests.test_c2_one_step import tiny_model


class ContextComparisonTests(unittest.TestCase):
    def test_cropped_mask_and_fixed_single_label_reach_model(self):
        x=np.linspace(1,.9,64,dtype=np.float32)[None,:]
        x[0,[0,32,45,63]]=np.nan
        w=dict(context=x,context_cycle=np.arange(1,65)[None,:],target=np.array([[.89]],dtype=np.float32),
               target_cycle=np.array([[65]]),context_observed=np.array([60]),last_observed_lag=np.array([2]))
        c=cropped(w,32)
        self.assertEqual(c['context_observed'][0],29)
        self.assertEqual(c['last_observed_lag'][0],2)
        np.testing.assert_array_equal(c['context_cycle'],np.arange(33,65)[None,:])
        np.testing.assert_array_equal(c['target'],w['target'])
        np.testing.assert_array_equal(persistence(c['context'])['LastValue'],x[:,62])
        for training in (False,True):
            batch=next(iter(make_dataset(fixed_inputs(c,[0]),32,training=training)))
            self.assertEqual(batch['context'].shape[1],32)
            self.assertEqual(batch['future_target'].shape[1],1)
            self.assertTrue(torch.isnan(batch['future_covariates']).all())
            model=tiny_model();loss,output=one_step_loss(model,batch)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(output.quantile_preds).all())
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(w['context_observed'][0],60)

    def fixture(self):
        rows=[]
        # Mean of per-record skill is 0.25; pooled or ratio-of-mean errors differ.
        for b,last,model in [('a',1.,1.),('b',2.,np.sqrt(2))]:
            for method,error in [('LastValue',last),('zero-shot',model),('head-only',model)]:
                for t in (65,66):rows.append(dict(dataset='x',context_length=32,battery_id=b,
                    method=method,target_cycle=t,phase='late',target=1.,prediction=1.+error))
        return pd.DataFrame(rows)

    def test_ratios_are_per_record_and_zero_baseline_is_undefined(self):
        p=self.fixture();b=paired_metrics(p);s=summarize_paired(b)
        r=s[s.method.eq('head-only')&s.phase.eq('all')].iloc[0]
        self.assertAlmostEqual(r.mse_skill_lastvalue,.25)
        p.loc[p.battery_id.eq('a')&p.method.eq('LastValue'),'prediction']=1.
        s=summarize_paired(paired_metrics(p))
        r=s[s.method.eq('head-only')&s.phase.eq('all')].iloc[0]
        self.assertEqual(r.mse_skill_lastvalue_undefined,1)
        self.assertAlmostEqual(r.mse_skill_lastvalue,.5)

    def test_mismatched_timestamps_are_rejected(self):
        p=self.fixture()
        with self.assertRaises(ValueError):paired_metrics(p.drop(p.index[0]))


if __name__=='__main__':unittest.main()
