import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts import run_c2_context16 as runner
from scripts.report_c2_context16 import comparison_tables, write_report
from src.c2_one_step import fixed_inputs, make_dataset, one_step_loss
from tests.test_c2_one_step import tiny_model


class Context16Tests(unittest.TestCase):
    def test_sixteen_position_mask_label_and_head_gradient(self):
        x = np.linspace(1, .9, 64, dtype=np.float32)[None, :]
        x[0, [48, 61, 63]] = np.nan
        w = dict(context=x, context_cycle=np.arange(1,65)[None,:],
                 target=np.array([[.89]],dtype=np.float32), target_cycle=np.array([[65]]),
                 context_observed=np.array([61]), last_observed_lag=np.array([2]))
        cropped = runner.cropped(w,16)
        np.testing.assert_array_equal(cropped['context_cycle'],np.arange(49,65)[None,:])
        np.testing.assert_array_equal(cropped['target'],w['target'])
        self.assertEqual(cropped['context_observed'][0],13)
        self.assertEqual(cropped['last_observed_lag'][0],2)
        np.testing.assert_array_equal(runner.persistence(cropped['context'])['LastValue'],x[:,62])
        batch = next(iter(make_dataset(fixed_inputs(cropped,[0]),16)))
        model = tiny_model()
        loss, output = one_step_loss(model,batch)
        self.assertEqual(batch['context'].shape[1],16)
        self.assertEqual(batch['future_target'].shape[1],1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(output.quantile_preds).all())

    def test_all_missing_short_history_fails_instead_of_changing_targets(self):
        w = dict(context=np.full((1,64),np.nan),context_cycle=np.arange(1,65)[None,:],
                 target=np.array([[.8]]),target_cycle=np.array([[65]]))
        w['context'][0,0] = 1.
        with self.assertRaisesRegex(ValueError,'no observed input'):
            runner.cropped(w,16)

    def test_cuda_run_never_silently_falls_back_to_cpu(self):
        with patch.object(runner.base.torch.cuda,'is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA is required'):
                runner.runtime_info('cuda:0')

    def test_report_uses_matched_targets_and_fixed_lastvalue_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folders = []
            for length in (64,32,16):
                folder = root/f'results/c2-battery/one-step-{length}-reviewed-v3'
                (folder/'sample').mkdir(parents=True)
                folders.append((length,folder))
                rows, scores = [], []
                for method, errors in [('LastValue',[.1,.1,.1]),('zero-shot',[.05]*3),('head-only',[.025]*3)]:
                    y = np.array([1.,.9,.8])
                    p = y+errors
                    for i in range(3):
                        rows.append(dict(dataset='sample',context_length=length,battery_id='A',
                                         target_cycle=i+65,phase='middle',target=y[i],prediction=p[i],method=method))
                    for phase in ('all','middle'):
                        scores.append(dict(dataset='sample',context_length=length,method=method,
                                           phase=phase,**runner.base.point_metrics(y,p)))
                pd.DataFrame(rows).to_csv(folder/'sample/predictions.csv.gz',index=False)
                pd.DataFrame(scores).to_csv(folder/'summary.csv',index=False)
            _,_,_,delta = comparison_tables(folders,['sample'])
            np.testing.assert_allclose(delta.lastvalue_skill_change_pp,0)
            out = folders[-1][1]
            write_report(out,dict(groups={'sample':{}}),pd.read_csv(out/'summary.csv'))
            self.assertIn('Test R²',(out/'report.md').read_text(encoding='utf-8'))
            self.assertTrue((out/'comparison.png').stat().st_size>0)
            p = pd.read_csv(out/'sample/predictions.csv.gz')
            p.loc[p.method.eq('LastValue'),'prediction'] += .001
            p.to_csv(out/'sample/predictions.csv.gz',index=False)
            with self.assertRaises(AssertionError):
                comparison_tables(folders,['sample'])


if __name__ == '__main__':
    unittest.main()
