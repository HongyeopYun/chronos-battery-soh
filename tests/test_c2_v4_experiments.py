import unittest
import numpy as np
import torch
from src.c2_v4_experiments import soh_windows,rul_windows,small_folds,raw_features


def cell(values,eol=None):
    return dict(values=np.asarray(values,float),cycles=np.arange(1,len(values)+1),eol=eol,last=len(values))


class V4ExperimentTests(unittest.TestCase):
    def test_soh_target_never_input_and_common_times(self):
        c=cell(np.arange(100,dtype=float))
        w=soh_windows(c)
        np.testing.assert_array_equal(w['cycle'],np.arange(65,101))
        np.testing.assert_array_equal(w['context'][:,-1]+1,w['target'])
        for length in (2,4,8,16,32,64):
            self.assertTrue(np.all(w['context'][:,-length:]<w['target'][:,None]))

    def test_missing_positions_are_not_compressed(self):
        values=np.arange(100,dtype=float);values[61:64]=np.nan
        w=soh_windows(cell(values))
        self.assertNotIn(65,w['cycle'])
        i=np.flatnonzero(w['cycle']==66)[0]
        self.assertTrue(np.isnan(w['context'][i,-4:-1]).all())
        self.assertEqual(w['context'][i,-1],64.)

    def test_rul_absorbing_eol_and_censoring(self):
        w=rul_windows(cell(np.ones(100),eol=61))
        np.testing.assert_array_equal(w['cycle'],[32,48])
        np.testing.assert_array_equal(w['target'],[29,13])
        censored=rul_windows(cell(np.ones(100)))
        self.assertTrue(np.isnan(censored['target']).all())

    def test_future_changes_do_not_modify_context(self):
        a=cell(np.linspace(1,.7,100),eol=80)
        b=cell(a['values'].copy(),eol=90);b['values'][48:]=.95
        wa,wb=rul_windows(a),rul_windows(b)
        np.testing.assert_array_equal(wa['context'][:2],wb['context'][:2])

    def test_disjoint_cells_and_each_test_once(self):
        folds=small_folds(['a','b','c','d'])
        self.assertEqual(sorted(b for f in folds for b in f['test_ids']),['a','b','c','d'])
        for f in folds:
            self.assertFalse(set(f['train_ids'])&set(f['validation_ids']+f['test_ids']))
            self.assertFalse(set(f['validation_ids'])&set(f['test_ids']))
        with self.assertRaises(ValueError):small_folds(['a','b','c'])

    def test_absolute_level_and_last_valid_are_retained(self):
        x=torch.tensor([[1.,.9,float('nan')]])
        f=raw_features(x,(torch.tensor([[.95]]),torch.tensor([[.05]])))
        torch.testing.assert_close(f,torch.tensor([[.95,.05,.9]]))

    def test_pinball_loss_direction(self):
        from scripts.run_c2_v4_suite import cache_output
        from types import SimpleNamespace
        # Prediction zero, target +1: q=.1 contributes .2, q=.9 contributes 1.8.
        model=SimpleNamespace(num_quantiles=2,quantiles=torch.tensor([.1,.9]),
            output_patch_embedding=lambda h:torch.cat([torch.zeros((1,1,16)),torch.ones((1,1,16))*.5],dim=-1),
            instance_norm=SimpleNamespace(inverse=lambda q,ls:q))
        loss,_=cache_output(model,(torch.zeros(1,1,1),(torch.zeros(1,1),torch.ones(1,1)),torch.ones(1,1)))
        self.assertAlmostEqual(float(loss),.2+.9,places=6)


if __name__=='__main__': unittest.main()
