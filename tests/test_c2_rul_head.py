import unittest
import numpy as np
import pandas as pd
import torch
from src.c2_rul_head import source_trajectories, masked_quantile_loss


class RulHeadTests(unittest.TestCase):
    def test_nan_padding_does_not_change_loss_or_receive_gradient(self):
        p=torch.tensor([[[0.,7.,9.]]],requires_grad=True)
        y=torch.tensor([[1.,float('nan'),float('nan')]])
        loss=masked_quantile_loss(p,y,torch.tensor([.5]))
        self.assertAlmostEqual(float(loss),1.)
        loss.backward()
        torch.testing.assert_close(p.grad,torch.tensor([[[-1.,0.,0.]]]))

    def test_cells_have_equal_weight_with_different_observation_counts(self):
        p=torch.zeros(2,1,3)
        y=torch.tensor([[1.,float('nan'),float('nan')],[3.,3.,3.]])
        self.assertAlmostEqual(float(masked_quantile_loss(p,y,torch.tensor([.5]))),2.)

    def test_empty_supervision_is_rejected(self):
        with self.assertRaises(ValueError):
            masked_quantile_loss(torch.zeros(1,1,2),torch.full((1,2),float('nan')),torch.tensor([.5]))

    def test_partial_future_ends_at_last_observed_record(self):
        frame=pd.DataFrame(dict(battery_id=['a']*5,cycle=np.arange(1,6),soh=[1.,.98,.96,.94,.92],
                                capacity_ah=[2.,1.96,1.92,1.88,1.84],quality_eligible=True,is_observed=True,
                                capacity_kind='full_discharge'))
        frame.loc[3,'quality_eligible']=False
        record=source_trajectories(frame,context_length=3,horizon=4)['a']
        np.testing.assert_allclose(record['context'],[1.,.98,.96],rtol=1e-6)
        np.testing.assert_allclose(record['target'],[np.nan,.92,np.nan,np.nan],rtol=1e-6,equal_nan=True)
        frame.loc[4,'soh']=.8
        changed=source_trajectories(frame,context_length=3,horizon=4)['a']
        np.testing.assert_array_equal(record['context'],changed['context'])


if __name__=='__main__':unittest.main()
