import unittest
import numpy as np
import torch
from src.c2_rul_improvement import fit_scaler,transform,new_head,output,loss_value


class ImprovementTests(unittest.TestCase):
    def test_training_only_scaler_and_cell_balance(self):
        x=np.array([[0.,1.],[2.,3.],[100.,101.]])
        s=fit_scaler(x,['a','a','b'],'all')
        np.testing.assert_allclose(s['center'],[50.5,51.5])
        before=s['center'].copy();transform(np.array([[100000.,-100000.]]),s)
        np.testing.assert_array_equal(s['center'],before)

    def test_metadata_scaling_keeps_encoder_coordinates(self):
        x=np.arange(30,dtype=float).reshape(5,6)
        s=fit_scaler(x,['a']*5,'metadata');z=transform(x,s)
        np.testing.assert_array_equal(z[:,:3],x[:,:3]);np.testing.assert_allclose(z[:,3:].mean(0),0,atol=1e-6)

    def test_unchanged_head_initial_constant_and_gradients(self):
        head=new_head(771,20,30,42,'cpu');x=torch.randn(8,771)
        pred=output(head,x,20);torch.testing.assert_close(pred,torch.full((8,),30.))
        for name in ['mae','mse','huber1','huber01']:
            head.zero_grad();loss=loss_value(output(head,x,20),torch.arange(8.)*3,20,name);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(any(p.grad.abs().sum()>0 for p in head.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()
