import numpy as np
import unittest
from src.c2_rul_context_nominal import windows,assemble

def test_matched_history_excludes_future_and_eol():
    c=dict(last=180,eol=150,values=np.arange(180,dtype=float))
    a=windows(c,8,128,16);b=windows(c,128,128,16)
    np.testing.assert_array_equal(a['cycle'],[128,144])
    np.testing.assert_array_equal(a['target'],[22,6])
    np.testing.assert_array_equal(a['context'],b['context'][:,-8:])
    c['values'][144:]=-999
    np.testing.assert_array_equal(windows(c,128,128,16)['context'],b['context'])

def test_censoring_is_not_an_event_and_missing_positions_preserved():
    c=dict(last=160,eol=None,values=np.ones(160));c['values'][126]=np.nan
    a=windows(c,8,128,16)
    assert np.isnan(a['target']).all() and np.isnan(a['context'][0,-2])
    assert a['cycle'][-1]==160

def test_same_source_labels_for_every_length():
    c=dict(last=180,eol=170,values=np.ones(180));cells={('g','b'):c}
    a=assemble(cells,'g',['b'],'16',length=8,start=128)
    b=assemble(cells,'g',['b'],'16',length=128,start=128)
    np.testing.assert_array_equal(a['cycle'],b['cycle'])
    np.testing.assert_array_equal(a['target'],b['target'])
    assert len(a['target'])==16

def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(f) for name,f in globals().items() if name.startswith('test_'))

if __name__=='__main__':unittest.main()
