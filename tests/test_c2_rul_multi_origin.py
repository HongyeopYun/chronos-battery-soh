import unittest
import numpy as np
import pandas as pd
from scripts.run_c2_rul_multi_origin import build_source, expand_fold

class MultiOriginTests(unittest.TestCase):
    def frame(self):
        d=pd.DataFrame(dict(battery_id=['A']*130,cycle=np.arange(1,131),capacity_ah=np.linspace(2,1.5,130),
            quality_eligible=True,is_observed=True,capacity_kind='full_discharge'))
        d['soh']=d.capacity_ah/2
        return d,pd.DataFrame(dict(battery_id=['A'],reference_capacity_ah=[2.]))

    def test_fixed_length_and_target_alignment(self):
        d,m=self.frame();s=build_source(d,m,[64,80,96])
        for origin in [64,80,96]:
            r=s[f'A@{origin}']
            np.testing.assert_allclose(r['context'],d.soh.iloc[origin-64:origin],rtol=1e-6)
            self.assertEqual(len(r['context']),64)
            self.assertAlmostEqual(r['target'][0],d.soh.iloc[origin],places=6)
            self.assertTrue(np.isnan(r['target'][130-origin:]).all())
        d.loc[d.cycle.eq(80),'quality_eligible']=False
        with self.assertRaises(AssertionError):build_source(d,m,[64,80,96])

    def test_future_cannot_change_admission_or_context(self):
        d,m=self.frame();a=build_source(d,m,[64,80,96])
        d.loc[d.cycle.gt(96),'capacity_ah']=1.;d['soh']=d.capacity_ah/2
        b=build_source(d,m,[64,80,96])
        self.assertEqual(set(a),set(b))
        for k in a:np.testing.assert_array_equal(a[k]['context'],b[k]['context'])

    def test_past_eol_blocks_later_origins(self):
        d,m=self.frame();d.loc[d.cycle.eq(79),'capacity_ah']=1.3;d['soh']=d.capacity_ah/2
        with self.assertRaises(AssertionError):build_source(d,m,[64,80,96])

    def test_physical_split_and_fixed_validation(self):
        s={f'{b}@{t}':dict(battery_id=b,origin_cycle=t) for b in ['A','B','C','D'] for t in [64,80,96]}
        f=expand_fold(dict(train_ids=['A','B'],validation_ids=['C'],test_ids=['D']),s)
        self.assertEqual(len(f['train_ids']),6)
        self.assertEqual(f['validation_ids'],['C@64'])
        self.assertTrue(all(s[k]['battery_id'] in ['A','B'] for k in f['train_ids']))

if __name__=='__main__':unittest.main()
