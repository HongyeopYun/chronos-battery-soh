import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from src.battery_zeroshot import ZeroShotConfig
from src.battery_model_zeroshot import run_model_battery, verify_result, PROTOCOL
from scripts.unify_window_r2 import migrate
from test_battery_zeroshot import save_bundle
from test_battery_model_zeroshot import fake_quantiles, fake_figures


class WindowR2MigrationTests(unittest.TestCase):
    def test_migration_preserves_other_metrics_and_rejects_changed_forecasts(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            bundle = save_bundle(base/'input', sizes=(100, 140))
            cfg = ZeroShotConfig(model_id='amazon/chronos-2', context_length=3, prediction_length=2, inference_seeds=(42,))
            with patch('src.battery_model_zeroshot.load_model', return_value=object()), \
                 patch('src.battery_model_zeroshot.predict_quantiles', side_effect=fake_quantiles), \
                 patch('src.battery_zeroshot_plots.forecast_figures', side_effect=fake_figures), \
                 patch('src.battery_model_zeroshot.clear_device_cache'):
                directory = run_model_battery(bundle,cfg,base/'output',scope='full')
            path = directory/'c2-battery_run.json'
            m = json.loads(path.read_text())
            m['protocol']={'r2':'retired pooled definition'}
            path.write_text(json.dumps(m))
            metrics_path = directory/'c2-battery_battery_metrics.csv'
            before = pd.read_csv(metrics_path,keep_default_na=False,na_values=[''])
            broken = before.copy()
            broken['r2']=0.999
            broken.to_csv(metrics_path,index=False)
            state={}
            def point(record,config,pipeline,compact):
                return fake_quantiles(pipeline,record['windows']['context'],config)[0]
            with patch('scripts.unify_window_r2.STATE',base/'state'), \
                 patch('scripts.unify_window_r2.ROOT',base), \
                 patch('scripts.unify_window_r2.predict',side_effect=point):
                migrate(path,object(),state)
            after = pd.read_csv(metrics_path,keep_default_na=False,na_values=[''])
            pd.testing.assert_frame_equal(before,after,check_dtype=False,atol=1e-12,rtol=1e-10)
            self.assertEqual(json.loads(path.read_text())['protocol'],PROTOCOL)
            verify_result(directory,'c2-battery')
            m['protocol']={'r2':'retired pooled definition'}
            path.write_text(json.dumps(m))
            with patch('scripts.unify_window_r2.STATE',base/'state'), \
                 patch('scripts.unify_window_r2.ROOT',base), \
                 patch('scripts.unify_window_r2.predict',side_effect=lambda *args: point(*args)+.1):
                with self.assertRaises(AssertionError):
                    migrate(path,object(),state)
            pd.testing.assert_frame_equal(pd.read_csv(metrics_path,keep_default_na=False,na_values=['']),after)


if __name__=='__main__':
    unittest.main()
