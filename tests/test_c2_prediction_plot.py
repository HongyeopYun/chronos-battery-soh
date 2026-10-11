"""The measured trajectory must not disappear when evaluation windows are absent."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import run_c2_one_step_context as runner


class PredictionPlotTests(unittest.TestCase):
    def test_missing_predictions_do_not_hide_actual_soh(self):
        cycles=np.arange(1,169)
        truth=pd.DataFrame(dict(battery_id='test',cycle=cycles,soh=1-cycles*.001,
                                capacity_ah=1-cycles*.001,is_observed=True,quality_eligible=cycles!=90))
        targets=np.r_[np.arange(65,90),np.arange(155,169)]
        predictions=pd.DataFrame([dict(battery_id='test',context_length=64,method=method,
                                      target_cycle=t,target=1-t*.001,prediction=1-(t-1)*.001)
                                  for method in ('LastValue','zero-shot','head-only') for t in targets])
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'testgroup').mkdir()
            with patch.object(runner,'OUT',Path(folder)), patch.object(runner,'load_bundle',return_value=(truth,None,None)), patch.object(runner.plt,'close'):
                runner.plot_predictions('testgroup',predictions,dict(folds=[dict(test_ids=['test'])]))
                figure=runner.plt.gcf()
                actual,*forecasts=figure.axes[0].lines
                np.testing.assert_array_equal(actual.get_xdata(),cycles)
                np.testing.assert_allclose(actual.get_ydata(),truth.soh)
                for line in forecasts:
                    values=pd.Series(line.get_ydata(),index=line.get_xdata())
                    self.assertTrue(values.loc[90:154].isna().all())
                    self.assertEqual(values.notna().sum(),len(targets))
                self.assertTrue((Path(folder)/'testgroup/representative.png').exists())
            runner.plt.close(figure)


if __name__=='__main__':
    unittest.main()
