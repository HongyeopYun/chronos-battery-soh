"""Independent strict release audit of the saved two-position experiment."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts import run_c2_context2 as runner


def main():
    runner.configure()
    out = runner.OUT
    plan = runner.base.read(out/'protocol.json')
    assert runner.base.read(out/'status.json')['status'] == 'complete'
    result = runner.verify(plan)
    saved = pd.read_csv(out/'lastvalue_battery_metrics.csv')
    saved = saved[saved.context_length.eq(2)].set_index(['dataset','battery_id','method','phase'])
    count = 0
    for group in plan['groups']:
        p = pd.read_csv(out/group/'predictions.csv.gz')
        for battery, rows in p.groupby('battery_id'):
            for phase in ('all','early','middle','late'):
                part = rows if phase == 'all' else rows[rows.phase.eq(phase)]
                if part.empty:
                    continue
                y = part.groupby('target_cycle').target.first()
                predictions = part.pivot(index='target_cycle',columns='method',values='prediction').reindex(y.index)
                mse = predictions.sub(y,axis=0).pow(2).mean()
                for method in runner.METHODS:
                    expected = 1-mse[method]/mse['LastValue'] if mse['LastValue']>0 else np.nan
                    actual = saved.loc[(group,battery,method,phase),'mse_skill_lastvalue']
                    np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-12,equal_nan=True)
                    count += 1
    result.update(strict_mse_skill_rtol=1e-8,strict_mse_skill_atol=1e-12,
                  strict_mse_skill_rows=count,
                  note='Independent strict audit supplements the frozen runner metric tolerance.')
    (out/'local_release_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
