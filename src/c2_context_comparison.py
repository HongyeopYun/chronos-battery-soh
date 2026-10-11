"""Paired persistence comparisons for matched one-step prediction tables."""
import numpy as np
import pandas as pd

METHODS = ('LastValue', 'zero-shot', 'head-only')


def paired_metrics(predictions):
    """Compute ratios within each record, never from macro-averaged RMSE/R2."""
    rows = []
    for (dataset, length, battery), frame in predictions.groupby(
            ['dataset', 'context_length', 'battery_id'], sort=False):
        if frame.duplicated(['method', 'target_cycle']).any():
            raise ValueError('Duplicate method/target pair')
        if set(frame.method) != set(METHODS):
            raise ValueError('Expected LastValue and the two C2 conditions only')
        if not frame.groupby('target_cycle').target.nunique().eq(1).all():
            raise ValueError('Different targets between methods')
        if not frame.groupby('target_cycle').phase.nunique().eq(1).all():
            raise ValueError('Different phases between methods')
        for phase in ('all', 'early', 'middle', 'late'):
            part = frame if phase == 'all' else frame[frame.phase.eq(phase)]
            if part.empty:
                continue
            point = part.pivot(index='target_cycle', columns='method', values='prediction')
            if not np.isfinite(point.to_numpy()).all():
                raise ValueError('Missing or nonfinite matched prediction')
            target = part.groupby('target_cycle').target.first().reindex(point.index)
            error = point.sub(target, axis=0)
            mse, mae = error.pow(2).mean(), error.abs().mean()
            for method in METHODS:
                row = dict(dataset=dataset, context_length=length, battery_id=battery,
                           phase=phase, method=method, n_targets=len(point),
                           mse=float(mse[method]), mae=float(mae[method]),
                           rmse=float(np.sqrt(mse[method])),
                           lastvalue_mse=float(mse['LastValue']), lastvalue_mae=float(mae['LastValue']))
                for reference, suffix in [('LastValue', 'lastvalue'), ('zero-shot', 'zero')]:
                    row[f'mse_skill_{suffix}'] = float(1-mse[method]/mse[reference]) if mse[reference] > 0 else np.nan
                row['mae_skill_lastvalue'] = float(1-mae[method]/mae['LastValue']) if mae['LastValue'] > 0 else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_paired(battery):
    rows = []
    for keys, frame in battery.groupby(['dataset', 'context_length', 'method', 'phase'], sort=False):
        row = dict(zip(['dataset', 'context_length', 'method', 'phase'], keys))
        row.update(n_batteries=len(frame), n_targets=int(frame.n_targets.sum()),
                   mae=float(frame.mae.mean()), rmse=float(frame.rmse.mean()))
        for metric in ('mse_skill_lastvalue', 'mae_skill_lastvalue', 'mse_skill_zero'):
            value = frame[metric]
            row[metric] = float(value.mean())
            row[metric+'_median'] = float(value.median())
            row[metric+'_defined'] = int(value.notna().sum())
            row[metric+'_undefined'] = int(value.isna().sum())
            row[metric+'_wins'] = int(value.gt(0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def compare_lengths(battery):
    keys = ['dataset', 'battery_id', 'method', 'phase']
    a = battery[battery.context_length.eq(64)].set_index(keys)
    b = battery[battery.context_length.eq(32)].set_index(keys)
    if set(a.index) != set(b.index):
        raise ValueError('Different evaluation records across context lengths')
    b = b.reindex(a.index)
    np.testing.assert_array_equal(a.n_targets, b.n_targets)
    result = a[['n_targets']].copy()
    for metric in ('mse', 'mae', 'rmse'):
        result[metric+'_64'] = a[metric]
        result[metric+'_32'] = b[metric]
    result['mse_skill_32_vs_64'] = 1-b.mse/a.mse.where(a.mse.gt(0))
    return result.reset_index()
