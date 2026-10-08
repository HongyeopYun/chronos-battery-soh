"""Finish the two low-priority T5 groups using at most three windows per record."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
import pandas as pd
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, baseline_predictions, write_json, sha256
from src.battery_model_zeroshot import load_model, phase_labels, PHASES
from src.battery_zeroshot_minimal import SCORES, verify_compact_result
from src.soh_evaluation import window_r2_values, R2_DEFINITION
from src.soh_experiment import predict_samples, summarize_samples, clear_device_cache


def finish(group, pipeline):
    directory = ROOT / 'results/t5-battery' / group
    manifest = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    config = ZeroShotConfig(**manifest['config'])
    if sha256(Path(manifest['bundle']) / 'manifest.json') != manifest['bundle_fingerprint']:
        raise ValueError('Input fingerprint differs')
    records, _, data, _ = prepare_battery_windows(manifest['bundle'], config)
    selected = []
    for record in records:
        w = record['windows']
        cycles = data[(data.dataset == record['dataset']) & (data.battery_id == record['battery_id']) & data.is_observed].cycle.to_numpy()
        labels = phase_labels(w['forecast_start'], cycles)
        centers = cycles.min() + (np.arange(3) + .5) * (cycles.max() - cycles.min() + 1) / 3
        indices, phases = [], []
        for phase, center in zip(PHASES, centers):
            eligible = np.flatnonzero(labels == phase)
            if len(eligible):
                indices.append(eligible[np.abs(w['forecast_start'][eligible] - center).argmin()])
                phases.append(phase)
        selected.append({**record, 'windows': {k:v[indices] for k,v in w.items()}, 'phases': phases})
    contexts = np.concatenate([r['windows']['context'] for r in selected])
    print(f'REPRESENTATIVE {group}: {len(records)} records, {len(contexts)} windows', flush=True)
    point = summarize_samples(predict_samples(pipeline, contexts, config, config.inference_seeds[0]))['median']
    frame = pd.read_csv(directory / 'battery_metrics.csv', keep_default_na=False, na_values=[''])
    original = frame.copy()
    frame = frame.drop(columns=['r2_within'], errors='ignore')
    frame['r2'] = np.nan
    frame['r2_scope'] = 'representative_windows'
    frame['r2_n_windows'] = 0
    frame['r2_undefined_windows'] = 0
    rows, targets, starts, ids = [], [], [], []
    cursor = 0
    for record in selected:
        w = record['windows']; n = len(w['target'])
        forecasts = {**baseline_predictions(w), 'Chronos': point[cursor:cursor+n]}
        cursor += n
        mask = (frame.dataset == record['dataset']) & (frame.battery_id == record['battery_id'])
        if set(frame.loc[mask, 'model']) != set(forecasts):
            raise ValueError('Missing battery/model rows')
        for model, prediction in forecasts.items():
            values = window_r2_values(w['target'], prediction)
            valid = np.isfinite(values)
            item = mask & (frame.model == model)
            frame.loc[item, 'r2'] = values[valid].mean() if valid.any() else np.nan
            frame.loc[item, 'r2_n_windows'] = int(valid.sum())
            frame.loc[item, 'r2_undefined_windows'] = int(np.isnan(values).sum())
            for i in range(n):
                rows.append({'battery_id': record['battery_id'], 'model': model, 'phase': record['phases'][i],
                             'forecast_start': int(w['forecast_start'][i]), 'r2': values[i],
                             'representative_rmse': float(np.sqrt(np.square(w['target'][i] - prediction[i]).mean()))})
        targets.extend(w['target']); starts.extend(w['forecast_start']); ids.extend([record['battery_id']]*n)
    unchanged = [c for c in original if c not in ['r2', 'r2_within', 'r2_n_windows', 'r2_undefined_windows', 'r2_scope']]
    pd.testing.assert_frame_equal(frame[unchanged], original[unchanged])
    frame.to_csv(directory / 'battery_metrics.csv', index=False)
    summary = frame[frame.status == 'evaluated'].groupby('model', as_index=False)[SCORES].mean()
    summary['r2_scope'] = 'representative_windows'
    summary.to_csv(directory / 'summary.csv', index=False)
    pd.DataFrame(rows).to_csv(directory / 'representative_r2_windows.csv', index=False)
    np.savez_compressed(directory / 'representative_r2_predictions.npz', target=np.asarray(targets), median=point,
                        forecast_start=np.asarray(starts), battery_id=np.asarray(ids))
    manifest.update(r2_definition=R2_DEFINITION, r2_scope='representative_windows', r2_selected_windows=len(contexts),
                    r2_updated_at=datetime.now(timezone.utc).isoformat(),
                    r2_source='New representative-window inference with saved 100-sample median settings; one eligible window nearest each observed-period third midpoint. Full original non-R2 metrics retained.')
    write_json(directory / 'run.json', manifest)
    verify_compact_result(directory)
    print(f'DONE {group}: representative R2={summary[summary.model=="Chronos"].r2.iloc[0]:.6f}', flush=True)


def main():
    status_path = ROOT / 'results/window-r2/status.json'
    state = json.loads(status_path.read_text(encoding='utf-8'))
    state.update(status='running', mode='C2 full; remaining T5 representative windows', stopped_full_t5_at=state.get('current_battery'))
    write_json(status_path, state)
    config = ZeroShotConfig(**json.loads((ROOT/'results/t5-battery/mit_batch1/run.json').read_text(encoding='utf-8'))['config'])
    pipeline = load_model(config)
    try:
        for group in ['mit_batch1', 'mit_batch3']:
            finish(group, pipeline)
            path = str(Path('results/t5-battery') / group)
            if path not in state['completed_runs']:
                state['completed_runs'].append(path)
            state['current_run'] = path
            write_json(status_path, state)
        state.update(status='complete', current_battery=None, current_windows=0, finished_at=datetime.now(timezone.utc).isoformat(),
                     note='18 packages use all windows; T5 MIT batch1/3 R2 uses representative windows only by user request')
    except Exception as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        write_json(status_path, state)
        del pipeline
        clear_device_cache()


if __name__ == '__main__':
    main()
