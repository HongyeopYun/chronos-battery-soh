"""Migrate all retained battery summaries to arithmetic mean of window R2.

Missing raw forecasts are inferred with each saved configuration. Small temporary
point caches allow identical full/pilot windows to share the same prediction.
Other metrics and representative curves are preserved and RMSE is cross-checked.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
import pandas as pd
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, baseline_predictions, write_json, sha256
from src.battery_model_zeroshot import PROTOCOL, load_model, predict_quantiles, phase_labels, write_tables, write_report
from src.battery_zeroshot_minimal import R2_DEFINITION, write_compact_report, GROUPS, write_suite_index
from src.soh_evaluation import mean_window_r2, window_r2_values
from src.soh_experiment import predict_samples, summarize_samples, clear_device_cache

STATE = ROOT / 'results/window-r2'
CACHE = STATE / '.point_cache'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def predict(record, cfg, pipeline, compact):
    w = record['windows']
    identity = asdict(cfg)
    for key in ['battery_ids', 'max_windows_per_battery', 'calce_cross_file_windows']:
        identity.pop(key, None)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    digest.update(w['context'].tobytes())
    digest.update(w['target'].tobytes())
    path = CACHE / (digest.hexdigest() + '.npz')
    if path.exists():
        with np.load(path) as saved:
            return saved['median64' if compact else 'median32'].copy()
    if 't5' in cfg.model_id:
        samples = predict_samples(pipeline, w['context'], cfg, cfg.inference_seeds[0])
        median64 = summarize_samples(samples)['median']
        median32 = np.median(samples, axis=1)
    else:
        median32, _ = predict_quantiles(pipeline, w['context'], cfg)
        median64 = median32
    np.savez_compressed(path, median64=median64, median32=median32)
    return median64 if compact else median32


def migrate(path, pipeline, state):
    m = read(path)
    compact = path.name == 'run.json'
    if (m.get('r2_definition') == R2_DEFINITION if compact else m.get('protocol') == PROTOCOL):
        return
    cfg = ZeroShotConfig(**m['config'])
    if len(cfg.inference_seeds) != 1:
        raise ValueError('Battery migration requires one saved inference seed')
    if sha256(Path(m['bundle']) / 'manifest.json') != m['bundle_fingerprint']:
        raise ValueError('Saved input fingerprint differs')
    records, coverage, data, _ = prepare_battery_windows(Path(m['bundle']), cfg)
    tag = 't5-battery' if 't5' in cfg.model_id else 'c2-battery'
    directory = path.parent
    metrics_path = directory / ('battery_metrics.csv' if compact else f'{tag}_battery_metrics.csv')
    frame = pd.read_csv(metrics_path, keep_default_na=False, na_values=[''])
    if 'r2_within' in frame:
        frame = frame.drop(columns='r2_within')
    frame['r2'] = np.nan
    frame['r2_n_windows'] = 0
    frame['r2_undefined_windows'] = 0
    progress_path = directory / '.window_r2_progress.csv'
    done = set()
    if progress_path.exists():
        previous = pd.read_csv(progress_path, keep_default_na=False, na_values=[''])
        for record in records:
            mask = (previous.dataset == record['dataset']) & (previous.battery_id == record['battery_id'])
            if mask.any() and (previous.loc[mask, 'r2_n_windows'] + previous.loc[mask, 'r2_undefined_windows']).gt(0).all():
                done.add(record['battery_id'])
                frame.loc[mask, ['r2', 'r2_n_windows', 'r2_undefined_windows']] = previous.loc[mask, ['r2', 'r2_n_windows', 'r2_undefined_windows']]
    for record in records:
        if record['battery_id'] in done:
            continue
        state.update(current_run=str(directory.relative_to(ROOT)), current_battery=record['battery_id'],
                     current_windows=len(record['windows']['target']))
        write_json(STATE / 'status.json', state)
        tick = time.perf_counter()
        print(f"R2 {tag}/{directory.name}/{record['battery_id']} windows={state['current_windows']}", flush=True)
        point = predict(record, cfg, pipeline, compact)
        forecasts = {**baseline_predictions(record['windows']), ('Chronos' if compact else tag): point}
        series = data[(data.dataset == record['dataset']) & (data.battery_id == record['battery_id']) & data.is_observed]
        labels = phase_labels(record['windows']['forecast_start'], series.cycle.to_numpy())
        selected = (frame.dataset == record['dataset']) & (frame.battery_id == record['battery_id'])
        if set(frame.loc[selected, 'model']) != set(forecasts):
            raise ValueError('Missing battery identifiers or models in retained metric table')
        for index, row in frame.loc[selected].iterrows():
            mask = np.ones(len(labels), bool) if compact or row.phase == 'all' else labels == row.phase
            target, pred = record['windows']['target'][mask], forecasts[row.model][mask]
            rmse = np.sqrt(np.square(target - pred).mean())
            # A different estimator/config cannot silently produce R2 for an old RMSE.
            np.testing.assert_allclose(rmse, row.rmse, rtol=2e-5, atol=5e-8,
                                       err_msg=f'Prediction differs: {directory}/{record["battery_id"]}/{row.model}')
            values = window_r2_values(target, pred)
            frame.loc[index, 'r2'] = mean_window_r2(target, pred)
            frame.loc[index, 'r2_n_windows'] = np.isfinite(values).sum()
            frame.loc[index, 'r2_undefined_windows'] = np.isnan(values).sum()
        frame.to_csv(progress_path, index=False)
        print(f"DONE {record['battery_id']} elapsed={time.perf_counter()-tick:.1f}s", flush=True)
    if compact:
        frame.to_csv(metrics_path, index=False)
        summary = frame[frame.status == 'evaluated'].groupby('model', as_index=False)[['rmse','r2','wql','picp80','width80']].mean()
        summary.to_csv(directory / 'summary.csv', index=False)
        m['r2_definition'] = R2_DEFINITION
        write_compact_report(directory, cfg, coverage, summary, m['figures'])
    else:
        summary = write_tables(directory, tag, frame.to_dict('records'))
        m['protocol'] = PROTOCOL
        write_report(directory, tag, cfg, summary, m['figures'], m.get('run_scope','pilot'), coverage)
    m['r2_updated_at'] = datetime.now(timezone.utc).isoformat()
    m['r2_source'] = 'Per-window predictions re-inferred with saved input/model/seed/batch settings; original RMSE cross-checked'
    write_json(path, m)
    progress_path.unlink(missing_ok=True)


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    paths = sorted((ROOT / 'results/c2-battery').rglob('c2-battery_run.json'))
    # New CALCE and small pilot first. Full T5 reuses matching pilot predictions.
    paths += sorted((ROOT / 'results/t5-battery').rglob('t5-battery_run.json'))
    paths += [ROOT / 'results/t5-battery' / g / 'run.json' for g in GROUPS]
    state = {'status':'running','runs_total':len(paths),'completed_runs':[], 'started_at':datetime.now(timezone.utc).isoformat()}
    pipeline, family = None, None
    try:
        for path in paths:
            m = read(path)
            if m['status'] != 'complete':
                raise ValueError(f'Incomplete source: {path}')
            compact = path.name == 'run.json'
            if (m.get('r2_definition') == R2_DEFINITION if compact else m.get('protocol') == PROTOCOL):
                state['completed_runs'].append(str(path.parent.relative_to(ROOT)))
                continue
            if family != m['config']['model_id']:
                del pipeline
                clear_device_cache()
                cfg = ZeroShotConfig(**m['config'])
                pipeline, family = load_model(cfg), cfg.model_id
            migrate(path, pipeline, state)
            state['completed_runs'].append(str(path.parent.relative_to(ROOT)))
            write_json(STATE / 'status.json', state)
        from src.battery_model_zeroshot import write_model_suite, write_comparison
        write_model_suite(ROOT / 'results/c2-battery/full', 'c2-battery', GROUPS)
        write_comparison()
        write_suite_index(ROOT / 'results/t5-battery')
        from scripts.run_calce_zeroshot import write_comparison as calce_comparison
        calce_comparison(ROOT / 'results/t5-c2-battery-comparison/calce64-16')
        state['status'] = 'complete'
        state['finished_at'] = datetime.now(timezone.utc).isoformat()
        for file in CACHE.glob('*.npz'):
            file.unlink()
        CACHE.rmdir()
        print('WINDOW_R2_COMPLETE', flush=True)
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(STATE / 'status.json', state)
        del pipeline
        clear_device_cache()


if __name__ == '__main__':
    main()
