"""Display NASA's first eligible forecast without changing phase scores."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
import pandas as pd
from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, write_json
from src.battery_model_zeroshot import load_model, predict_quantiles, phase_labels, PHASES
from src.battery_zeroshot_plots import forecast_figures
from src.soh_experiment import clear_device_cache


def main():
    directory = ROOT / 'results/c2-battery/full/nasa_aging'
    path = directory / 'c2-battery_run.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    config = ZeroShotConfig(**manifest['config'])
    records, _, data, _ = prepare_battery_windows(manifest['bundle'], config)
    metrics = pd.read_csv(directory / 'c2-battery_battery_metrics.csv')
    pipeline = load_model(config)
    plots = []
    try:
        for record in records:
            w = record['windows']
            median, q = predict_quantiles(pipeline, w['context'], config)
            expected = metrics[(metrics.battery_id == record['battery_id']) & (metrics.model == 'c2-battery') & (metrics.phase == 'all')].rmse.iloc[0]
            np.testing.assert_allclose(np.sqrt(np.square(w['target']-median).mean()), expected, rtol=2e-5, atol=5e-8)
            cycles = data[(data.battery_id == record['battery_id']) & data.is_observed].cycle.to_numpy()
            labels = phase_labels(w['forecast_start'], cycles)
            centers = cycles.min() + (np.arange(3)+.5)*(cycles.max()-cycles.min()+1)/3
            indices = [0]
            for phase, center in zip(PHASES[1:], centers[1:]):
                eligible = np.flatnonzero(labels == phase)
                indices.append(eligible[np.abs(w['forecast_start'][eligible]-center).argmin()])
            if w['forecast_start'][0] != 65:
                raise ValueError('Expected first NASA forecast at cycle 65')
            forecast = {**{k:v[indices].copy() for k,v in w.items()}, 'median':median[indices], 'q10':q[indices,:,0], 'q90':q[indices,:,-1]}
            plots.append({**record, 'forecast':forecast})
        figures = forecast_figures(directory, plots, config, show_window_scores=True, model_label='c2-battery',
                    filename_prefix='c2-battery_', window_labels=['First eligible', 'Middle', 'Late'],
                    representative_description='First eligible forecast starts at cycle 65; middle/late use observed-period thirds')
        manifest['figures'] = figures
        manifest['figure_selection'] = 'First panel shows the first eligible forecast (65-80), not the observed-period early phase. Middle/late keep their original representative windows; all scores unchanged.'
        write_json(path, manifest)
        report = directory / 'c2-battery_report.md'
        note = '\n**NASA 그림 보완:** 첫 패널은 cycle 65~80의 첫 유효 예측창이다. 관측 기간 첫 1/3의 초반 점수가 아니며, 초반 평가창 수 0과 기존 구간별 점수는 유지한다.\n'
        text = report.read_text(encoding='utf-8')
        if note not in text:
            report.write_text(text+note, encoding='utf-8')
        print('NASA first-window panels restored for all four batteries; scores unchanged.', flush=True)
    finally:
        del pipeline
        clear_device_cache()


if __name__ == '__main__':
    main()
