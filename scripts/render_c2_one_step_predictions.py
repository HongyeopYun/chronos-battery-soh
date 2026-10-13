"""Render saved one-step predictions with only eligible SOH observations.

Presentation-only entry point: keep the completed experiments' frozen source,
data, protocol, predictions and scores intact.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.soh_data import EXCLUDED_FLAGS, load_bundle, sha256, write_json

BUNDLES = ROOT / 'data/processed/reviewed_native_v3'
DEFAULT_OUT = ROOT / 'results/c2-battery/one-step-16-reviewed-v3'
NOTE = '시각화는 정제 기준을 통과한 SOH만 표시한다. 제외된 관측점은 점·선 모두 숨기며, 해당 위치는 빈 구간으로 남긴다. 원래 cycle 위치와 평가 전의 유효한 측정 구간은 유지한다.'


def make_figure(data, predictions, battery, length, group):
    rows = data[data.battery_id.eq(battery)].set_index('cycle')
    index = np.arange(rows.index.min(), rows.index.max() + 1)
    rows = rows.reindex(index)
    eligible = (rows.quality_eligible.eq(True) & rows.is_observed.eq(True)
                & np.isfinite(rows.soh) & rows.capacity_ah.gt(0)
                & rows.capacity_kind.isin(['full_discharge', 'reference_test']))
    flagged = rows.quality_flags.fillna('').map(
        lambda value: bool(set(str(value).split(';')) & EXCLUDED_FLAGS))
    eligible &= ~flagged
    truth = rows.soh.where(eligible)
    selected = predictions[predictions.battery_id.eq(battery)
                           & predictions.context_length.eq(length)]
    if not eligible.reindex(selected.target_cycle, fill_value=False).all():
        raise ValueError('Saved predictions contain an ineligible target')

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(index, truth, color='black', lw=1.3, label='Eligible measured SOH')
    for method, color in [('LastValue', '#888888'), ('zero-shot', '#2375ad'),
                          ('head-only', '#934daf')]:
        values = selected[selected.method.eq(method)].set_index('target_cycle').reindex(index)
        ax.plot(index, values.prediction, lw=.9, color=color, label=method)
    ax.set(title=f'L={length} -> 1', xlabel='Original cycle / observation index', ylabel='SOH')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.suptitle(f'{group} | {battery}\nEligible measurements; predictions at evaluated times')
    fig.tight_layout(rect=(0, 0, 1, .94))

    # Verify the rendered values, including gaps and valid pre-evaluation history.
    rendered = ax.lines[0].get_ydata()
    assert np.isnan(rendered[~eligible.to_numpy()]).all()
    np.testing.assert_array_equal(rendered[eligible], rows.loc[eligible, 'soh'])
    assert not ax.collections  # No excluded-point scatter layer.
    counts = dict(battery_id=battery, plotted_observations=int(eligible.sum()),
                  hidden_observations=int((rows.is_observed.eq(True) & ~eligible).sum()))
    return fig, counts


def render(out):
    out = Path(out).resolve()
    plan = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
    if len(plan['contexts']) != 1:
        raise ValueError('Expected a single-context one-step run')
    # Capture all numerical artifacts and frozen experiment metadata.
    protected = [p for p in out.rglob('*') if p.is_file()
                 and (p.name.endswith(('.csv', '.csv.gz', '.json')))
                 and p.name != 'prediction_rendering.json']
    before = {p: sha256(p) for p in protected}
    figures = {}
    for group, spec in plan['groups'].items():
        bundle = BUNDLES / group
        assert sha256(bundle / 'soh.csv') == spec['data_sha256'], group
        data, _, _ = load_bundle(bundle)
        prediction_path = out / group / 'predictions.csv.gz'
        predictions = pd.read_csv(prediction_path)
        battery = spec['folds'][0]['test_ids'][0]
        fig, counts = make_figure(data, predictions, battery, plan['contexts'][0], group)
        path = out / group / 'representative.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures[group] = dict(**counts, figure_sha256=sha256(path),
                              predictions_sha256=sha256(prediction_path))
        print(f'{group}: {counts}', flush=True)
    assert all(sha256(p) == digest for p, digest in before.items())
    # Current report receives an explicit display-policy note; frozen older
    # reports can keep their text when this renderer is used for other runs.
    report = out / 'report.md'
    if out == DEFAULT_OUT.resolve() and report.exists():
        content = report.read_text(encoding='utf-8')
        if NOTE not in content:
            first_group = next(iter(plan['groups']))
            marker = f'![{first_group}]'
            content = content.replace(marker, NOTE + '\n\n' + marker, 1)
            report.write_text(content, encoding='utf-8')
    write_json(out / 'prediction_rendering.json', dict(
        rendered_at=datetime.now(timezone.utc).isoformat(),
        renderer_sha256=sha256(Path(__file__)),
        policy='Eligible observations only; NaN gaps; original cycle positions; no excluded markers.',
        numerical_artifacts_unchanged=True, figures=figures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    render(parser.parse_args().out)
