"""Report matched 64/32/16 one-step results without changing earlier runs."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.c2_context_comparison import METHODS, paired_metrics, summarize_paired


def comparison_tables(folders, groups):
    records, summaries = [], []
    reference = {}
    for length, folder in folders:
        for group in groups:
            frame = pd.read_csv(folder / group / 'predictions.csv.gz')
            frame = frame[frame.method.isin(METHODS)]
            last = frame[frame.method.eq('LastValue')].sort_values(['battery_id', 'target_cycle'])
            keys = ['battery_id', 'target_cycle', 'target', 'prediction', 'phase']
            if group in reference:
                pd.testing.assert_frame_equal(last[keys].reset_index(drop=True), reference[group])
            else:
                reference[group] = last[keys].reset_index(drop=True)
            records.append(paired_metrics(frame))
        summary = pd.read_csv(folder / 'summary.csv')
        summaries.append(summary[summary.method.isin(METHODS)])
    battery = pd.concat(records, ignore_index=True)
    scores = pd.concat(summaries, ignore_index=True)
    paired = summarize_paired(battery)
    keys = ['dataset', 'method', 'phase']
    new = paired[paired.context_length.eq(16)].set_index(keys)
    deltas = []
    for length in (64, 32):
        old = paired[paired.context_length.eq(length)].set_index(keys)
        if set(old.index) != set(new.index):
            raise ValueError('Missing reference records in context comparison')
        old = old.reindex(new.index)
        table = new[['n_batteries', 'n_targets']].copy()
        table['reference_context_length'] = length
        table['lastvalue_skill_change_pp'] = 100 * (new.mse_skill_lastvalue - old.mse_skill_lastvalue)
        deltas.append(table.reset_index())
    return battery, paired, scores, pd.concat(deltas, ignore_index=True)


def write_report(out, plan, summary):
    import matplotlib.pyplot as plt

    root = Path(out).parents[2]
    folders = [(64, root/'results/c2-battery/one-step-64-reviewed-v3'),
               (32, root/'results/c2-battery/one-step-32-reviewed-v3'), (16, out)]
    battery, paired, scores, deltas = comparison_tables(folders, plan['groups'])
    battery.to_csv(out/'lastvalue_battery_metrics.csv', index=False)
    paired.to_csv(out/'lastvalue_comparison.csv', index=False)
    scores.to_csv(out/'context_comparison_scores.csv', index=False)
    deltas.to_csv(out/'context16_lastvalue_change.csv', index=False)
    lines = ['# C2 현재 SOH 예측: 64→1 · 32→1 · 16→1', '',
        '과거 SOH 16개 기록 위치로 현재 SOH 한 점을 예측한다. 실제 과거 SOH로 입력을 갱신하는 rolling one-step 평가다.', '',
        '- 기존 reviewed_native_v3 정제·NaN 마스크, 배터리 분할, few-shot 학습/검증 정답과 97,868개 테스트 시점을 유지한다.',
        '- 비교 조건: LastValue, C2 zero-shot, C2 head-only. Linear 제외.',
        '- Head-only: backbone 고정, output_patch_embedding만 학습. Fold당 학습 8개·검증 4개 정답, LR 1e-6/1e-5, 각각 100 steps, seed 42. 검증으로 head 선택 후 테스트한다.',
        '- R²: 같은 배터리 기록의 여러 테스트 시점 한 점 예측을 모아서 계산한 뒤 기록별 동일 가중 평균. 음수 유지, 정의 불가능한 값의 수는 CSV에 표시한다.',
        '- RMSE/MAE는 SOH 비율 단위, 기록별 동일 가중 평균이다. LastValue MSE 감소율도 각 기록의 비율을 먼저 구한 뒤 평균한다.',
        '- 기존 64/32 결과는 XPU, 새 결과는 runtime.json에 기록된 장치에서 계산된다. 하드웨어·패키지 차이에 의한 수치 및 head 선택 차이는 입력 길이 효과와 완전히 분리되지 않는다.',
        '- CALCE의 파일 사이 미확인 물리 cycle, MIT 계측 불일치, 결측 입력 관련 기존 제한을 유지한다.', '',
        '## 16→1 결과', '',
        '| 데이터셋 | 조건 | Test R² | RMSE | MAE | LastValue 대비 MSE 감소율 |',
        '|---|---|---:|---:|---:|---:|']
    current = summary[summary.phase.eq('all')].set_index(['dataset', 'method'])
    skills = paired[paired.phase.eq('all')].set_index(['dataset', 'context_length', 'method'])
    for group in plan['groups']:
        for method in METHODS:
            row = current.loc[(group, method)]
            skill = skills.loc[(group, 16, method), 'mse_skill_lastvalue']
            lines.append(f'| {group} | {method} | {row.temporal_r2:.6f} | {row.rmse:.6f} | {row.mae:.6f} | {100*skill:+.2f}% |')
    lines += ['', '## 입력 길이별 head-only 비교', '',
        '| 데이터셋 | 64 R² | 32 R² | 16 R² | 64 LV 감소율 | 32 LV 감소율 | 16 LV 감소율 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    indexed = scores[scores.phase.eq('all')].set_index(['dataset', 'context_length', 'method'])
    for group in plan['groups']:
        r2 = [indexed.loc[(group, length, 'head-only'), 'temporal_r2'] for length in (64, 32, 16)]
        skill = [skills.loc[(group, length, 'head-only'), 'mse_skill_lastvalue'] for length in (64, 32, 16)]
        lines.append('| '+group+' | '+' | '.join([*(f'{v:.6f}' for v in r2), *(f'{100*v:+.2f}%' for v in skill)])+' |')
    lines += ['', '입력 길이 간 차이는 동일한 LastValue 기준 MSE 감소율의 차이(percentage points)로 표시한다.', '']
    lines += ['![입력 길이 비교](comparison.png)', '',
        '분위수 역전은 정렬·보정하지 않는다. 점 예측 지표와 별개로 summary.csv의 quantile_crossing_fraction 및 invalid_interval_fraction을 확인한다.', '',
        '[전체 지표](context_comparison_scores.csv) · [LastValue 비교](lastvalue_comparison.csv) · [길이별 차이](context16_lastvalue_change.csv) · [검증](verification.json)', '']
    for group in plan['groups']:
        lines += [f'![{group}]({group}/representative.png)', '']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    groups = list(plan['groups'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True, sharex=True)
    for ax, method in zip(axes, ('zero-shot', 'head-only')):
        for length, offset, color in ((64, -.24, '#90a4ae'), (32, 0, '#2467ad'), (16, .24, '#d67c23')):
            values = [100*skills.loc[(g, length, method), 'mse_skill_lastvalue'] for g in groups]
            ax.barh(np.arange(len(groups))+offset, values, height=.22, color=color, label=f'{length} inputs')
        ax.axvline(0, color='#555', lw=.8)
        ax.set(title=method, xlabel='MSE reduction vs LastValue (%)')
        ax.set_yticks(np.arange(len(groups)), groups)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.legend(*axes[1].get_legend_handles_labels(), loc='upper right', ncol=3)
    fig.suptitle('Matched one-step SOH prediction: 64 / 32 / 16 inputs', x=.39)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(out/'comparison.png', dpi=170)
    plt.close(fig)
