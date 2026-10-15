"""Explain saved indirect-RUL trajectories; no training or new inference."""
from pathlib import Path
import json
import sys
from html import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.c2_v4_experiments import load_cells
from src.battery_zeroshot import sha256

SOURCE = ROOT / 'results/c2-battery/v4-gpu-suite-v2'
OUT = ROOT / 'results/c2-battery/indirect-rul-representatives-v1'
NAMES = {'mit_batch2': 'MIT Batch 2', 'nasa_aging': 'NASA',
         'xjtu_2c': 'XJTU 2C', 'xjtu_3c': 'XJTU 3C', 'calce_cs2_1c': 'CALCE'}
GROUPS = [
    ('good', '잘한 사례', '#16805c', '종료가 가까운 시점에서 SOH 하락과 80% 도달 시점을 함께 맞힘'),
    ('partial', '애매한 사례', '#b37812', 'SOH 곡선은 가깝지만, 80% 아래로 내려가지 않아 RUL 산출 실패'),
    ('poor', '못한 사례', '#c04a53', '초기 관측만으로는 장기 하락을 놓치고 80% 도달도 예측하지 못함'),
]
CHOICES = [
    ('good', 'mit_batch2', 'b2c34', 464, '실제 50 cycle 남은 시점에서 51 cycle로 예측. 아래 초기 시점과 같은 셀이다.'),
    ('good', 'xjtu_2c', 'batch1_2C_battery-7', 352, '실제 31 cycle에 비해 5 cycle 늦게 예측. 수명 말기 하락을 따라간다.'),
    ('partial', 'xjtu_3c', 'batch2_3C_battery-7', 256, 'SOH RMSE는 작지만 예측 중앙값이 80% 위에 남아 EOL을 놓친다.'),
    ('partial', 'calce_cs2_1c', 'CS2_38', 528, '실측의 마지막 하락을 충분히 따라가지 못해 80% 도달을 예측하지 못한다.'),
    ('poor', 'nasa_aging', 'B0007', 32, '실측은 빠르게 감소하지만 예측은 높은 SOH에 머문다.'),
    ('poor', 'mit_batch2', 'b2c34', 32, '첫 번째 성공 사례와 같은 셀. 이 초기 시점에서는 이후의 하락과 EOL을 놓친다.'),
]


def metrics_and_data(cells, ph):
    frame = pd.read_csv(SOURCE / 'rul_zero_matched_predictions.csv')
    rows, saved, fingerprints = [], {}, {}
    exact = frame[frame.event_observed]
    for (g, b), block in exact.groupby(['dataset', 'battery_id']):
        cell = cells[g, b]
        path = SOURCE / 'rul_trajectories' / g / f'{b}.npz'
        fingerprints[str(path.relative_to(ROOT))] = sha256(path)
        with np.load(path) as z:
            assert str(z['protocol_sha256']) == ph
            for r in block.itertuples():
                t = int(r.origin_cycle)
                i = np.flatnonzero(z['cycle'] == t).item()
                q = z['quantiles'][i].copy()
                assert q.shape == (3, 1024) and np.isfinite(q).all()
                cross = np.flatnonzero(q[1] <= .8)
                pred_rul = float(cross[0] + 1) if len(cross) else np.nan
                assert bool(len(cross)) == bool(r.predicted_crossing)
                assert not len(cross) or pred_rul == r.predicted_rul
                assert int(cell['eol']) == r.true_eol_cycle
                assert cell['eol'] - t == r.true_rul
                stop = min(int(cell['eol']), t + q.shape[1])
                observed = cell['values'][t:stop]
                finite = np.isfinite(observed)
                assert finite.any()
                diff = q[1, :len(observed)][finite] - observed[finite]
                rows.append(dict(dataset=g, battery_id=b, origin=t, true_rul=float(r.true_rul),
                    pred_rul=pred_rul, n=int(finite.sum()), rmse_pp=float(np.sqrt(np.mean(diff**2))*100),
                    mae_pp=float(np.mean(abs(diff))*100), rul_error=abs(pred_rul-r.true_rul),
                    relative_rul_error=abs(pred_rul-r.true_rul)/r.true_rul,
                    origin_fraction=t/cell['eol']))
                if any((g, b, t) == choice[1:4] for choice in CHOICES):
                    saved[g, b, t] = q
    return pd.DataFrame(rows), saved, fingerprints


def title(r):
    battery = r['battery_id'].split('_')[-1] if r['dataset'].startswith('xjtu') else r['battery_id']
    return f"{NAMES[r['dataset']]} · {battery}  |  예측 시작 t={r['origin']}"


def score_text(r):
    pred = '>1,024 (미도달)' if pd.isna(r['pred_rul']) else f"{r['pred_rul']:.0f}"
    unit = '기록' if r['dataset'] == 'calce_cs2_1c' else 'cycle'
    return f"RUL 실제 {r['true_rul']:.0f} → 예측 {pred} {unit}  ·  SOH RMSE {r['rmse_pp']:.2f}%p"


def draw(ax, r, cell, q, full=False):
    t = r['origin']
    predx = np.arange(1, 1025)
    right = 1024 if full else int(max(r['true_rul'], 0 if pd.isna(r['pred_rul']) else r['pred_rul']) * 1.23 + 8)
    x = cell['cycles'] - t
    valid_range = (x >= -31) & (x <= right)
    actual = cell['values'][valid_range] * 100
    # NaNs are preserved: excluded measurements are never drawn or bridged.
    ax.axvspan(-31, 0, color='#eef2f6', zorder=0)
    ax.plot(x[valid_range], actual, color='#222b36', lw=1.8, zorder=4)
    ordered = (q[0] <= q[1]) & (q[1] <= q[2])
    ax.fill_between(predx, q[0]*100, q[2]*100, where=ordered, color='#2384bd', alpha=.13)
    ax.plot(predx, q[1]*100, color='#087fb9', lw=2, zorder=3)
    ax.axhline(80, color='#d78121', lw=1.3, ls='--')
    ax.axvline(0, color='#808c9a', lw=1, ls=':')
    ax.axvline(r['true_rul'], color='#27313e', lw=1.15, ls='--')
    if pd.notna(r['pred_rul']):
        ax.axvline(r['pred_rul'], color='#087fb9', lw=1.2, ls=':')
        ax.scatter([r['pred_rul']], [q[1, int(r['pred_rul'])-1]*100], color='#087fb9', s=28, zorder=5)
    ax.set_xlim(-31, right)
    displayed = q[1, :right]*100
    lo = min(79.3, np.nanmin(actual), np.min(displayed))
    hi = max(80.7, np.nanmax(actual), np.max(displayed))
    pad = max(.65, (hi-lo)*.09)
    # The zoom view keeps its y-axis focused on observed/median paths.
    ax.set_ylim(lo-pad, hi+pad)
    ax.set_ylabel('SOH (%)')
    ax.set_xlabel('예측 시작 이후 기록 수' if r['dataset'] == 'calce_cs2_1c' else '예측 시작 이후 cycle')
    ax.grid(alpha=.16)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(('전체 예측 범위 1,024개' if full else title(r)) + '\n' + score_text(r), fontsize=10.5, loc='left', pad=12)


def legend(fig, y):
    handles = [Line2D([0], [0], color='#222b36', lw=2, label='적격 실측'),
               Line2D([0], [0], color='#087fb9', lw=2, label='Zero-shot 예측 중앙값'),
               Line2D([0], [0], color='#d78121', ls='--', label='SOH 80%'),
               Line2D([0], [0], color='#27313e', ls='--', label='실제 EOL'),
               Line2D([0], [0], color='#087fb9', ls=':', label='예측 EOL')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, y), ncol=5, frameon=False, fontsize=10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'font.size': 10, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    cells, hashes = load_cells(ROOT / 'data/processed/reviewed_v4/forecast_safe')
    protocol = json.loads((SOURCE / 'protocol.json').read_text(encoding='utf-8'))
    assert hashes == protocol['data_artifact_sha256']
    ph = sha256(SOURCE / 'protocol.json')
    metrics, saved, fingerprints = metrics_and_data(cells, ph)
    metrics.to_csv(OUT / 'candidate_metrics.csv', index=False)
    selected = []
    for category, g, b, t, description in CHOICES:
        found = metrics[(metrics.dataset == g) & (metrics.battery_id == b) & (metrics.origin == t)]
        assert len(found) == 1
        r = found.iloc[0].to_dict()
        r.update(category=category, description=description, origin=int(t), n=int(r['n']))
        assert r['true_rul'] >= 20 and r['n'] >= 15
        if category == 'good':
            assert r['relative_rul_error'] <= .2 and r['rmse_pp'] <= 2
        elif category == 'partial':
            assert pd.isna(r['pred_rul']) and r['rmse_pp'] < 1
        else:
            assert pd.isna(r['pred_rul']) and r['rmse_pp'] > 5
        selected.append(r)
    pd.DataFrame(selected).to_csv(OUT / 'selected_cases.csv', index=False, encoding='utf-8-sig')
    overview, axes = plt.subplots(3, 2, figsize=(15, 15.5))
    overview.subplots_adjust(left=.065, right=.975, top=.80, bottom=.085, hspace=.87, wspace=.22)
    overview.suptitle('간접 RUL: SOH 궤적 대표 6사례', x=.065, y=.982, ha='left', fontsize=22, weight='bold')
    overview.text(.065, .950, '동일 조건: 입력 32개 → 미래 1,024개 · Zero-shot · 초기 기준용량의 80% EOL', fontsize=12)
    overview.text(.065, .926, '성공 사례는 수명 말기 예측이다. 아래 그림은 실제 EOL 주변 확대이며, 전체 예측은 INDEX에서 확인.', fontsize=10.5, color='#586270')
    legend(overview, .909)
    cards = []
    for i, r in enumerate(selected):
        q = saved[r['dataset'], r['battery_id'], r['origin']]
        cell = cells[r['dataset'], r['battery_id']]
        draw(axes.flat[i], r, cell, q)
        category, label, color, subtitle = GROUPS[i//2]
        pos = axes.flat[i].get_position()
        if i % 2 == 0:
            overview.text(.065, pos.y1+.061, f'{i//2+1}. {label}', color=color, weight='bold', fontsize=16)
        filename = f"{i+1:02d}_{r['dataset']}_{r['battery_id']}_t{r['origin']}.png"
        fig, aa = plt.subplots(1, 2, figsize=(15, 5.4))
        fig.subplots_adjust(left=.06, right=.98, top=.7, bottom=.15, wspace=.22)
        fig.suptitle(f'{label} | {title(r)}', x=.06, y=.98, ha='left', fontsize=17, weight='bold', color=color)
        fig.text(.06, .905, r['description'], fontsize=11)
        legend(fig, .86)
        draw(aa[0], r, cell, q)
        draw(aa[1], r, cell, q, full=True)
        fig.text(.06, .035, '음수 x: 입력 이력 · 음영: 순서가 정상인 q10–q90 · 제외 관측은 숨김 · RMSE: 예측 시작 다음부터 실제 EOL까지 적격 실측', fontsize=9, color='#586270')
        fig.savefig(OUT / filename, dpi=150)
        plt.close(fig)
        cards.append(dict(**r, filename=filename, label=label, color=color))
    overview.text(.065, .019, '사후 선정한 설명용 사례이며 전체 성능의 평균이 아님. SOH 오차 단위: %p. CALCE 수명 단위: 기록 수.', fontsize=10, color='#586270')
    overview.savefig(OUT / 'overview.png', dpi=160)
    plt.close(overview)
    noncross = int(metrics.pred_rul.isna().sum())
    content = []
    for category, label, color, subtitle in GROUPS:
        content.append(f'<section id="{category}"><h2 style="color:{color}">{label}</h2><p>{subtitle}</p>')
        for r in cards:
            if r['category'] != category:
                continue
            content.append(f'<article><h3>{escape(title(r))}</h3><p class="metric">{escape(score_text(r))}</p><p>{escape(r["description"])}</p><a href="{r["filename"]}" target="_blank"><img loading="lazy" src="{r["filename"]}" alt="{escape(title(r))}"></a><a href="{r["filename"]}" download>그림 다운로드</a></article>')
        content.append('</section>')
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>간접 RUL · 대표 궤적 6사례</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#202c3c;font:16px/1.65 "Malgun Gothic",sans-serif}header,main{max-width:1320px;margin:auto;padding:32px}header{padding-bottom:16px}h1{font-size:34px;line-height:1.3;margin:0 0 15px}h2{margin:32px 0 8px}h3{margin:0;font-size:21px}p{margin:9px 0}a{color:#126cac}nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:22px}nav a{padding:8px 15px;border-radius:9px;background:white;text-decoration:none;border:1px solid #d8e1ed}article,.note{background:white;border:1px solid #dce4ef;border-radius:13px;padding:24px;margin:18px 0}img{width:100%;height:auto;display:block}.metric{font-weight:bold;color:#17477b}.muted{color:#5b6777;font-size:14px}.note{border-left:5px solid #244f82}section{scroll-margin-top:20px}</style><header><h1>간접 RUL: 대표 SOH 궤적 6사례</h1><p>입력 32개 → 미래 SOH 1,024개 → 예측 중앙값이 80%에 처음 도달하는 시점으로 RUL 계산.</p><p>저장된 v4 Zero-shot 결과에서 골랐으며, 추가 학습·추론 없이 그래프만 구성했습니다.</p>'''
    html += f'<div class="note"><b>전체 결과와 함께 보기</b><p>정확한 EOL이 있는 시험 54셀의 1,461개 예측 시점 중 {noncross:,}개({noncross/len(metrics)*100:.1f}%)는 1,024개 예측 범위 안에서 80%에 도달하지 못했습니다. 아래 성공 사례는 그중 수명 말기에 해당합니다.</p><p><b>SOH 80%는 이 기존 실험의 Q / 초기 기준용량 기준입니다.</b> 이후 MIT 정격용량 80%로 다시 만든 직접 RUL 실험과 기준이 다릅니다.</p><p class="muted">미도달 &gt;1,024는 RUL=1,024라는 뜻이 아닙니다. 정답이 없는 검열 셀은 성공·실패 순위에서 제외했습니다.</p></div>'
    html += '<nav><a href="#good">잘한 사례</a><a href="#partial">애매한 사례</a><a href="#poor">못한 사례</a><a href="overview.png" target="_blank">6개 한눈에 보기</a><a href="selected_cases.csv" download>선정 사례 수치</a></nav></header><main>'
    html += ''.join(content)
    html += '<div class="note"><b>선정·표시 기준</b><p>잘한 사례: RUL 상대오차 20% 이하, SOH RMSE 2%p 이하. 애매한 사례: SOH RMSE 1%p 미만이지만 80% 미도달. 못한 사례: SOH RMSE 5%p 초과이며 80% 미도달. 모두 실제 RUL 20 이상·적격 미래 관측 15개 이상인 후보에서 설명을 위해 사후 선정했습니다.</p><p>SOH RMSE는 예측 시작 다음부터 실제 EOL까지의 적격 실측과 예측 중앙값으로 계산합니다. 각 사례의 평가 구간 길이는 다릅니다. 첫 그림은 EOL 주변 확대, 두 번째는 전체 1,024개 예측입니다. 음수 x는 입력 이력입니다. 제외된 측정점은 표시하지 않고 결측 구간을 유지합니다.</p><p>같은 MIT b2c34도 t=32에서는 실패하고 t=464에서는 성공했습니다. 긴 미래 예측 성능과 종료 직전 예측 성능을 구분해서 보아야 합니다. CALCE의 x축과 RUL은 물리 cycle이 보장되지 않은 기록 수입니다.</p><p><a href="candidate_metrics.csv">전체 EOL 관측 후보 수치</a> · <a href="verification.json">검증 기록</a></p></div></main></html>'
    (OUT / 'index.html').write_text(html, encoding='utf-8')
    assert len(saved) == len(CHOICES) == 6 and len(metrics) == 1461
    assert sha256(SOURCE / 'protocol.json') == ph
    for path, expected in fingerprints.items():
        assert sha256(ROOT / path) == expected
    verification = dict(status='passed', new_training=False, new_inference=False,
        source_protocol_sha256=ph, data_artifact_sha256=hashes,
        source_prediction_sha256=sha256(SOURCE / 'rul_zero_matched_predictions.csv'),
        trajectory_files_sha256=fingerprints, exact_origin_count=len(metrics),
        exact_cell_count=metrics.groupby(['dataset', 'battery_id']).ngroups,
        noncrossing_count=noncross, representative_count=6,
        threshold='0.8 of initial reference capacity; original v4 protocol',
        checks=['data hashes match original protocol', 'NPZ protocol hashes match',
                'recomputed crossings match saved RUL CSV', 'EOL labels match eligible observations',
                'all six cases satisfy declared descriptive selection criteria',
                'excluded observations remain NaN', 'source trajectories unchanged'],
        output_sha256={p.name:sha256(p) for p in OUT.iterdir() if p.is_file() and p.name!='verification.json'})
    (OUT / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(OUT), selected=6, origins=len(metrics), noncrossing=noncross), ensure_ascii=False))


if __name__ == '__main__':
    main()
