"""Produce a scientific report from audited, saved NASA predictions only."""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = ROOT/'results/c2-battery/rul-nasa-fewshot-report-v1'
ROBUST = ROOT/'results/c2-battery/rul-nasa-fewshot-robustness-v1'
LABELS = {'historical_L32':'기존 Head · L32', 'tuned_L32':'개선 Head · L32',
          'tuned_context':'개선 Head · L8/16/32 선택', 'full_warm_L32':'개선 Head → 전체 가중치 학습'}
COLORS = {'historical_L32':'#999da8', 'tuned_L32':'#6b36d8', 'tuned_context':'#e39423', 'full_warm_L32':'#008d85'}
ORDER = list(LABELS)


def metrics(frame, extra=()):
    rows = []
    for scope, subset in [('original_stride16', frame[frame.original_stride16]), ('dense_stride1', frame)]:
        for key, a in subset.groupby(['method', 'seed', *extra, 'battery_id']):
            y, p = a.true_rul.to_numpy(), a.predicted_rul.to_numpy()
            d = p-y
            den = np.sum((y-y.mean())**2)
            row = dict(zip(['method', 'seed', *extra, 'battery_id'], key))
            rows.append(dict(**row, scope=scope, n=len(a), mae=float(abs(d).mean()), rmse=float(np.sqrt(np.mean(d*d))),
                             r2=float(1-np.sum(d*d)/den) if den > 0 else np.nan, bias=float(d.mean())))
    return pd.DataFrame(rows)


def table(frame, labels=None):
    a = frame.copy()
    if labels:
        a = a.rename(columns=labels)
    return a.to_html(index=False, border=0, classes='data', float_format=lambda x:f'{x:.2f}', escape=True)


def save(fig, name):
    fig.savefig(OUT/name, dpi=175, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def trajectories(frame, methods, filename, title):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for ax, b in zip(axes.flat, sorted(frame.battery_id.unique())):
        a = frame[frame.battery_id == b]
        eol = int((a.origin + a.true_rul).iloc[0])
        x = np.arange(32, eol+1)
        ax.plot(x, eol-x, color='#161b2a', lw=2, label='실제 RUL')
        for method in methods:
            p = a[a.method == method].pivot(index='origin', columns='seed', values='predicted_rul').reindex(x[:-1])
            avg, lo, hi = p.mean(axis=1), p.min(axis=1), p.max(axis=1)
            # Reindexing leaves missing/excluded observations as NaN gaps.
            ax.plot(p.index, avg, color=COLORS[method], lw=1.8, ls='--' if method=='full_warm_L32' else '-', label=LABELS[method])
            if method == 'tuned_L32':
                ax.fill_between(p.index, lo, hi, color=COLORS[method], alpha=.12)
            marked = ((p.index.to_numpy()-32)%16)==0
            ax.scatter(p.index[marked], avg[marked], s=20, color=COLORS[method], zorder=4)
        ax.set(title=f'{b}  |  EOL = {eol}', xlabel='예측 시점 (cycle)', ylabel='남은 수명 (cycles)')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=len(handles), frameon=False)
    fig.suptitle(title+'\n선: 3 seed 평균 · 점: 기존 stride16 평가 시점 · 음영: 개선 Head의 seed 최솟값~최댓값', fontsize=14)
    save(fig, filename)


def main():
    assert json.loads((OUT/'audit.json').read_text(encoding='utf-8'))['status'] == 'passed'
    plt.rcParams.update({'font.family':'Malgun Gothic', 'axes.unicode_minus':False, 'font.size':11})
    frame = pd.read_csv(OUT/'predictions.csv')
    cm = metrics(frame)
    cm.to_csv(OUT/'cell_metrics.csv', index=False)
    sm = cm.groupby(['scope','method','seed'])[['mae','rmse','r2','bias']].mean().reset_index()
    sm.to_csv(OUT/'seed_metrics.csv', index=False)
    summary = sm.groupby(['scope','method'])[['mae','rmse','r2','bias']].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    primary = summary[summary.scope=='original_stride16'].set_index('method').reindex(ORDER)
    reduction = 100*(1-primary.loc['tuned_L32','mae_mean']/primary.loc['historical_L32','mae_mean'])
    display = pd.DataFrame({'방법':[LABELS[m] for m in ORDER],
                            'MAE (cycle)':[f'{r.mae_mean:.2f} ± {r.mae_std:.2f}' for _,r in primary.iterrows()],
                            'RMSE':primary.rmse_mean.to_numpy(), '셀별 R² 평균':primary.r2_mean.to_numpy()})
    per_cell = cm[cm.scope=='original_stride16'].groupby(['battery_id','method'])[['mae','r2','n']].mean().reset_index()
    cell_table = per_cell.pivot(index='battery_id',columns='method',values='mae').reindex(columns=ORDER)
    cell_table['감소율 (%)'] = 100*(1-cell_table.tuned_L32/cell_table.historical_L32)
    cell_table = cell_table.reset_index().rename(columns=LABELS|{'battery_id':'셀'})
    stage = frame.copy()
    stage['stage'] = np.select([stage.life_fraction<=.5, stage.life_fraction<=.8], ['early','middle'], default='late')
    stage['absolute_error'] = abs(stage.predicted_rul-stage.true_rul)
    stage_cells = stage.groupby(['method','seed','battery_id','stage']).agg(mae=('absolute_error','mean'),n=('absolute_error','size')).reset_index()
    stage_cells.to_csv(OUT/'stage_cell_metrics.csv',index=False)
    stage_scores = stage_cells.groupby(['method','seed','stage']).mae.mean().groupby(['method','stage']).mean().unstack()
    stage_counts = stage.groupby('stage').agg(cells=('battery_id','nunique'))
    counts = stage[stage.method=='tuned_L32'].groupby(['battery_id','stage']).origin.nunique().unstack(fill_value=0)
    counts.to_csv(OUT/'stage_origin_counts.csv')
    trajectories(frame, ['historical_L32','tuned_L32','full_warm_L32'], 'nasa_before_after.png', 'NASA 직접 RUL few-shot: 같은 입력 32개, 같은 학습 정답 32개')
    trajectories(frame, ['tuned_L32','tuned_context'], 'context_comparison.png', '입력 길이 선택의 추가 효과')
    fig, axes = plt.subplots(1,2,figsize=(13,4.6),constrained_layout=True)
    for i, method in enumerate(ORDER):
        r=primary.loc[method]
        axes[0].bar(i,r.mae_mean,yerr=r.mae_std,color=COLORS[method],capsize=4,width=.65)
        axes[0].text(i,r.mae_mean+r.mae_std+.5,f'{r.mae_mean:.2f}',ha='center',fontsize=11)
    axes[0].set_xticks(range(4),['기존 Head','개선 Head\nL32','길이 선택\nL8/16/32','전체 가중치\n추가 학습'])
    axes[0].set(ylabel='MAE (cycles, 낮을수록 좋음)',title='기존과 동일한 stride16 평가')
    axes[0].set_ylim(0,32)
    for i,method in enumerate(['historical_L32','tuned_L32']):
        ys=stage_scores.loc[method,['early','middle','late']].to_numpy()
        axes[1].bar(np.arange(3)+(i-.5)*.32,ys,width=.32,color=COLORS[method],label=LABELS[method])
    axes[1].set_xticks(range(3),['초기 ≤50%\n3셀','중기 50–80%\n4셀','말기 >80%\n4셀'])
    axes[1].set(ylabel='MAE (cycles)',title='수명 구간별 · stride1 보조 평가')
    axes[1].legend(frameon=False)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'metrics.png')
    robust_html = ''
    if (ROBUST/'independent_audit.json').exists():
        assert json.loads((ROBUST/'independent_audit.json').read_text())['status']=='passed'
        r = pd.read_csv(ROBUST/'predictions.csv')
        rc = metrics(r, extra=('validation_id',))
        rc.to_csv(OUT/'robustness_cell_metrics.csv',index=False)
        # Equal averaging of all 3 partitions and 3 seeds within each test cell, then 4 cells.
        rr = rc.groupby(['scope','method','battery_id'])[['mae','rmse','r2']].mean().reset_index()
        rs = rr.groupby(['scope','method'])[['mae','rmse','r2']].mean().reset_index()
        rs.to_csv(OUT/'robustness_summary.csv',index=False)
        names={'none_huber1':'기존 스케일 + Huber','none_mae':'기존 스케일 + MAE',
               'metadata_huber1':'SOH 특징 표준화 + Huber','metadata_mae':'SOH 특징 표준화 + MAE'}
        rd=rs[rs.scope=='original_stride16'].drop(columns='scope').copy()
        rd['method']=rd.method.map(names)
        rp=rr[rr.scope=='original_stride16'].pivot(index='battery_id',columns='method',values='mae').rename(columns=names).reset_index()
        fig, ax=plt.subplots(figsize=(9,4.7),constrained_layout=True)
        for i,m in enumerate(names):
            values=rc[(rc.scope=='original_stride16')&(rc.method==m)]
            avg=values.groupby('battery_id').mae.mean().mean()
            ax.bar(i,avg,color=['#9fa5b2','#c3b8e6','#7460ac','#6b36d8'][i],width=.65)
            ax.text(i,avg+.5,f'{avg:.2f}',ha='center')
        ax.set_xticks(range(4),['기존 스케일\nHuber','기존 스케일\nMAE','SOH 특징 표준화\nHuber','SOH 특징 표준화\nMAE'])
        ax.set(ylabel='MAE (cycles)',title='모든 셀 분할 × 3 seed: 스케일과 손실의 효과 분리')
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        save(fig,'robustness.png')
        robust_html=f'''<section><h2>분할을 바꿔도 개선되는가?</h2><p>시험 셀마다 나머지 3셀을 모두 한 번씩 검증 셀로 사용했다. 각 경우 나머지 2셀에서 16개씩만 학습했다. 입력은 L32로 고정하고, 스케일 2종 × 손실 2종을 각각 보고했다. 학습률과 체크포인트는 각 방법·seed에서 검증 MAE로 선택했다.</p>
        <img src="robustness.png" alt="모든 셀 분할에서 스케일과 손실의 효과"/>{table(rd,{'method':'방법','mae':'MAE','rmse':'RMSE','r2':'셀별 R² 평균'})}{table(rp,{'battery_id':'시험 셀'})}
        <p><b>손실만 바꾸는 효과는 작고, 기존 SOH 특징의 표준화 효과가 컸다.</b> 표준화 후에는 Huber와 MAE의 전체 평균이 비슷하다. 따라서 앞의 검증에서 MAE가 선택됐다는 사실을 “MAE 손실 자체가 핵심”으로 해석하면 안 된다.</p>
        <p class="note">4개의 실제 시험 셀에 대한 반복 분할이다. 12개의 독립 시험 셀이 생긴 것은 아니다. 앞의 결과에서 출발한 후속 진단이므로 독립 외부 검증으로 해석하지 않는다.</p></section>'''
    audit=json.loads((OUT/'audit.json').read_text())
    title='NASA 직접 RUL few-shot 개선'
    page=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
    <style>body{{margin:0;background:#f4f5f9;color:#18233a;font:16px/1.65 'Malgun Gothic',sans-serif}}main{{max-width:1180px;margin:40px auto;padding:0 24px}}h1{{font-size:34px;margin:8px 0}}h2{{font-size:23px;margin-top:0}}section{{background:white;border:1px solid #e4e6ee;border-radius:16px;padding:28px;margin:22px 0}}.tag{{color:#6942bd;font-weight:700}}.hero{{background:#172848;color:white}}.hero strong{{font-size:30px;color:#e0d2ff}}img{{display:block;width:100%;height:auto;margin:20px 0}}.data{{border-collapse:collapse;width:100%;font-size:15px;margin:16px 0}}th,td{{border-bottom:1px solid #e4e6ee;padding:12px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f2f1f8}}.note{{color:#586078;font-size:14px}}a{{color:#6534c6}}code{{background:#eee;padding:2px 5px}}details{{margin-top:16px}}li{{margin:8px 0}}@media(max-width:760px){{main{{padding:0 10px}}section{{padding:16px;overflow-x:auto}}.data{{font-size:12px}}th,td{{padding:8px}}}}</style></head>
    <body><main><p class="tag">Chronos-2 · NASA · 직접 RUL · Few-shot</p><h1>{title}</h1>
    <section class="hero"><p>기존과 같은 셀 분할·입력·정답 수·평가 시점</p><strong>MAE 25.12 → 10.41 cycles · {reduction:.1f}% 감소</strong><p>Head 입력의 기존 SOH 특징 스케일과 학습 설정을 조정했다. 모델 층·차원·입력 변수는 그대로 유지했다.</p></section>
    <section><h2>무엇을 비교했나</h2><ul><li>입력: 최근 SOH 32개 → 해당 시점의 남은 수명 1개. 정답은 EOL − 현재 cycle.</li>
    <li>학습: 별도 셀 2개 × RUL 정답 16개 = 32개. 검증 셀 1개의 전체 유효 정답으로 설정·체크포인트 선택, 시험 셀 1개는 가중치 학습에 사용하지 않았다.</li>
    <li>NASA EOL: 기존과 동일한 초기 기준 SOH 80% 첫 도달. B0005=101, B0006=61, B0007=124, B0018=75. 정격 용량 80%로 바꾼 실험은 아니다.</li>
    <li>평가: seed 42·43·44 각각 셀별 지표를 구하고, 4셀을 동일 가중 평균한 뒤 seed 평균을 계산했다. ±는 seed 간 표준편차다.</li></ul>
    <p class="note">32개는 학습 정답 수다. 별도 검증 셀의 정답은 추가로 사용한다. 과거 셀의 수명 전반에서 학습 예시를 고른 방식이며, 새 시험 셀의 미래 EOL을 학습에 넣지 않는다.</p></section>
    <section><h2>기존 평가 시점에서의 결과</h2>{table(display)}<img src="metrics.png" alt="MAE 및 수명 구간별 성능"/>
    <p>주 비교는 기존과 같은 stride16이다. 4셀의 서로 다른 시점 총 16개를 seed별로 평가한다. 입력 길이까지 선택한 결과는 고정 L32보다 추가 이득이 없었다. 전체 가중치 학습은 검증에서 12개 중 10개가 학습 전 Head로 되돌아갔다.</p>
    <p class="note">R²는 한 셀의 여러 시점 예측을 모아 계산한 뒤 셀별 동일 가중 평균이다. B0006은 기존 평가 시점이 2개뿐이고 R²가 음수여서 전체 평균을 크게 낮춘다. 점을 전부 합쳐 계산한 pooled R²가 아니다.</p></section>
    <section><h2>셀별로 얼마나 달라졌나</h2>{table(cell_table)}<img src="nasa_before_after.png" alt="NASA 4셀 RUL 개선 전후"/>
    <p>4셀 모두 기존 Head보다 MAE가 낮아졌다. 다만 B0006은 여전히 과대 추정이 남아 있다. 선은 더 촘촘한 stride1 저장 예측을 보여 주며, 점이 주 비교에 쓰인 기존 시점이다. 음영은 3 seed 범위이며 신뢰구간이 아니다.</p></section>
    {robust_html}
    <section><h2>입력 길이를 바꾸는 효과</h2><img src="context_comparison.png" alt="고정 L32와 검증으로 선택한 context 비교"/><p>검증으로 선택된 입력 길이는 B0005·B0006은 32, B0007은 16, B0018은 8이었다. B0007에서는 좋아졌지만 B0018에서는 오히려 나빠져, 고정 L32의 전체 성능을 넘지 못했다.</p></section>
    <section><h2>해석과 남은 한계</h2><p>이번 결과는 내부 구조 변경 없이도 Head 학습 설정에서 상당한 개선 여지가 있었다는 증거다. 기존 768개 모델 특징은 그대로 두고, 이미 사용하던 평균 SOH·SOH 변동폭·마지막 SOH의 3개 특징만 학습 셀 기준으로 표준화했다. 기존 분할의 탐색에서는 이 표준화와 MAE 손실이 함께 선택됐다. 원인을 분리한 반복 분할 비교에서는 표준화의 효과가 크고, 표준화 후 Huber와 MAE의 성능은 비슷했다. 전체 가중치를 학습하면 더 좋아진다는 증거는 이번 비교에서 얻지 못했다.</p>
    <p>NASA 4셀은 수명이 서로 다르고, 특정 셀 하나를 검증에 쓰는 선택도 불안정할 수 있다. 기존 시험 결과를 이미 본 뒤 진행한 탐색 실험이므로 미사용 외부 데이터에서의 성능 보장으로 해석하면 안 된다. 특히 B0006의 성능은 아직 충분하지 않다.</p>
    <p class="note">수명 초기·중기·말기 분석은 t/EOL 기준의 사후 평가다. 실제 예측 입력에는 EOL과 수명 비율을 넣지 않는다. 시작 시점이 t=32이므로 EOL=61인 B0006은 초기 ≤50% 구간에 들어가지 않는다.</p>
    <details><summary>검증 및 원본 파일</summary><p>선택된 Head 36개를 노트북에서 다시 불러 예측을 재현했다. 최대 차이 {audit['max_head_prediction_difference']:.2g} cycle. 스케일러를 학습 정답 32개에서 독립적으로 다시 계산했고, 기존 예측과의 일치·셀 분리·선택 규칙·소스 및 데이터 해시를 확인했다. 전체 가중치 모델은 집 GPU에서 재로딩 검증했고 큰 가중치 파일은 집 데스크톱에 보관했다.</p>
    <p><a href="summary.csv">평균 지표</a> · <a href="cell_metrics.csv">셀·seed별 지표</a> · <a href="predictions.csv">전체 예측</a> · <a href="audit.json">검증 기록</a> · <a href="stage_origin_counts.csv">구간별 관측 수</a> · <a href="reproduction.json">재현 정보</a></p></details></section>
    </main></body></html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print(summary.to_string(index=False))
    print('REPORT',OUT/'index.html')


if __name__ == '__main__':
    main()
