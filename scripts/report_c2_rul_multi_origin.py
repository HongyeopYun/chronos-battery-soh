"""Report the paired XPU multi-origin intervention without hiding non-hits."""
import numpy as np
import pandas as pd
from src.c2_rul import eligible_mask, score_trajectory, summarize_rul
from scripts.run_c2_rul_pilot import forecasts_to_frame
from scripts.report_c2_rul_context_sweep import trajectory_diagnostics

def build_report(out, plan, data, tasks, arrays):
    scores=[];summaries=[];forecasts=[];bands=[];diagnostics=[]
    methods=['zero-shot','single','multi']
    for method in methods:
        q=arrays[method]
        s=pd.DataFrame([dict(method=method,**score_trajectory(t,v[:,4])) for t,v in zip(tasks,q)])
        scores.append(s);summaries.append(dict(method=method,**summarize_rul(s)))
        f=forecasts_to_frame(tasks,q,data);f['method']=method;forecasts.append(f)
        for t,v in zip(tasks,q):
            rows,diag=trajectory_diagnostics(t,v,f[f.battery_id.eq(t['battery_id'])],64,method)
            bands.extend(rows);diagnostics.append(diag)
    scores=pd.concat(scores,ignore_index=True);summary=pd.DataFrame(summaries)
    scores.to_csv(out/'rul_predictions.csv',index=False);summary.to_csv(out/'summary.csv',index=False)
    pd.concat(forecasts,ignore_index=True).to_csv(out/'soh_predictions.csv.gz',index=False)
    pd.DataFrame(bands).to_csv(out/'horizon_diagnostics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'trajectory_diagnostics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    styles=[('zero-shot','Zero-shot','#2467ad'),('single','Head: source t64','#9d55aa'),('multi','Head: source t64/80/96','#138572')]
    for i,(ax,t) in enumerate(zip(axes.flat,tasks)):
        raw=data[data.battery_id.eq(t['battery_id'])].sort_values('cycle')
        ax.plot(raw.cycle,raw.soh.where(eligible_mask(raw)),color='black',lw=1.3,label='Eligible measurements')
        ax.axvspan(1,64,color='#999999',alpha=.12)
        for key,label,color in styles:
            ax.plot(np.arange(65,321),arrays[key][i,:,4],color=color,lw=1.4,label=label)
        ax.axhline(t['threshold_soh'],color='#555555',ls='--',lw=.8,label='EOL threshold')
        ax.axvline(64,color='#888888',ls=':',lw=.8)
        if t['true_eol_cycle'] is not None:ax.axvline(t['true_eol_cycle'],color='#c17d23',ls=':',lw=1)
        ax.set(title=t['battery_id'],xlabel='Discharge record ordinal',ylabel='SOH');ax.grid(alpha=.18)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
    fig.suptitle('More source origins: same test t64, L64, H256; fresh XPU controls',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.87));fig.savefig(out/'trajectories.png',dpi=170);plt.close(fig)
    fmt=lambda v:'>256' if pd.isna(v) else str(int(v))
    lines=['# NASA RUL: 학습 시작 시점 확장 비교','',
        '입력 64, 예측 256은 유지했다. 학습 시작 시점만 t64 하나에서 t64·80·96 세 개로 늘렸다. 두 head 조건과 zero-shot 모두 같은 노트북 XPU에서 새로 실행했다.','',
        '- 각 fold는 학습 2셀, 검증 1셀, 평가 1셀로 분리한다. 검증 및 평가는 t64로 고정한다.',
        '- 각 셀의 창 수가 같아 셀별 가중치는 동일하다. 창마다 유효 정답 평균 손실을 계산한 뒤 창별로 평균한다.',
        '- 학습 창은 2→6개지만 고유 미래 SOH 정답 수는 증가하지 않는다. 중복 정답 노출과 입력 열화 구간의 다양성이 늘어난다.',
        '- LR 1e-6/1e-5 각 100 steps, seed42, 검증 SOH loss와 분위수 역전 기준으로 step0 포함 선택. 8개 head를 모두 고정한 후 평가한다.',
        '- EOL은 최초 유효 용량 1.4Ah 이하. B0007은 우측 검열. 미도달은 >256이며 오류 집계에서 숨기지 않는다.',
        '- 훈련용 미래 정답은 기존과 같이 관측된 post-EOL 용량도 포함한다. 시험 SOH 오차는 EOL 또는 관측 종료까지만 계산한다.',
        '- 평활화·보간·단조화·미래 실측 재입력 없음. 제외 관측은 그림에 표시하지 않는다.','',
        '| 셀 | 실제 RUL | Zero-shot | 기존 t64 head | t64/80/96 head |',
        '|---|---:|---:|---:|---:|']
    for t in tasks:
        truth=str(t['true_rul']) if t['true_rul'] is not None else f">{t['observed_rul_lower_bound']}"
        vals=[fmt(scores[scores.battery_id.eq(t['battery_id'])&scores.method.eq(m)].iloc[0].predicted_rul) for m in methods]
        lines.append('| '+t['battery_id']+' | '+truth+' | '+' | '.join(vals)+' |')
    lines+=['','실제 EOL이 관측된 3셀 기준:','','| 조건 | EOL 미도달 | 유한 예측 조건부 MAE | 전체 MAE 하한 |','|---|---:|---:|---:|']
    for _,r in summary.iterrows():
        lines.append(f"| {r.method} | {int(r.n_no_crossing_event_predictions)}/3 | {r.conditional_mae_cycles:.2f} | {r.mae_lower_bound_all_event_origins:.2f} |")
    lines+=['','오차 단위는 방전 기록 수. 조건부 MAE는 EOL 도달 예측 사례만의 값이며, 전체 MAE 하한은 정확한 MAE가 아니다.',
        '','단일 seed, NASA 4셀(사건 3셀)의 탐색 연구다. 기존 시험 결과를 본 뒤 설계한 후속 실험이므로 독립 외부 검증으로 해석하지 않는다.',
        '','![예측 경로](trajectories.png)','','[정답 수와 중복 노출](label_budget.csv) · [RUL 원점수](rul_predictions.csv) · [검증 결과](verification.json)','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
