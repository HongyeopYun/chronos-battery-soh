"""Presentation-only refresh; the completed model protocol remains unchanged."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.battery_zeroshot import sha256,write_json
from scripts.run_c2_context32 import OUT,PRIOR
from scripts.run_c2_one_step_context import GROUPS,read,now


def main():
    assert read(OUT/'status.json')['status']=='complete'
    before={str(folder.relative_to(ROOT))+'/'+g:sha256(folder/g/'predictions.csv.gz')
            for folder in (PRIOR,OUT) for g in GROUPS}
    summary=pd.read_csv(OUT/'lastvalue_comparison.csv')
    a=summary[summary.context_length.eq(64)].set_index(['dataset','method','phase'])
    b=summary[summary.context_length.eq(32)].set_index(['dataset','method','phase']).reindex(a.index)
    comparison=a[['n_batteries','n_targets']].copy()
    comparison['mse_skill_lastvalue_64']=a.mse_skill_lastvalue
    comparison['mse_skill_lastvalue_32']=b.mse_skill_lastvalue
    comparison['delta_mse_skill_lastvalue']=b.mse_skill_lastvalue-a.mse_skill_lastvalue
    comparison.to_csv(OUT/'context32_vs64_lastvalue_change.csv')

    fig,axes=plt.subplots(1,2,figsize=(12,5.7),sharey=True)
    all_values=100*summary[summary.method.ne('LastValue')&summary.phase.eq('all')].mse_skill_lastvalue
    limits=(min(-15,float(all_values.min())-12),max(88,float(all_values.max())+12))
    for ax,method,title in zip(axes,['zero-shot','head-only'],['C2 zero-shot','C2 head-only']):
        for length,offset,color in [(64,-.17,'#90a4ae'),(32,.17,'#2467ad')]:
            d=summary[summary.context_length.eq(length)&summary.method.eq(method)&summary.phase.eq('all')].set_index('dataset').loc[list(GROUPS)]
            values=100*d.mse_skill_lastvalue
            bars=ax.barh(np.arange(len(GROUPS))+offset,values,height=.29,color=color,label=f'{length} past positions')
            for bar,v in zip(bars,values):
                label=f'{v:+.2f}%' if abs(v)<1 else f'{v:+.1f}%'
                ax.text(v+(1 if v>=0 else -1),bar.get_y()+bar.get_height()/2,label,
                        ha='left' if v>=0 else 'right',va='center',fontsize=8)
        ax.axvline(0,color='#555555',lw=.8)
        ax.set_xlim(*limits)
        ax.set(title=title,xlabel='MSE reduction vs LastValue (%) — higher is better')
        ax.set_yticks(np.arange(len(GROUPS)),GROUPS)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.legend(loc='upper right',fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('Past SOH history: 64 vs 32 positions -> one target',fontsize=14)
    fig.text(.5,.015,'Same 97,868 targets; compute the MSE ratio per battery record, then average. Negative = worse than LastValue.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.95));fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)

    original=OUT/'.runtime/generated_report.md'
    if not original.exists():original.write_bytes((OUT/'report.md').read_bytes())
    report=original.read_text(encoding='utf-8')
    start=report.index('## 입력 축소와 head 학습의 효과')
    end=report.index('단일 seed의 점추정이다.',start)
    lines=['## 입력 축소와 head 학습의 효과','',
           '입력 축소 효과는 **같은 LastValue 기준**의 MSE 감소율이 64개에서 32개로 얼마나 변했는지 비교한다. 첫 두 열은 위 성능표의 `32 − 64`이며 단위는 감소율의 percentage point(%p)다. 양수가 개선이다. 마지막 열은 32개 입력에서 head-only의 zero-shot 대비 MSE 감소율이다.','',
           '| 데이터셋 | 64→32 zero-shot 변화(%p) | 64→32 head-only 변화(%p) | 32 head vs zero-shot MSE 감소율 |',
           '|---|---:|---:|---:|']
    for g in GROUPS:
        z=100*comparison.loc[(g,'zero-shot','all'),'delta_mse_skill_lastvalue']
        h=100*comparison.loc[(g,'head-only','all'),'delta_mse_skill_lastvalue']
        head=100*b.loc[(g,'head-only','all'),'mse_skill_zero']
        lines.append(f'| {g} | {z:+.2f} | {h:+.2f} | {head:+.2f}% |')
    lines+=['',
        '32개 head-only는 LastValue 기준으로 MIT batch2·3, XJTU 3C에서 64개 조건보다 높은 감소율을 보였다. NASA·CALCE·MIT batch1·XJTU 2C는 낮아졌다. MIT batch1은 step 0이 선택돼 head 학습의 추가 이득이 없다. CALCE는 검증으로 선택한 head가 테스트에서는 zero-shot보다 악화됐다.','',
        '같은 LastValue 분모를 쓰는 위 비교를 주 비교로 삼는다. 참고 파일 `context32_vs64_summary.csv`는 각 모델의 64개 입력 MSE를 분모로 한 별도 상대비교다. 배터리별 분모가 달라 두 평균의 순위가 달라질 수 있으므로 두 수치를 같은 지표로 해석하지 않는다.','',
        '32개 head-only의 테스트 분위수 역전 비율은 NASA 7.69%, CALCE 17.37%, XJTU 2C 9.12%다. CALCE에는 q10>q90인 예측도 있어 해당 기록의 구간 폭·포함률을 NaN으로 남겼다. 검증에서의 guard가 테스트의 분위수 순서를 보장하지는 않는다. 위 표는 중앙값 예측의 점오차다.','']
    report=report[:start]+'\n'.join(lines)+'\n'+report[end:]
    report=report.replace('[입력 32 vs 64 비교](context32_vs64_summary.csv)',
                          '[같은 LastValue 기준 입력 축소 효과](context32_vs64_lastvalue_change.csv)')
    report=report.replace('재현: `.venv\\Scripts\\python.exe scripts/run_c2_context32.py --stage all`',
        '실험 재현: `.venv\\Scripts\\python.exe scripts/run_c2_context32.py --stage all`. 최종 표시 재현: `.venv\\Scripts\\python.exe scripts/render_c2_context32_results.py`.')
    (OUT/'report.md').write_text(report,encoding='utf-8')
    after={str(folder.relative_to(ROOT))+'/'+g:sha256(folder/g/'predictions.csv.gz')
           for folder in (PRIOR,OUT) for g in GROUPS}
    assert before==after
    write_json(OUT/'rendering.json',dict(rendered_at=now(),renderer_sha256=sha256(__file__),
        unchanged_prediction_sha256=after,original_protocol_sha256=sha256(OUT/'protocol.json'),
        note='Only plot layout, interpretation, and a fixed-LastValue comparison table changed. Training, selection and predictions were unchanged.'))


if __name__=='__main__':main()
