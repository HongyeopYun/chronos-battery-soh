"""Whole-test-cell galleries of saved SOH L4 and historical direct RUL runs."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.c2_v4_experiments import load_cells
from src.c2_head_protocol import natural_key
from src.battery_zeroshot import sha256

OLD=ROOT/'results/c2-battery/v4-gpu-suite-v2'
LEARNING=ROOT/'results/c2-battery/rul-learning-curve-v1'
NAMES={'nasa_aging':'NASA','calce_cs2_1c':'CALCE','mit_batch1':'MIT Batch 1','mit_batch2':'MIT Batch 2',
       'mit_batch3':'MIT Batch 3','xjtu_2c':'XJTU 2C','xjtu_3c':'XJTU 3C'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=150,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def r2(y,p):
    den=np.sum((y-y.mean())**2)
    return 1-np.sum((y-p)**2)/den if len(y)>1 and den>0 else np.nan


def soh(out,cells,items):
    protocol=read(OLD/'protocol.json');ph=sha256(OLD/'protocol.json')
    count=0
    for group,folds in protocol['split_design']['soh'].items():
        ids=sorted([b for f in folds for b in f['test_ids']],key=natural_key)
        assert len(ids)==len(set(ids))
        for battery in ids:
            cell=cells[group,battery];source=OLD/'soh_predictions'/group/f'L4_{battery}.npz'
            with np.load(source) as z:
                assert str(z['protocol_sha256'])==ph
                t=z['cycle'];y=z['target'];zero=z['zero'][:,1];head=z['head'][:,1]
                np.testing.assert_array_equal(cell['values'][t-1],y)
                assert np.isfinite(cell['values'][t-1]).all()
                fig,ax=plt.subplots(figsize=(12,4.4))
                ax.plot(cell['cycles'],cell['values'],color='#17202a',lw=1.3,label='Eligible measured SOH')
                for pred,color,label in [(z['last_value'],'#9b9b9b','LastValue'),(zero,'#087dbc','Zero-shot'),(head,'#9446bb','Head-only')]:
                    line=np.full(len(cell['cycles']),np.nan);line[t-1]=pred
                    ax.plot(cell['cycles'],line,color=color,lw=1.,alpha=.9,label=label)
                ax.set(title=f'{NAMES[group]} | {battery} | SOH 4 -> 1 (rolling measured history)',
                    xlabel='Original cycle / observation index',ylabel='SOH = Q / initial reference')
                ax.grid(alpha=.2);ax.legend(loc='best',fontsize=9)
                fig.text(.08,.005,f'Full recorded axis; predictions only at evaluated t >= 65. Excluded points are gaps.  '
                    f'Zero-shot MAE={np.mean(abs(zero-y)):.5f}, R²={r2(y,zero):.3f}; '
                    f'Head MAE={np.mean(abs(head-y)):.5f}, R²={r2(y,head):.3f}',fontsize=8)
                fig.tight_layout(rect=(0,.045,1,1))
                rel=Path('soh_L4')/group/f'{battery}.png';save(fig,out/rel)
                items.append(dict(section='soh',dataset=group,battery=battery,path=rel.as_posix(),
                    description='입력 4개 → 현재 SOH 1개. 전체 실측 이력과 모든 저장 평가점.',
                    n_predictions=len(t),source=source.relative_to(ROOT).as_posix(),source_sha256=sha256(source)))
                count+=1
        print('SOH',group,len(ids),flush=True)
    assert count==145


def direct(out,items):
    source=OLD/'rul_direct_predictions.csv';frame=pd.read_csv(source);count=0
    for group in NAMES:
        d=frame[frame.dataset==group]
        for battery in sorted(d.battery_id.unique(),key=natural_key):
            z=d[d.battery_id==battery];ref=z[z.method=='head'].sort_values('origin_cycle')
            event=bool(ref.event_observed.iloc[0]);scope=ref.scope.iloc[0]
            fig,ax=plt.subplots(figsize=(12,4.4))
            if event:
                ax.plot(ref.origin_cycle,ref.true_rul,'k-',lw=1.8,label='True RUL (initial-reference 80%)')
            else:
                ax.plot(ref.origin_cycle,ref.observed_rul_lower_bound,'k--',lw=1.8,label='Observed follow-up lower bound; NOT true RUL')
            for method,color,label in [('head','#9446bb','Head-only'),('full','#d2691e','Full fine-tuning'),('lora','#009e73','LoRA')]:
                q=z[z.method==method].sort_values('origin_cycle')
                np.testing.assert_array_equal(q.origin_cycle,ref.origin_cycle)
                ax.plot(q.origin_cycle,q.predicted_rul,'o-',color=color,ms=2.7,lw=1.,label=label)
            ax.set(title=f'{NAMES[group]} | {battery} | Direct RUL, L=32 | {scope}',
                   xlabel='Prediction origin (current cycle / observation)',ylabel='Predicted remaining record positions')
            ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(loc='best',fontsize=9)
            note='HISTORICAL RUN: EOL = 80% of initial reference. NOT the corrected MIT nominal-80% run.'
            if not event:note+='  No exact event label under this old definition.'
            fig.text(.08,.005,note,fontsize=8,color='#9a4c00');fig.tight_layout(rect=(0,.045,1,1))
            rel=Path('rul_original_L32')/group/f'{battery}.png';save(fig,out/rel)
            items.append(dict(section='rul_original',dataset=group,battery=battery,path=rel.as_posix(),
                description='기존 입력 32개, Head·Full·LoRA. 초기용량 80% 기준 / '+('실제 RUL 표시' if event else '검열: 검정 점선은 추적기간 하한이며 정답 RUL이 아님'),
                event_observed=event,scope=scope,n_predictions=len(ref),source=source.relative_to(ROOT).as_posix(),source_sha256=sha256(source)))
            count+=1
        print('RUL original',group,d.battery_id.nunique(),flush=True)
    assert count==157


def learning(out,items):
    source=LEARNING/'predictions.csv';frame=pd.read_csv(source,dtype={'budget':str});count=0
    # Every source-cell count and label budget; show each actual seed as well
    # as the average so no fitted repetition is hidden by averaging.
    for group in NAMES:
        d=frame[frame.dataset==group]
        for n in sorted(d.n_train_cells.unique()):
            for battery in sorted(d.battery_id.unique(),key=natural_key):
                z=d[(d.battery_id==battery)&(d.n_train_cells==n)]
                if z.empty:continue
                fig,axes=plt.subplots(2,2,figsize=(13,8.5))
                ref=z[(z.method=='head')&(z.budget=='4')&(z.seed==42)].sort_values('origin_cycle')
                for ax,seed in zip(axes.ravel(),[42,43,44,None]):
                    ax.plot(ref.origin_cycle,ref.true_rul,'k-',lw=1.8,label='True RUL')
                    for budget,color in [('4','#d58821'),('16','#087dbc'),('all','#9446bb')]:
                        q=z[(z.method=='head')&(z.budget==budget)].pivot(index='origin_cycle',columns='seed',values='predicted_rul')
                        assert q.columns.tolist()==[42,43,44]
                        np.testing.assert_array_equal(q.index,ref.origin_cycle)
                        ax.plot(q.index,q.mean(axis=1) if seed is None else q[seed],color=color,lw=1.2,label=f'Head: {budget} labels/cell')
                        if seed is None:ax.fill_between(q.index,q.min(axis=1),q.max(axis=1),color=color,alpha=.10)
                    q=z[(z.method=='ridge')&(z.budget=='all')]
                    q=q.groupby('origin_cycle').predicted_rul.mean() if seed is None else q[q.seed==seed].set_index('origin_cycle').predicted_rul
                    ax.plot(q.index,q,'--',color='#009e73',lw=1.2,label='Ridge: all labels/cell')
                    ax.set(title=f'Seed {seed}' if seed is not None else '3-seed mean + min-max range',
                           xlabel='Prediction origin',ylabel='Remaining record positions')
                    ax.set_ylim(bottom=0);ax.grid(alpha=.2)
                axes[0,0].legend(loc='best',fontsize=8)
                fig.suptitle(f'{NAMES[group]} | {battery} | Direct RUL L=32 | {n} training cells')
                fig.text(.04,.005,'HISTORICAL initial-reference 80% EOL. Shaded = min-max across 3 seeds, NOT a confidence interval. Mean is not one fitted model.',fontsize=8,color='#9a4c00')
                fig.tight_layout(rect=(0,.035,1,.96))
                rel=Path('rul_learning_L32')/group/f'{battery}_train{n}.png';save(fig,out/rel)
                items.append(dict(section='rul_learning',dataset=group,battery=battery,path=rel.as_posix(),
                    description=f'입력 32개 / 학습 {n}셀 / 정답 4·16·전체 / seed 42·43·44 각각과 평균을 모두 표시.',
                    n_train_cells=int(n),n_predictions=len(ref),source=source.relative_to(ROOT).as_posix(),source_sha256=sha256(source)))
                count+=1
        if len(d):print('RUL learning',group,d.battery_id.nunique(),flush=True)
    assert count==92


def gallery(out,items):
    title={'soh':'SOH 입력 4개 · 전체 구간','rul_original':'기존 직접 RUL · Head / Full / LoRA','rul_learning':'직접 RUL · 학습 자료 확대'}
    cards=[]
    for item in items:
        cards.append(f'<article data-section="{item["section"]}" data-dataset="{item["dataset"]}" data-battery="{escape(item["battery"],quote=True)}"><h2>{NAMES[item["dataset"]]} · {escape(item["battery"])}</h2><p>{escape(item["description"])}</p><a href="{item["path"]}" target="_blank"><img loading="lazy" src="{item["path"]}" alt="{escape(item["description"],quote=True)}"></a><a class="download" href="{item["path"]}" download>PNG 다운로드</a></article>')
    sections=''.join(f'<option value="{key}">{value} ({sum(x["section"]==key for x in items)})</option>' for key,value in title.items() if any(x['section']==key for x in items))
    datasets=''.join(f'<option value="{key}">{value}</option>' for key,value in NAMES.items())
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SOH·RUL 전체 셀 그래프</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#16202c;font-family:Arial,"Malgun Gothic",sans-serif}header,main{max-width:1400px;margin:auto;padding:24px}h1{margin:0 0 14px}header p{line-height:1.65}.note{background:#fff2d8;padding:15px;border-left:4px solid #ca841a}.bar{position:sticky;top:0;z-index:2;background:#fff;padding:14px;border-bottom:1px solid #ddd;display:flex;gap:12px;flex-wrap:wrap;align-items:center}select,input{padding:10px;border:1px solid #acb4be;border-radius:5px;font-size:15px}article{background:white;margin:0 0 24px;padding:18px;border-radius:9px;box-shadow:0 2px 10px #0001}article h2{font-size:18px;margin:0}article p{color:#53616e;font-size:14px}img{width:100%;height:auto}.download{font-size:13px;color:#1c639c}article[hidden]{display:none}@media print{.bar,.download{display:none}article{break-inside:avoid;box-shadow:none}body{background:white}}</style>
    <header><h1>SOH·RUL 전체 셀 그래프</h1><p>대표 셀만 고르지 않고, 저장된 시험 예측이 있는 모든 셀을 포함했습니다. 그림을 클릭하면 원본 PNG가 열립니다. 데이터셋·셀 이름으로 골라 볼 수 있습니다.</p><p><b>SOH:</b> 입력 4개 → 한 점을 매 시점 다시 예측한 결과입니다. 한 번에 먼 미래 전체를 예측한 궤적이 아닙니다. 실측 전 구간을 표시하고, 예측은 저장된 공통 평가 구간(t≥65)에만 표시합니다. 제외된 관측은 선을 끊어 표시합니다.</p><p class="note"><b>RUL 조건 주의:</b> 아래 RUL은 기존 <b>입력 32개·초기 측정용량의 80%</b> 기준으로 저장된 결과입니다. 입력 4개 RUL 또는 새 MIT 정격용량 80%·공식 수명 라벨로 재학습한 결과가 아닙니다. 검열 셀의 검정 점선은 확인된 추적기간의 하한이며, 실제 남은 수명 정답이 아닙니다. CALCE의 단위는 기록 위치입니다.</p></header>
    <div class="bar"><label>그래프 <select id="section">SECTIONS</select></label><label>데이터셋 <select id="dataset"><option value="all">전체</option>DATASETS</select></label><label>셀 검색 <input id="search" placeholder="예: b2c10"></label><strong id="count"></strong></div><main>CARDS</main><script>
    const cards=[...document.querySelectorAll('article')],section=document.querySelector('#section'),dataset=document.querySelector('#dataset'),search=document.querySelector('#search');function filter(){let n=0;for(const c of cards){let show=c.dataset.section===section.value&&(dataset.value==='all'||c.dataset.dataset===dataset.value)&&c.dataset.battery.toLowerCase().includes(search.value.toLowerCase());c.hidden=!show;if(show)n++;}document.querySelector('#count').textContent=n+'개 그래프';}for(const el of [section,dataset,search])el.addEventListener('input',filter);filter();</script></html>'''
    (out/'index.html').write_text(html.replace('SECTIONS',sections).replace('DATASETS',datasets).replace('CARDS','\n'.join(cards)),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--soh-only',action='store_true');parser.add_argument('--reuse-soh',action='store_true');args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    cells,fingerprints=load_cells(ROOT/'data/processed/reviewed_v4/forecast_safe')
    assert fingerprints==read(OLD/'protocol.json')['data_artifact_sha256']
    items=[]
    if args.reuse_soh:
        old=read(out/'verification.json')
        for row in pd.read_csv(out/'plot_inventory.csv').to_dict('records'):
            if row['section']!='soh':continue
            assert sha256(out/row['path'])==old['files'][row['path']]
            assert sha256(ROOT/row['source'])==row['source_sha256']
            items.append(row)
        assert len(items)==145
    else:soh(out,cells,items)
    if not args.soh_only:direct(out,items);learning(out,items)
    gallery(out,items)
    pd.DataFrame(items).to_csv(out/'plot_inventory.csv',index=False)
    checks=dict(created_at=datetime.now(timezone.utc).isoformat(),counts={s:sum(i['section']==s for i in items) for s in sorted({i['section'] for i in items})},
        all_saved_test_cells_included=True,excluded_soh_values_are_nan=True,soh_input_length=4,rul_input_length=32,
        rul_definition='Historical initial-reference80%, not nominal80%',render_source_sha256=sha256(__file__),
        files={i['path']:sha256(out/i['path']) for i in items})
    (out/'verification.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    archive=out.parent/(out.name+'.zip')
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    print(json.dumps(dict(counts=checks['counts'],archive=str(archive),bytes=archive.stat().st_size)),flush=True)


if __name__=='__main__':main()
