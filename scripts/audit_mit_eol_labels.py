"""Read-only audit of author-provided MIT nominal-80% cycle-life labels."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
import pandas as pd
from src.c2_rul import eligible_mask

URL='https://raw.githubusercontent.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/master/Load%20Data.ipynb'
PAPER='https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=ROOT/'results/c2-battery/mit-eol-label-audit-v1'
    if (out/'verification.json').exists():raise ValueError('Completed audit is immutable')
    out.mkdir(parents=True,exist_ok=True)
    source=out/'upstream_load_data.ipynb'
    if not source.exists():source.write_bytes(urlopen(URL,timeout=30).read())
    nb=json.loads(source.read_text(encoding='utf-8'))
    assignments={};excluded={'batch1':set(),'batch3':set()}
    for cell in nb['cells']:
        if cell.get('cell_type')!='code':continue
        tree=ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
                if node.targets[0].id in ('batch1_keys','batch2_keys','add_len'):
                    assignments[node.targets[0].id]=ast.literal_eval(node.value)
            if isinstance(node,ast.Delete):
                for target in node.targets:
                    if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id in excluded:
                        excluded[target.value.id].add(ast.literal_eval(target.slice))
    links=dict(zip(assignments['batch1_keys'],zip(assignments['batch2_keys'],assignments['add_len'])))
    assert len(links)==5 and len(excluded['batch1'])==5 and len(excluded['batch3'])==6
    protected=[p for k in (1,2,3) for p in (ROOT/f'data/processed/reviewed_v4/forecast_safe/mit_batch{k}').glob('*') if p.is_file()]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    raw={};source_stats={}
    for k in (1,2,3):
        path=ROOT/f'data/raw/MIT/MIT_Batch_{k}.mat'
        source_stats[path.name]=dict(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
        with h5py.File(path,'r') as f:
            batch=f['batch']
            for i in range(len(batch['summary'])):
                s=f[batch['summary'][i,0]]
                raw[f'b{k}c{i}']=dict(life=float(np.asarray(f[batch['cycle_life'][i,0]]).ravel()[0]),
                    q=np.asarray(s['QDischarge']).ravel(),cycle=np.asarray(s['cycle']).ravel())
    rows=[];details=[]
    for k in (1,2,3):
        d=pd.read_csv(ROOT/f'data/processed/reviewed_v4/forecast_safe/mit_batch{k}/soh.csv')
        for bid,z in d.groupby('battery_id'):
            z=z.sort_values('cycle');a=raw[bid];life=a['life'];q=a['q'];cycles=a['cycle'];extra='';add=0
            if bid in links:
                extra,add=links[bid];n=len(q)
                assert len(raw[extra]['q'])==add
                q=np.r_[q,raw[extra]['q']];cycles=np.r_[cycles,raw[extra]['cycle']+n];life+=add
            np.testing.assert_array_equal(z.cycle,cycles)
            np.testing.assert_allclose(z.capacity_ah,q,rtol=1e-12,atol=1e-12)
            ok=z[eligible_mask(z)];old=ok[ok.soh<=.8];nom=ok[ok.capacity_ah<=.88]
            cross=cycles[np.isfinite(q)&(q>0)&(q<=.88)]
            valid=bid not in excluded['batch1']|excluded['batch3'] and np.isfinite(life)
            status=('author_censored_not_reached_80' if bid in excluded['batch1'] else
                    'author_noisy_channel_excluded' if bid in excluded['batch3'] else 'author_eol_label_usable')
            rows.append(dict(dataset=f'mit_batch{k}',battery_id=bid,physical_battery_id=bid,status=status,
                official_label_usable=valid,raw_cycle_life=a['life'],continuation=extra,continuation_add=add,
                official_eol_cycle=life if valid else np.nan,source_number_after_merge=life,
                nominal_capacity_ah=1.1,eol_capacity_ah=.88,last_stored_cycle=int(z.cycle.max()),
                last_eligible_cycle=int(ok.cycle.max()) if len(ok) else np.nan,min_positive_raw_capacity_ah=float(q[q>0].min()),
                raw_nominal80_crossing=cross[0] if len(cross) else np.nan,
                eligible_nominal80_crossing=nom.cycle.iloc[0] if len(nom) else np.nan,
                old_initial80_crossing=old.cycle.iloc[0] if len(old) else np.nan))
            if valid and len(cross) and (not len(nom) or nom.cycle.iloc[0]!=life):
                qrows=z[(z.cycle>=life-1)&(z.cycle<=life+1)].copy()
                details.extend(qrows[['battery_id','cycle','capacity_ah','quality_eligible','exclusion_reasons']].to_dict('records'))
    audit=pd.DataFrame(rows);valid=audit[audit.official_label_usable].copy()
    counts=valid.groupby('dataset').size().to_dict()
    assert counts=={'mit_batch1':41,'mit_batch2':43,'mit_batch3':40}
    b2=valid[valid.dataset=='mit_batch2']
    np.testing.assert_array_equal(b2.official_eol_cycle,b2.raw_nominal80_crossing)
    b13=valid[valid.dataset!='mit_batch2']
    assert b13.raw_nominal80_crossing.isna().all()
    assert (b13.official_eol_cycle-b13.last_stored_cycle==1).all()
    assert valid.physical_battery_id.is_unique and len(valid)==124
    assert not set(valid.battery_id)&set(assignments['batch2_keys'])
    audit.to_csv(out/'all_physical_cells_audit.csv',index=False)
    valid.to_csv(out/'usable_official_eol_labels.csv',index=False)
    pd.DataFrame(details).to_csv(out/'qc_crossing_differences.csv',index=False)
    summary=valid.groupby('dataset',as_index=False).agg(cells=('battery_id','size'),eol_min=('official_eol_cycle','min'),
        eol_median=('official_eol_cycle','median'),eol_max=('official_eol_cycle','max'))
    summary.to_csv(out/'summary.csv',index=False)
    assert all(sha(ROOT/p)==h for p,h in before.items())
    diff=b2.old_initial80_crossing-b2.official_eol_cycle
    lines=['# MIT EOL 원본 라벨 빠른 검토','',
        '기존 데이터·모델·성능표는 변경하지 않았다. 이 폴더는 신규 학습 결과가 아니라 원본 수명 라벨 감사 결과다.','',
        '## 결론','',
        '- 원 논문의 EOL은 정격용량 1.1Ah의 80%, 즉 0.88Ah이다. 기존 실험은 초기 측정용량 Q_ref의 80%였으므로 서로 다른 목표를 예측했다.',
        '- 제작자 처리 규칙을 적용한 수명 라벨은 Batch1 41셀, Batch2 43셀, Batch3 40셀, 총124개 물리 셀에서 사용할 수 있다.',
        '- Batch1·3의 유효81셀은 제공된 cycle_life가 마지막 저장 cycle보다 정확히1 크다. 저장된 용량 곡선에는 0.88Ah 교차가 없다. 관측 교차가 없다는 사실을 곧바로 원본 수명 라벨 부재로 해석하면 안 된다.',
        '- 마지막 cycle+1을 새 정답으로 추측하지 않는다. 별도 제공된 cycle_life를 읽고 제작자의 예외·연속 측정 보정을 적용한다. 미도달5셀도 숫자상 last+1을 가지므로 단순 덧셈은 잘못된 정답을 만든다.',
        f'- Batch2 43셀 모두 원본 cycle_life와 원시 QDischarge<=0.88 최초 cycle이 정확히 일치한다. 기존 초기용량80% 교차는 공식라벨보다 {int(diff.min())}~{int(diff.max())}cycle 늦다(중앙값 {diff.median():.0f}).',
        '- b2c40·b2c46은 현재 QC가 최초 교차 관측을 제외해 적격 관측으로 재계산하면 1cycle 늦어진다. 공식 라벨과 입력 관측 QC는 별도 관리해야 한다. 입력에서 제외된 점을 복원하거나 라벨을 임의로 바꾸지 않았다.',
        '- Batch1의 b1c8/10/12/13/22는 제작자가 80% 미도달로 지정했다. 수명 숫자가 들어 있어도 exact EOL로 사용하지 않는다. 원래 추적 기록을 유지하고 검열 셀로 따로 취급한다.',
        '- Batch3의 b3c2/23/32/37/42/43은 제작자 noisy channel 제외 규칙을 유지한다.',
        '- b1c0..4에 b2c7/8/9/15/16을 합치고 cycle_life에는 각각662/981/1060/208/482를 더한다. 보정 EOL은1852/2160/2237/1434/1709이며 독립적인 Batch2 셀로 중복 사용하지 않는다.','',
        '## 후속 실험에 적용할 정의','',
        '- MIT SOH_nominal=Q/1.1. EOL은 검증된 제작자 cycle_life, RUL(t)=EOL-t.',
        '- 원래 cycle 위치와 적격 입력 마스크를 유지하고 t<EOL만 평가한다. Batch2의 EOL 이후 기록은 RUL 학습·평가에서 제외한다.',
        '- 초기용량 정규화를 입력 표현으로 유지할 수도 있지만, 그때는 궤적의 임계값을 셀별0.88/Q_ref로 바꿔야 한다. 무조건0.8을 적용하면 같은 EOL이 아니다.',
        '- 별도 SOH 용량 교차가 저장되지 않은 Batch1·3에서도 공식 EOL 라벨로 직접 RUL과 궤적의 EOL 예측을 평가할 수 있다. 미래 SOH 곡선 자체는 관측된 범위까지만 평가한다.',
        '- 배치별 성능과 교차배치 일반화를 분리하고, 같은 물리 셀을 학습·검증·시험에 중복 배치하지 않는다.',
        '- 정격80% 기준으로 바꾸면 목표와 데이터 범위가 바뀐다. 기존 결과는 초기용량80% 실험으로 보존하고 새 실행으로 비교해야 한다.','',
        '## 근거','',f'- [원 논문]({PAPER})',f'- [제작자 Load Data]({URL})',
        '- 로컬 data/raw/MIT/MIT_Batch_1.mat, MIT_Batch_2.mat, MIT_Batch_3.mat의 cycle_life·summary를 직접 대조했다.',
        '- [전체135개 물리 셀 감사](all_physical_cells_audit.csv) · [사용 가능124개 라벨](usable_official_eol_labels.csv) · [QC 차이](qc_crossing_differences.csv) · [검증](verification.json)','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    verification=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),physical_cells=len(audit),usable_labels=counts,
        raw_nominal_crossing_matches_batch2=43,source_label_last_plus_one_batch1_batch3=81,
        label_qc_first_crossing_differences=len(details)//3,old_label_delay_min=int(diff.min()),old_label_delay_median=float(diff.median()),old_label_delay_max=int(diff.max()),
        processed_artifact_sha256_unchanged=before,raw_file_stats=source_stats,raw_hash_scope='Raw summaries and labels read; full multi-GB file hashes not recomputed in this quick audit.',
        upstream_url=URL,upstream_snapshot_sha256=sha(source),audit_source_sha256=sha(__file__),
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='verification.json'})
    (out/'verification.json').write_text(json.dumps(verification,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(summary.to_string(index=False));print('QC crossing differences:',pd.DataFrame(details).to_string(index=False))
    print('PASS: 124 labels; 135 physical records; old processed data unchanged')


if __name__=='__main__':main()
