"""NASA L=64 indirect RUL pilot; fixed H=256, native Chronos-2 zero-shot."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
import numpy as np
import pandas as pd
from src.c2_rul import eligible_mask, make_tasks, score_trajectory, summarize_rul
from src.soh_data import load_bundle

OUT = ROOT / 'results/c2-battery/rul-nasa-l64-h256-zero-v1'
BUNDLE = ROOT / 'data/processed/reviewed_native_v3/nasa_aging'
REVISION = '29ec3766d36d6f73f0696f85560a422f50e8498c'
SOURCES = ['scripts/run_c2_rul_pilot.py', 'src/c2_rul.py', 'src/soh_data.py']
QUANTILES = [float(q/10) for q in range(1,10)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def prepare():
    data, metadata, _ = load_bundle(BUNDLE)
    audit, tasks = make_tasks(data, metadata)
    protocol = dict(schema_version=1, scope='NASA four-cell zero-shot pilot, not a full RUL benchmark',
                    model_id='amazon/chronos-2', revision=REVISION, context_length=64,
                    prediction_length=256, origin_stride=16, primary_origin_cycle=64, seed=42,
                    eol_capacity_ah=1.4,
                    eol_definition='First eligible measured capacity <= 1.4 Ah; event is absorbing. No interpolation. Three consecutive values are a label sensitivity diagnostic only.',
                    eol_source='https://data.nasa.gov/dataset/li-ion-battery-aging-datasets',
                    unit='NASA discharge record ordinal, not elapsed hours or all original MAT operations',
                    input='Latest 64 original positions through current t inclusive; excluded observations are NaN. Never supply actual measurements after t within a forecast.',
                    origins='Fixed grid t=64,80,96,...; require current observation eligible and no EOL observed at or before t. Future EOL and lifetime fractions do not choose origins.',
                    forecasting='One independent univariate group per origin, cross_learning=False, float32, q50 point trajectory. Fixed 256 steps, no smoothing, clipping, monotonicity projection, quantile sorting or added extrapolation.',
                    uncertainty='Saved quantiles are pointwise SOH quantiles; threshold crossings are NOT asserted to be calibrated RUL quantiles.',
                    censoring='No observed EOL: report RUL > last eligible cycle - t; no exact RUL error. No forecast crossing within H: report predicted RUL > H, not H. Show non-hit counts alongside conditional finite-case errors and all-case error lower bounds.',
                    scoring='Primary comparison at common t=64. Secondary fixed-grid per-cell MAE/RMSE/bias, then equal-cell means. No pooled temporal R2 across origins. Only descriptive cross-cell R2 at t=64 when all observed-event cells have finite predictions.',
                    soh_scoring='Prefix errors of the SAME H=256 forecast, at available eligible observations through EOL (or censoring); prefixes are not separately generated short-horizon forecasts.',
                    adaptation='None. Zero train/validation labels; one-step tuned heads are not reused. This run makes no claim about head-only RUL performance.',
                    future_conditions='Continuation of each recorded aging-test regime; future operational covariates are unavailable.',
                    source_sha256={p:sha(ROOT/p) for p in SOURCES},
                    data_sha256={name:sha(BUNDLE/name) for name in ['soh.csv','batteries.csv','manifest.json']},
                    n_tasks=len(tasks), tasks=[{k:v for k,v in task.items() if k != 'context'} for task in tasks])
    OUT.mkdir(parents=True, exist_ok=True)
    locked = OUT/'protocol.json'
    if locked.exists() and json.loads(locked.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Protocol or code changed: preserve this run and use a new output directory')
    write_json(locked, protocol)
    audit.to_csv(OUT/'eol_audit.csv',index=False)
    inputs = np.stack([t['context'] for t in tasks])
    np.savez_compressed(OUT/'inputs.npz', context=inputs,
                        origin=np.array([t['origin_cycle'] for t in tasks]),
                        battery=np.array([t['battery_id'] for t in tasks]))
    print('PREPARED:',len(audit),'cells;',len(tasks),'origins;',int(audit.event_observed.sum()),'observed EOL cells',flush=True)
    return protocol, data, audit, tasks, inputs


def forecasts_to_frame(tasks, values, data):
    blocks = []
    for task, q in zip(tasks, values):
        battery, origin = task['battery_id'], task['origin_cycle']
        raw = data[data.battery_id.eq(battery)].copy().set_index('cycle')
        raw['eligible'] = eligible_mask(raw)
        future = np.arange(origin+1, origin+257)
        matched = raw.reindex(future)
        keep = matched.eligible.eq(True).to_numpy()
        target = np.where(keep, matched.soh, np.nan)
        through_eol = np.ones(256, dtype=bool) if task['true_eol_cycle'] is None else future <= task['true_eol_cycle']
        block = pd.DataFrame(dict(battery_id=battery, origin_cycle=origin,
                                  horizon_step=np.arange(1,257), target_cycle=future,
                                  target_soh=target, score_eligible=keep & through_eol))
        for i,qlevel in enumerate(QUANTILES):
            block[f'q{round(qlevel*100):02d}'] = q[:,i]
        blocks.append(block)
    return pd.concat(blocks,ignore_index=True)


def save_reports(protocol, data, audit, tasks, q):
    rows = pd.DataFrame([score_trajectory(task, value[:,4]) for task,value in zip(tasks,q)])
    rows.to_csv(OUT/'rul_predictions.csv',index=False)
    trajectories = forecasts_to_frame(tasks,q,data)
    trajectories.to_csv(OUT/'soh_predictions.csv.gz',index=False,compression='gzip')
    cell = pd.DataFrame([dict(battery_id=b,**summarize_rul(part)) for b,part in rows.groupby('battery_id')])
    cell.to_csv(OUT/'battery_metrics.csv',index=False)
    primary = rows[rows.origin_cycle.eq(64)]
    summary = dict(primary_t64=summarize_rul(primary), secondary_all_origins=summarize_rul(rows))
    event_cell = cell[cell.n_event_origins.gt(0)]
    summary['secondary_equal_cell'] = {key:float(event_cell[key].mean()) if event_cell[key].notna().any() else None
                                     for key in ['conditional_mae_cycles','conditional_rmse_cycles','conditional_bias_cycles','mae_lower_bound_all_event_origins']}
    observed = primary[primary.event_observed]
    summary['primary_t64']['descriptive_cross_cell_r2'] = None
    if len(observed)>1 and observed.predicted_crossing.all():
        y=observed.true_rul.to_numpy(dtype=float); p=observed.predicted_rul.to_numpy(dtype=float)
        sst=float(np.square(y-y.mean()).sum())
        if sst>0:summary['primary_t64']['descriptive_cross_cell_r2']=float(1-np.square(p-y).sum()/sst)
    summary['pointwise_quantile_crossing_fraction'] = float((np.diff(q,axis=-1)<0).any(axis=-1).mean())
    summary['pointwise_q10_q90_inversion_fraction'] = float((q[...,0]>q[...,-1]).mean())
    write_json(OUT/'summary.json',summary)
    prefix=[]
    for (battery,origin), block in trajectories.groupby(['battery_id','origin_cycle']):
        for h in [16,32,64,128,256]:
            use=block[block.score_eligible & block.horizon_step.le(h)]
            y,p=use.target_soh.to_numpy(),use.q50.to_numpy();e=p-y
            sst=float(np.square(y-y.mean()).sum()) if len(y) else 0
            prefix.append(dict(battery_id=battery,origin_cycle=origin,prefix_horizon=h,
                               n_scored=len(y),mae_soh_pp=float(np.mean(abs(e))*100) if len(e) else np.nan,
                               rmse_soh_pp=float(np.sqrt(np.mean(e**2))*100) if len(e) else np.nan,
                               r2=float(1-np.square(e).sum()/sst) if len(y)>1 and sst>0 else np.nan))
    pd.DataFrame(prefix).to_csv(OUT/'soh_prefix_metrics.csv',index=False)
    plot(data,audit,rows,trajectories)
    lines=['# C2 RUL 파일럿: NASA, SOH 경로를 통한 간접 예측','',
           '최초 NASA 4개 배터리 zero-shot 파일럿이다. 입력 64개 → 미래 SOH 256개 → 1.4 Ah 최초 도달 시점으로 RUL을 계산했다.',
           '이 실행은 head-only 실험이나 전체 데이터셋 검증이 아니다. 단위는 NASA 방전 기록 순번이다.','',
           '## 고정한 정의','',
           '- EOL: 유효 실측 용량이 최초로 1.4 Ah 이하인 기록. [NASA 원자료 기준](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets).',
           '- 현재 t까지 64개 이력만 사용한다. 한 번의 256개 예측 중 실제 미래 관측을 넣지 않는다.',
           '- t=64를 주 평가로 고정하고, t=80,96,...에서 이미 EOL을 관측하지 않은 경우에만 추가 평가한다. 수명 비율로 시점을 고르지 않았다.',
           '- 초기 관측 용량으로 정규화한 SOH를 유지하고 EOL 선은 각 배터리에서 1.4/Qref로 변환했다. SOH 70%를 일괄 적용하지 않았다.',
           '- 임계값 최초 도달을 사용하며 순간 하락도 이 정의의 일부다. 3개 연속 하회 시작 시점은 정답 민감도 진단으로만 기록했다.',
           '- B0007은 실제 EOL 미관측이다. 관측 종료를 EOL로 만들지 않고 정확한 RUL MAE/RMSE에서 제외하며 관측 종료 이전의 거짓 도달 예측을 별도 표시한다.',
           '- 예측이 256 이내 임계값에 도달하지 않으면 RUL > 256으로 남긴다. 256으로 채우거나 해당 예측을 조용히 제외하지 않는다.',
           '- q10~q90은 시점별 SOH 구간이며 RUL 신뢰구간이 아니다. 분위수 역전도 수정하지 않는다.',
           '- 제외된 관측은 그래프에 표시하지 않고 이력에서는 NaN 위치로 유지한다. 보간·평활화·단조화·추가 외삽은 없다.','',
           '## 동일한 t=64에서의 결과','',
           '| 배터리 | 실제 EOL | 실제 RUL | 예측 RUL | 예측−실제 |',
           '|---|---:|---:|---:|---:|']
    def fmt(value):return f'{int(value)}' if pd.notna(value) else '—'
    for row in primary.itertuples():
        truth=fmt(row.true_rul) if row.event_observed else f'> {int(row.observed_rul_lower_bound)}'
        pred=fmt(row.predicted_rul) if row.predicted_crossing else '> 256'
        lines.append(f'| {row.battery_id} | {fmt(row.true_eol_cycle)} | {truth} | {pred} | {fmt(row.error_cycles)} |')
    s=summary['primary_t64']
    lines+=['',f"실제 EOL 관측 셀 {s['n_event_batteries']}개 중 유한 RUL 예측 {s['n_finite_event_predictions']}개, 256 내 미도달 {s['n_no_crossing_event_predictions']}개."]
    if s['conditional_mae_cycles'] is not None:
        lines.append(f"유한 예측만의 MAE {s['conditional_mae_cycles']:.2f}, RMSE {s['conditional_rmse_cycles']:.2f}, bias {s['conditional_bias_cycles']:+.2f} (방전 기록 수). 모든 사건 셀의 MAE 하한 {s['mae_lower_bound_all_event_origins']:.2f}.")
    else:
        lines.append(f"유한 예측이 없어 정확한 MAE/RMSE를 계산할 수 없다. 모든 사건 셀의 MAE 하한은 {s['mae_lower_bound_all_event_origins']:.2f}이다.")
    lines+=['','예측 미도달이 있으면 유한 사례 점수는 조건부 점수다. 하한은 실제 오차의 대체 점수가 아니다. 주 평가의 셀 수가 3개뿐이므로 R²는 정의될 때만 기술적으로 제시한다.',
            '',f"주 평가의 기술적 셀 간 R²: {s['descriptive_cross_cell_r2']}. 서로 겹치는 여러 예측 시점을 독립 표본으로 취급하지 않는다.",
            '', '![t64에서의 SOH 예측](trajectories_t64.png)','',
            '## 추가 시점 결과','',
            '배터리별 오차를 계산한 뒤 사건이 관측된 배터리들을 동일 가중 평균했다. 실제 잔여수명이 짧은 추가 시점이 포함되므로 조기 수명 예측 성능으로 일반화하지 않는다.','',
            '| 배터리 | 시점 수 | 실제 EOL 관측 | 유한 예측 수 | 조건부 MAE | 조건부 RMSE | 조건부 bias |',
            '|---|---:|---|---:|---:|---:|---:|']
    for row in cell.itertuples():
        metric=lambda x:f'{x:.2f}' if pd.notna(x) else '—'
        lines.append(f'| {row.battery_id} | {row.n_origins} | {row.n_event_origins>0} | {row.n_finite_event_predictions} | {metric(row.conditional_mae_cycles)} | {metric(row.conditional_rmse_cycles)} | {metric(row.conditional_bias_cycles)} |')
    lines+=['','![시점별 RUL](rul_by_origin.png)','',
            '## 해석 범위와 파일','',
            '- 이 파일럿은 관측된 NASA 4개 이력에 한정된다. 데이터셋 간 일반화나 특정 기준 모델 대비 우위를 주장하지 않는다.',
            '- 실제 정답이 관측된 3개 셀에 대한 오차와 B0007의 우측 검열 결과를 구분한다.',
            '- 저장한 256개 예측의 일부 구간을 채점한 SOH prefix 점수는 별도 64→16 실험 점수와 같지 않다. 정답이 남아 있는 유효 관측부터 EOL까지만 채점한다.',
            '- 기존 8/16/32/64→1 실험 및 데이터는 변경하지 않았다.',
            '- `eol_audit.csv`: EOL·검열·정규화 기준. `rul_predictions.csv`: 모든 예측 시점과 RUL 오차.',
            '- `soh_predictions.csv.gz`: 미래 SOH 분위수와 유효 실측값. `soh_prefix_metrics.csv`: 거리별 가용 점수.',
            '- `protocol.json`, `runtime.json`, `verification.json`: 사전 정의, 실행 환경, 검증.','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return rows,summary


def plot(data,audit,rows,trajectories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for ax,(battery,frame) in zip(axes.flat,data.groupby('battery_id')):
        a=audit.set_index('battery_id').loc[battery]
        line=frame.sort_values('cycle').copy();line['display']=np.where(eligible_mask(line),line.soh*100,np.nan)
        p=trajectories[trajectories.battery_id.eq(battery)&trajectories.origin_cycle.eq(64)]
        score=rows[rows.battery_id.eq(battery)&rows.origin_cycle.eq(64)].iloc[0]
        ax.plot(line.cycle,line.display,color='#262b32',lw=1.8,label='Eligible measurements')
        ax.plot(p.target_cycle,p.q50*100,color='#1768b5',lw=1.8,label='C2 zero-shot q50')
        valid=p.q10.le(p.q90)
        ax.fill_between(p.target_cycle,p.q10*100,p.q90*100,where=valid,color='#1768b5',alpha=.13,label='Pointwise q10-q90')
        ax.axhline(a.threshold_soh*100,color='#c34736',ls='--',lw=1.3,label='EOL capacity = 1.4 Ah')
        ax.axvline(64,color='#808080',ls=':',lw=1.2)
        if a.event_observed:ax.axvline(a.eol_cycle,color='#c34736',ls=':',lw=1.2)
        if score.predicted_crossing:ax.axvline(score.predicted_eol_cycle,color='#1768b5',ls=':',lw=1.2)
        actual=str(int(score.true_rul)) if a.event_observed else f'> {int(score.observed_rul_lower_bound)}'
        pred=str(int(score.predicted_rul)) if score.predicted_crossing else '> 256'
        ax.set_title(f'{battery} | t=64 | true RUL {actual}, predicted {pred}',fontsize=12)
        ax.set_xlabel('Discharge record index');ax.set_ylabel('SOH (% of first measured capacity)')
        ax.grid(alpha=.18);ax.set_xlim(1,320)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=4,fontsize=10)
    fig.suptitle('NASA RUL pilot | 64 observed positions -> 256 future predictions\nNo future measurements supplied within a forecast',fontsize=15)
    fig.savefig(OUT/'trajectories_t64.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(battery,part) in zip(axes.flat,rows.groupby('battery_id')):
        if part.event_observed.all():ax.plot(part.origin_cycle,part.true_rul,'o-',color='#262b32',label='True RUL')
        else:ax.plot(part.origin_cycle,part.observed_rul_lower_bound,'o--',color='#262b32',label='True RUL strictly above this bound')
        valid=part.predicted_crossing
        ax.plot(part.loc[valid,'origin_cycle'],part.loc[valid,'predicted_rul'],'s-',color='#1768b5',label='Predicted RUL')
        if (~valid).any():ax.scatter(part.loc[~valid,'origin_cycle'],np.full((~valid).sum(),256),marker='^',color='#1768b5',label='Predicted RUL > 256')
        ax.set_title(battery);ax.set_xlabel('Prediction origin (discharge record index)');ax.set_ylabel('Remaining discharge records')
        ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle('Each origin uses only its latest 64 observations; censored values remain bounds',fontsize=12)
    fig.savefig(OUT/'rul_by_origin.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--device',default='xpu',choices=['xpu','cuda','cpu'])
    parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    protocol,data,audit,tasks,inputs=prepare()
    if args.prepare_only:return
    if args.report_only:
        with np.load(OUT/'forecast_quantiles.npz') as archive:q=archive['quantiles']
        save_reports(protocol,data,audit,tasks,q);return
    if (OUT/'status.json').exists() and json.loads((OUT/'status.json').read_text())['status']=='complete':
        raise ValueError('Completed run is immutable; use --report-only or a new output directory')
    started=now();write_json(OUT/'status.json',dict(status='loading',started_at=started,protocol_sha256=sha(OUT/'protocol.json')))
    import torch
    from chronos import Chronos2Pipeline
    np.random.seed(42);torch.manual_seed(42)
    pipeline=Chronos2Pipeline.from_pretrained('amazon/chronos-2',revision=REVISION,device_map=args.device,
        dtype=torch.float32,cache_dir=str(ROOT/'models/huggingface'),local_files_only=True)
    pipeline.model.eval()
    runtime=dict(started_at=started,host=platform.node(),platform=platform.platform(),python=platform.python_version(),
                 device=args.device,device_name=(torch.xpu.get_device_name(0) if args.device=='xpu' else torch.cuda.get_device_name(0) if args.device=='cuda' else platform.processor()),
                 packages={p:version(p) for p in ['torch','chronos-forecasting','transformers','numpy','pandas']},
                 model_prediction_length=pipeline.model_prediction_length,
                 uses_autoregressive_horizon_extension=256>pipeline.model_prediction_length,
                 inference_batch_size=4)
    write_json(OUT/'runtime.json',runtime)
    print('MODEL READY',runtime['device_name'],'native horizon',pipeline.model_prediction_length,flush=True)
    write_json(OUT/'status.json',dict(status='predicting',started_at=started,protocol_sha256=sha(OUT/'protocol.json')))
    begin=time.perf_counter();values=[]
    for start in range(0,len(inputs),4):
        with torch.inference_mode():
            q,_=pipeline.predict_quantiles(inputs=[np.asarray(v,dtype=np.float32) for v in inputs[start:start+4]],
                prediction_length=256,context_length=64,quantile_levels=QUANTILES,batch_size=4,cross_learning=False)
        values.extend(v.cpu().numpy()[0] for v in q)
        print('PREDICTED',len(values),'/',len(inputs),flush=True)
    q=np.stack(values)
    if q.shape!=(len(tasks),256,9) or not np.isfinite(q).all():raise ValueError('Invalid model outputs')
    np.savez_compressed(OUT/'forecast_quantiles.npz',quantiles=q)
    rows,summary=save_reports(protocol,data,audit,tasks,q)
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in protocol['data_sha256'].items():assert sha(BUNDLE/name)==digest,name
    # Independently reconstruct threshold hits from persisted trajectories.
    disk=pd.read_csv(OUT/'soh_predictions.csv.gz')
    for row in rows.itertuples():
        forecast=disk[disk.battery_id.eq(row.battery_id)&disk.origin_cycle.eq(row.origin_cycle)]
        hit=forecast[forecast.q50.le(row.threshold_soh)]
        if row.predicted_crossing:assert int(hit.horizon_step.iloc[0])==int(row.predicted_rul)
        else:assert hit.empty
        assert forecast.target_cycle.min()==row.origin_cycle+1
    runtime.update(finished_at=now(),inference_and_report_seconds=time.perf_counter()-begin)
    write_json(OUT/'runtime.json',runtime)
    artifacts=[p for p in OUT.iterdir() if p.is_file() and p.name not in ['verification.json','status.json']]
    write_json(OUT/'verification.json',dict(status='passed',verified_at=now(),n_tasks=len(tasks),
        finite_quantiles=True,persisted_threshold_hits_recomputed=True,source_and_data_hashes_unchanged=True,
        artifacts_sha256={p.name:sha(p) for p in artifacts}))
    write_json(OUT/'status.json',dict(status='complete',started_at=started,finished_at=now(),protocol_sha256=sha(OUT/'protocol.json')))
    print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
