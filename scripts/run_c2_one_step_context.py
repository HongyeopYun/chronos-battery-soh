"""Initial C2 64->1 experiment on identical held-out timestamps."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.battery_zeroshot import ZeroShotConfig, prepare_battery_windows, sha256, write_json
from src.battery_model_zeroshot import load_model, predict_quantiles, phase_labels
from src.battery_fewshot_tuning import select_label_windows
from src.c2_head_protocol import GROUPS, check_physical_split
from src.soh_data import load_bundle
from src.reviewed_soh import POLICY, prepare_reviewed_windows
from src.c2_one_step import (METHODS, crop_windows, fixed_inputs, train_head,
                             restore_head, baselines, point_metrics, probability_metrics)

CONTEXTS = (64,)
OUT = ROOT / "results/c2-battery/one-step-64"
WEIGHTS = ROOT / "models/c2-head-one-step-64"
PRIOR = ROOT / "results/c2-battery/strict-v2-zero-head/protocol.json"
BUNDLES = ROOT / "data/processed/strict_v2"
REVIEWED = False


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def update(state, **kwargs):
    state.update(**kwargs, updated_at=now())
    write_json(OUT / "status.json", state)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = replace(ZeroShotConfig(**read(ROOT / "configs/c2-battery.json")), prediction_length=1)
    prior = read(PRIOR)
    plan = dict(schema_version=1, config=asdict(cfg), contexts=list(CONTEXTS),
                prior_protocol_sha256=sha256(PRIOR), seed=42, steps=100, learning_rates=[1e-6, 1e-5],
                max_windows_per_source_cell=4, head_trainable="output_patch_embedding only",
                matching="All L share 64->1 eligible target times; shorter inputs are suffixes. Source label times also fixed across L.",
                training="Independent initialization from original C2 per L/fold/LR, 4 tasks/step, AdamW, no refit; one actual label per task; 16-wide padding masked; loss multiplied by 16 to restore per-target scale.",
                selection="Validation-only minimum normalized native quantile loss; step zero allowed; decile crossing and invalid interval fractions cannot worsen versus step zero. All 13 heads locked before test scoring. Only context 64 in this experiment.",
                aggregation="Arithmetic mean of per-battery MAE/RMSE. temporal_r2 is computed across one-step timestamps within each battery/phase, then averages defined batteries with explicit counts. No per-window R2 for horizon one.",
                groups={}, source_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p)
                    for p in [Path(__file__), ROOT / 'src/c2_one_step.py', ROOT / 'src/battery_model_zeroshot.py',
                              ROOT / 'src/soh_data.py', ROOT / 'src/calce_windows.py']})
    if REVIEWED:
        plan['review_policy'] = POLICY
        plan['matching'] = 'Only 64 original past positions -> eligible y_t. NaN at excluded input positions. All models use identical target times and identical input masks.'
        plan['source_sha256']['src/reviewed_soh.py'] = sha256(ROOT / 'src/reviewed_soh.py')
        plan['bundle_root'] = str(BUNDLES)
    prepared, budgets, coverages = {}, [], []
    mit_source, mit_test = set(), set()
    for group in GROUPS:
        records, coverage, data, _ = (prepare_reviewed_windows if REVIEWED else prepare_battery_windows)(BUNDLES / group, cfg)
        by_id = {r['battery_id']: r for r in records}
        folds = [{k: f[k] for k in ('fold_id', 'train_ids', 'validation_ids', 'test_ids')}
                 for f in prior['groups'][group]['folds']]
        source = {b for f in folds for key in ('train_ids', 'validation_ids') for b in f[key]}
        if source - by_id.keys():
            raise ValueError('A locked source cell has no 64->1 window')
        count = min(4, min(len(by_id[b]['windows']['target']) for b in source))
        for fold in folds:
            if set(fold['test_ids']) - by_id.keys():
                raise ValueError('A previously tested cell lost one-step eligibility')
            check_physical_split(fold, {b: r['physical_battery_id'] for b, r in by_id.items()})
            labels = []
            for split, key in [('train', 'train_ids'), ('validation', 'validation_ids')]:
                for b in fold[key]:
                    w = by_id[b]['windows']
                    for i in select_label_windows(w, count):
                        row = dict(dataset=group, fold_id=fold['fold_id'], split=split, battery_id=b,
                                   index=int(i), target_cycle=int(w['target_cycle'][i, 0]))
                        labels.append(row)
                        for length in CONTEXTS:
                            cw = crop_windows(w, length)
                            budgets.append(dict(**row, context_length=length,
                                                context_start=int(cw['context_cycle'][i, 0]),
                                                context_end=int(cw['context_cycle'][i, -1]),
                                                observed_inputs=int(np.isfinite(cw['context'][i]).sum())))
            fold['label_budget'] = labels
            if group.startswith('mit_'):
                mit_source.update(by_id[b]['physical_battery_id'] for b in fold['train_ids'] + fold['validation_ids'])
                mit_test.update(by_id[b]['physical_battery_id'] for b in fold['test_ids'])
        ids = [b for f in folds for b in f['test_ids']]
        assert len(ids) == len(set(ids))
        coverage['group'] = group
        coverage['role'] = ['held_out_test' if b in ids else 'source_or_validation' if b in source else 'not_in_locked_split'
                            for b in coverage.battery_id]
        coverages.append(coverage)
        plan['groups'][group] = dict(folds=folds, windows_per_source_cell=count,
            train_windows=2*count, validation_windows=count, test_cells=len(ids),
            test_windows=sum(len(by_id[b]['windows']['target']) for b in ids),
            bundle_sha256=sha256(BUNDLES / group / 'manifest.json'),
            data_sha256=sha256(BUNDLES / group / 'soh.csv'))
        prepared[group] = dict(by_id=by_id, data=data)
    if mit_source & mit_test:
        raise ValueError('MIT physical cell crosses source/test groups')
    plan = json.loads(json.dumps(plan))
    if (OUT / 'protocol.json').exists() and read(OUT / 'protocol.json') != plan:
        raise ValueError('Protocol or source changed; use a separate output directory')
    write_json(OUT / 'protocol.json', plan)
    pd.DataFrame(budgets).to_csv(OUT / 'label_budget.csv', index=False)
    pd.concat(coverages).to_csv(OUT / 'coverage.csv', index=False)
    runtime = OUT / '.runtime'
    runtime.mkdir(exist_ok=True)
    (runtime / 'executed_runner.py').write_bytes(Path(__file__).read_bytes())
    for group, spec in plan['groups'].items():
        print(f"PLAN {group}: {spec['train_windows']}/{spec['validation_windows']} train/val; {spec['test_cells']} test cells, {spec['test_windows']} matched targets", flush=True)
    return cfg, plan, prepared


def selection_path(group, length, fold):
    return OUT / group / f'L{length}' / fold


def fit_all(pipeline, initial, plan, prepared, state):
    completed = 0
    frozen = {n: p._version for n, p in pipeline.model.named_parameters() if n not in initial}
    for group, spec in plan['groups'].items():
        for length in CONTEXTS:
            for fold in spec['folds']:
                directory = selection_path(group, length, fold['fold_id'])
                if (directory / 'selected.json').exists():
                    chosen = read(directory / 'selected.json')
                    if sha256(chosen['selected_weights']) != chosen['selected_weights_sha256']:
                        raise ValueError('Selected head checkpoint changed')
                else:
                    update(state, status='training', dataset=group, context_length=length,
                           fold=fold['fold_id'], completed_heads=completed, total_heads=13)
                    tasks = []
                    for split in ('train', 'validation'):
                        values = []
                        for label in fold['label_budget']:
                            if label['split'] == split:
                                w = crop_windows(prepared[group]['by_id'][label['battery_id']]['windows'], length)
                                values.extend(fixed_inputs(w, [label['index']]))
                        tasks.append(values)
                    print(f"FIT {group} L{length} {fold['fold_id']}", flush=True)
                    chosen = train_head(pipeline, initial, tasks, length, directory,
                        WEIGHTS / group / f'L{length}' / f"{fold['fold_id']}.pt")
                    print(f"SELECT {group} L{length} {fold['fold_id']}: step={chosen['step']} lr={chosen['learning_rate']:g}", flush=True)
                completed += 1
    if any(p._version != frozen[n] for n, p in pipeline.model.named_parameters() if n in frozen):
        raise ValueError('Frozen backbone changed')
    restore_head(pipeline.model, initial)
    update(state, status='all_heads_locked', completed_heads=completed, all_heads_locked_at=now(), frozen_backbone_verified=True)


def inference(pipeline, cfg, windows, path, weights_hash):
    fingerprint = sha256(OUT / 'protocol.json')
    if path.exists():
        with np.load(path) as z:
            assert str(z['protocol']) == fingerprint and str(z['weights']) == weights_hash
            np.testing.assert_array_equal(z['target'], windows['target'])
            np.testing.assert_array_equal(z['forecast_start'], windows['forecast_start'])
            return z['quantiles'].copy()
    q = predict_quantiles(pipeline, windows['context'], cfg)[1][:, 0]
    if q.shape != (len(windows['target']), 9) or not np.isfinite(q).all():
        raise ValueError('Invalid one-step predictions')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, quantiles=q, target=windows['target'], forecast_start=windows['forecast_start'],
                        protocol=fingerprint, weights=weights_hash)
    return q


def score(pipeline, cfg, initial, plan, prepared, state):
    rows = []
    for group, spec in plan['groups'].items():
        frames = []
        for length in CONTEXTS:
            config = replace(cfg, context_length=length)
            for fold in spec['folds']:
                selected = read(selection_path(group, length, fold['fold_id']) / 'selected.json')
                for b in fold['test_ids']:
                    update(state, status='test_inference', dataset=group, context_length=length, battery=b)
                    record = prepared[group]['by_id'][b]
                    w = crop_windows(record['windows'], length)
                    path = OUT / group / '.runtime' / f'L{length}_{b}'
                    restore_head(pipeline.model, initial)
                    zero = inference(pipeline, config, w, path.with_name(path.name + '_zero.npz'), cfg.revision)
                    if selected['step'] == 0:
                        head = zero.copy()
                    else:
                        weights = torch.load(selected['selected_weights'], map_location='cpu', weights_only=True)
                        restore_head(pipeline.model, weights)
                        head = inference(pipeline, config, w, path.with_name(path.name + '_head.npz'), selected['selected_weights_sha256'])
                    points = {**baselines(w['context']), 'zero-shot': zero[:, 4], 'head-only': head[:, 4]}
                    quantiles = {'zero-shot': zero, 'head-only': head}
                    data = prepared[group]['data']
                    cycles = data[(data.battery_id == b) & data.is_observed].cycle.to_numpy()
                    phases = phase_labels(w['forecast_start'], cycles)
                    target = w['target'][:, 0]
                    for method in METHODS:
                        frame = pd.DataFrame(dict(dataset=group, battery_id=b, physical_battery_id=record['physical_battery_id'],
                            fold_id=fold['fold_id'], context_length=length, method=method,
                            target_cycle=w['target_cycle'][:, 0], phase=phases, target=target, prediction=points[method],
                            context_observed=np.isfinite(w['context']).sum(axis=1),
                            last_observed_lag=w.get('last_observed_lag',np.ones(len(target),dtype=int))))
                        if method in quantiles:
                            for i in range(9):
                                frame[f'q{(i+1)*10}'] = quantiles[method][:, i]
                        frames.append(frame)
                        for phase in ('all', 'early', 'middle', 'late'):
                            mask = np.ones(len(target), dtype=bool) if phase == 'all' else phases == phase
                            if not mask.any():
                                continue
                            row = dict(dataset=group, battery_id=b, physical_battery_id=record['physical_battery_id'],
                                       fold_id=fold['fold_id'], context_length=length, method=method, phase=phase,
                                       n_targets=int(mask.sum()), **point_metrics(target[mask], points[method][mask]))
                            if method in quantiles:
                                row.update(probability_metrics(target[mask], quantiles[method][mask]))
                            rows.append(row)
            print(f'TEST DONE {group} L{length}', flush=True)
        directory = OUT / group
        predictions = pd.concat(frames, ignore_index=True)
        predictions.to_csv(directory / 'predictions.csv.gz', index=False)
        plot_predictions(group, predictions, spec)
        pd.DataFrame(rows).to_csv(OUT / 'battery_metrics.csv', index=False)
    restore_head(pipeline.model, initial)
    summarize(plan)
    verification = verify(plan)
    write_json(OUT / 'verification.json', verification)
    update(state, status='complete', finished_at=now(), **verification)


def summarize(plan):
    battery = pd.read_csv(OUT / 'battery_metrics.csv')
    rows = []
    for (group, length, method, phase), table in battery.groupby(['dataset', 'context_length', 'method', 'phase'], sort=False):
        row = dict(dataset=group, context_length=length, method=method, phase=phase,
                   n_batteries=len(table), n_targets=int(table.n_targets.sum()),
                   n_temporal_r2_defined=int(table.temporal_r2.notna().sum()),
                   n_temporal_r2_undefined=int(table.temporal_r2.isna().sum()), temporal_r2=float(table.temporal_r2.mean()))
        for metric in ('mae', 'rmse', 'bias', 'wql', 'picp80', 'width80', 'invalid_interval_fraction', 'quantile_crossing_fraction'):
            row[metric] = float(table[metric].mean(skipna=False))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'summary.csv', index=False)
    selections = []
    for group, spec in plan['groups'].items():
        for length in CONTEXTS:
            for fold in spec['folds']:
                s = read(selection_path(group, length, fold['fold_id']) / 'selected.json')
                selections.append(dict(dataset=group, context_length=length, fold_id=fold['fold_id'],
                    learning_rate=s['learning_rate'], step=s['step'], validation_loss=s['validation_loss'],
                    baseline_loss=s['baseline']['validation_loss'], train_windows=s['train_windows'],
                    validation_windows=s['validation_windows']))
    pd.DataFrame(selections).to_csv(OUT / 'selected_checkpoints.csv', index=False)
    plot_sweep(summary)
    if REVIEWED:
        write_reviewed_report(plan, summary)
        return
    lines = ['# C2 SOH: 과거 64개 → 다음 1개', '',
        '> 전처리 재검토: 이 표는 기존 strict_v2 마스크의 부분 평가 결과다. NASA 중간 구간 제외, MIT Batch 3의 대규모 적분 대조 보류, CALCE의 낮은 구간 커버리지가 확인됐다. 전체 열화 구간 성능이나 확정 모델 순위로 사용하지 않는다. [전체 감사](../../preprocessing_review_all/report.md)', '',
        '입력 SOH 64개로 바로 다음 SOH 1개를 예측했다. 모든 모델을 같은 테스트 시점에서 비교했다.', '',
        '## 실험 조건', '',
        '- 사용자 요청에 따라 64→1만 먼저 실행했다. 입력 길이 감소·구간 calibration·RUL은 이번 결과에 포함하지 않았다.',
        '- strict_v2 정제 데이터와 기존 배터리 분할을 사용했다. 제외된 MIT 6개 배터리는 포함하지 않는다. NASA/CALCE는 4-fold, MIT/XJTU는 기존 고정 분할이다.',
        '- 64→1의 모든 유효 테스트 시점을 사용했다. 기존 64→16과 테스트 시점 수가 달라 점수를 직접 비교하지 않는다.',
        '- 학습/검증은 별도 배터리의 완료된 이력에서 source별 최대 4개 단일 정답을 시점만으로 균등 선택했다. context에 포함된 SOH도 관측 비용이며 label_budget.csv에 정확한 범위를 기록했다. 기존 64→16의 학습창 수와 정답 개수가 같다는 뜻은 아니다.',
        '- 각 길이·fold·LR은 순정 C2에서 독립적으로 시작했다. 본체 고정, output head만 학습. LR 1e-6/1e-5, 각 100 steps, batch 4, seed 42, 20 steps마다 검증. step 0 허용, validation refit 없음.',
        '- C2는 내부적으로 16개 출력 patch를 만들지만 정답은 1개만 제공하고 나머지 15개는 loss에서 마스킹한다. 패딩으로 희석된 loss를 16배 보정했다. 학습 손실은 native quantile loss이며 검증의 역전 비율이 순정보다 악화되지 않는 후보 중 선택한다.',
        '- 모든 head 선택을 잠근 뒤 테스트를 평가했다. 한 seed의 첫 실험이며 일반화 결론에는 반복 검증이 필요하다.',
        '- 주지표는 배터리별 MAE·RMSE의 동일 가중 평균이다. SOH 비율 단위이므로 0.001은 0.1 percentage point다. LastValue는 직전 실제 SOH, Linear는 주어진 입력 이력의 선형 외삽이다.',
        '- 한 점짜리 window의 R²는 정의되지 않아 보고하지 않는다. CSV의 temporal_r2는 배터리별 여러 one-step 시점 전체를 모아 계산한 보조 지표다. 기존 Mean window R²와 다르며 미정의 배터리 수를 기록한다.',
        '- 모델 중앙값과 분위수는 정렬·clipping하지 않는다. q10>q90이 하나라도 있으면 해당 배터리/그룹의 PICP80·폭을 NaN으로 둔다. 이번 작업에는 구간 보정이 없다.',
        '- 실제 관측값으로 매 시점 입력을 갱신하는 rolling 1-step 예측이다. 원시 센서 기반 현재 SOH 추정과 구분한다. 초·중·후반은 원래 관측 범위를 3등분한 것이며 전체 수명 구간을 뜻하지 않는다.',
        '- 엄격한 품질 마스크를 통과한 관측 조건부 결과다. CALCE의 간격은 보존된 관측 순서를 뜻하며 파일 사이 누락 cycle의 불확실성이 남는다.', '',
        '| 데이터셋 | 길이 | 모델 | 배터리 / 시점 | MAE | RMSE |', '|---|---:|---|---:|---:|---:|']
    for group in GROUPS:
        for length in CONTEXTS:
            for method in METHODS:
                r = summary[(summary.dataset == group) & (summary.context_length == length) & (summary.method == method) & (summary.phase == 'all')].iloc[0]
                lines.append(f"| {group} | {length} | {method} | {r.n_batteries} / {r.n_targets:,} | {r.mae:.6f} | {r.rmse:.6f} |")
    lines += ['', '## 모델별 오차', '', '![64 to 1 comparison](comparison.png)', '',
              '[요약 CSV](summary.csv) · [배터리별 CSV](battery_metrics.csv) · [학습창](label_budget.csv) · [체크포인트](selected_checkpoints.csv) · [프로토콜](protocol.json)', '',
              '## 대표 배터리의 rolling 예측', '', '대표 배터리는 기존 분할의 첫 테스트 배터리로 점수와 무관하게 고정했다. 검은 선은 평가 여부와 무관한 전체 실제 관측이며, 주황 표식은 strict_v2 제외·보류 관측이다. 예측선만 평가 시점에 표시하고 예측 공백을 연결하지 않는다. 과거 결과의 예측값·평가 마스크는 변경하지 않았다.', '']
    for group in GROUPS:
        lines += [f'### {group}', '', f'![{group}]({group}/representative.png)', '', f'[전체 단일 시점 예측 CSV]({group}/predictions.csv.gz)', '']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_reviewed_report(plan, summary):
    coverage = pd.read_csv(OUT / 'coverage.csv')
    selected = pd.read_csv(OUT / 'selected_checkpoints.csv')
    held = coverage[coverage.role.eq('held_out_test')]
    old = pd.read_csv(ROOT / 'results/c2-battery/one-step-64/coverage.csv')
    old = old[old.role.eq('held_out_test')].groupby('group').evaluated_windows.sum()
    counts = held.groupby('group').agg(records=('battery_id','size'),targets=('evaluated_windows','sum'),
        missing_context=('targets_with_missing_context','sum'),min_observed=('min_context_observed','min'))
    counts['previous_targets'] = old
    counts.to_csv(OUT / 'coverage_comparison.csv')
    subsets=[]
    for group in GROUPS:
        predictions=pd.read_csv(OUT/group/'predictions.csv.gz')
        for (b,method),df in predictions.groupby(['battery_id','method']):
            for scope,mask in [('complete_context',df.context_observed.eq(64)),('missing_context',df.context_observed.lt(64))]:
                d=df[mask]
                if len(d):subsets.append(dict(dataset=group,battery_id=b,method=method,scope=scope,n_targets=len(d),**point_metrics(d.target,d.prediction)))
    subset=pd.DataFrame(subsets)
    subset.to_csv(OUT/'context_subset_battery_metrics.csv',index=False)
    subset.groupby(['dataset','method','scope']).agg(n_batteries=('battery_id','size'),n_targets=('n_targets','sum'),
        mae=('mae','mean'),rmse=('rmse','mean')).to_csv(OUT/'context_subset_summary.csv')
    lines=['# C2 현재 SOH 예측: 전처리 재검토 후 64→1 재실험','',
        '입력은 t-64~t-1의 SOH 기록 위치 64개, 정답은 y_t 하나다. 실제 과거 관측으로 매번 입력을 갱신하는 rolling one-step이며 센서 신호만으로 현재 SOH를 추정한 실험은 아니다. 입력 길이 감소·RUL은 실행하지 않았다.','',
        '## 데이터와 평가 범위','',
        '- `reviewed_native_v3`: 원본 Q/Qref·SOH·cycle은 유지했다. NASA의 충전 로그 부재 3건은 독립 방전 대조 후 유지했다. MIT의 원본 QDischarge를 우선하며 적분 불일치·기록 공백만으로 제외하지 않는다. MIT의 CV 계측 불일치 원인이 해결됐다는 뜻은 아니다. 카운터 reset·충방전 혼재·기타 실제 이상과 사전 noisy 채널 6개는 제외한다.',
        '- CALCE의 불완전 충전 관측은 정답에서 제외한다. 과거 입력에서 제외된 위치는 NaN으로 두고 C2의 native mask로 처리한다. 행을 삭제해 시간을 압축하거나 SOH를 보간하지 않는다. 결측이 있으면 유효 입력은 64개보다 적다.',
        '- 선형 기준선이 정의되도록 유효 과거 입력 최소 2개를 요구한다. 실제 최소 유효 입력 수는 아래 표에 보고한다. LastValue는 64개 위치 안의 마지막 유효 SOH, Linear는 유효값을 원래 위치 좌표에 회귀해 t를 예측한다.',
        '- 모든 모델은 같은 유효 정답 시점·같은 입력값과 마스크로 평가했다. CALCE는 파일 간 누락 물리 cycle이 알려지지 않은 관측 순서이며, MIT 연속 기록은 기존 record 경계에서 입력창을 재시작한다.',
        '- 기존과 대상 시점·일부 학습 label 위치가 바뀌었으므로 과거 strict_v2 점수와 수치만 비교해 모델 개선율로 해석하지 않는다.','',
        '| 데이터셋 | 테스트 기록 | 이전 → 이번 정답 시점 | 결측 입력이 있는 시점 | 최소 유효 입력 / 64 |',
        '|---|---:|---:|---:|---:|']
    for g in GROUPS:
        r=counts.loc[g]
        lines.append(f'| {g} | {int(r.records)} | {int(r.previous_targets):,} → {int(r.targets):,} | {int(r.missing_context):,} | {int(r.min_observed)} |')
    lines+=['','## 학습·지표','',
        '- C2 `amazon/chronos-2`, pinned revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, float32. Zero-shot은 순정 가중치, head-only는 output_patch_embedding만 학습하고 본체를 고정했다.',
        '- 기존 물리 배터리 분할 유지: NASA/CALCE 4-fold, 나머지 고정 분할, 총 13개 head. 각 fold는 별도 train 배터리 2개 × 정답 4개 = 8개 학습 label, validation 배터리 1개 × 4개 = 4개 검증 label이다. 테스트 배터리 자체를 calibration한 실험은 아니다. 과거 입력 SOH도 관측 비용이다.',
        '- 원래 C2에서 fold/LR별 독립 초기화, LR 1e-6/1e-5, 각 100 steps, batch 4, seed 42. 20 steps마다 검증하고 step 0도 후보에 포함했다. 정답 1개만 loss에 사용하고 내부 16-wide patch의 나머지는 mask한다. 모든 head 선택을 끝낸 뒤 테스트 추론을 실행했다.',
        '- 선택은 validation normalized native quantile loss이며 분위수 역전·잘못된 구간 비율이 순정보다 나빠지는 후보는 선택하지 않는다. 테스트 성능으로 학습·정제·체크포인트를 고르지 않았다.',
        '- 아래 RMSE는 배터리별 RMSE의 동일 가중 평균이며 **SOH percentage point(pp)** 단위다. CSV는 SOH 비율 단위다. 1-point window R²는 정의되지 않아 사용하지 않는다. CSV의 temporal_r2는 여러 예측 시점을 모은 보조 지표다.',
        '- 분위수를 사후 정렬·clipping하지 않았다. q10>q90인 관측이 있으면 해당 배터리의 PICP/구간폭은 NaN으로 노출한다. 단일 seed 결과다.','',
        '| 데이터셋 | LastValue RMSE(pp) | Linear RMSE(pp) | C2 zero-shot RMSE(pp) | C2 head-only RMSE(pp) |',
        '|---|---:|---:|---:|---:|']
    for g in GROUPS:
        r=summary[summary.dataset.eq(g)&summary.phase.eq('all')].set_index('method')
        lines.append('| '+g+' | '+' | '.join(f'{100*r.loc[m,"rmse"]:.5f}' for m in METHODS)+' |')
    lines+=['','![모델별 RMSE](comparison.png)','','## 선택한 head','',
        '| 데이터셋 / fold | LR | step | 학습 / 검증 label |','|---|---:|---:|---:|']
    for r in selected.itertuples():
        lines.append(f'| {r.dataset} / {r.fold_id} | {r.learning_rate:g} | {r.step} | {r.train_windows} / {r.validation_windows} |')
    lines+=['','step 0은 검증 기준에서 학습한 head가 선택되지 않아 순정 C2와 같은 결과라는 뜻이다.','',
        '## 이전 누락 사례의 재예측','',
        'NASA B0005, CALCE CS2_35, MIT b3c9는 이전 감사에서 구간 누락을 확인한 사례로, 모델 점수와 무관하게 표시했다. 검은 선은 전체 실제 측정값이고 예측선만 평가 가능한 정답 시점에 표시한다.','',
        '![복원된 평가 구간](restored_cases.png)','','## 대표 테스트 배터리','']
    for g in GROUPS:lines+=[f'### {g}','',f'![{g}]({g}/representative.png)','']
    lines+=['## 상세 결과와 재현','',
        '[프로토콜](protocol.json) · [검증 결과](verification.json) · [요약 점수](summary.csv) · [배터리별·초중후반 점수](battery_metrics.csv) · [체크포인트](selected_checkpoints.csv) · [학습 label·입력 범위](label_budget.csv)',
        '[평가 범위](coverage_comparison.csv) · [결측 유무별 점수](context_subset_summary.csv) · [데이터 변경 내역](../../preprocessing_reviewed_native_v3/reclassified_observations.csv) · [전처리 정책](../../preprocessing_reviewed_native_v3/policy.json)',
        '재현 명령: `.venv\\Scripts\\python.exe scripts/run_c2_one_step_context.py --reviewed --stage all`. 기존 strict_v2 결과·가중치와 다른 디렉터리에 저장했다.','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    fig,axes=plt.subplots(3,1,figsize=(13,12))
    for ax,(g,b) in zip(axes,[('nasa_aging','B0005'),('calce_cs2_1c','CS2_35'),('mit_batch3','b3c9')]):
        d,_,_=load_bundle(BUNDLES/g);d=d[d.battery_id.eq(b)].set_index('cycle')
        index=np.arange(d.index.min(),d.index.max()+1);d=d.reindex(index)
        ax.plot(index,d.soh.where(d.is_observed.eq(True)&d.capacity_ah.gt(0)),color='black',lw=1.2,label='All measured SOH')
        rejected=d.quality_eligible.eq(False)&d.capacity_ah.gt(0)
        ax.scatter(index[rejected],d.loc[rejected,'soh'],color='#d75a28',marker='x',s=18,label='Excluded label')
        p=pd.read_csv(OUT/g/'predictions.csv.gz');p=p[p.battery_id.eq(b)]
        for m,c in [('zero-shot','#2375ad'),('head-only','#934daf')]:
            points=p[p.method.eq(m)].set_index('target_cycle').reindex(index)
            ax.plot(index,points.prediction,color=c,lw=.9,label=m)
        ax.set(title=f'{g} | {b}',xlabel='Original record position',ylabel='SOH');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'restored_cases.png',dpi=160);plt.close(fig)


def plot_sweep(summary):
    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    colors = dict(zip(METHODS, ['#777777', '#c78c20', '#2375ad', '#934daf']))
    for ax, group in zip(axes.flat, GROUPS):
        table = summary[(summary.dataset == group) & (summary.phase == 'all')]
        for method in METHODS:
            rows = table[table.method == method].set_index('context_length').loc[list(CONTEXTS)]
            ax.bar(method, rows.rmse.iloc[0] * 100, label=method, color=colors[method])
        ax.set(title=group, ylabel='RMSE (SOH percentage points)')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    axes.flat[-1].axis('off')
    detail = ('64 past positions -> 1 eligible SOH\nExcluded input positions use native NaN masks\nSame held-out targets across models\nEqual battery weight; reviewed native Q\nSingle seed; validation-only selection' if REVIEWED else '64 past SOH observations -> 1 target\nSame held-out targets across models\nEqual battery weight; strict eligible data\nSingle seed; validation-only selection')
    axes.flat[-1].text(.02, .8, detail, va='top', fontsize=11, linespacing=1.8)
    fig.suptitle('Chronos-2: 64 past SOH values -> one next value', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(OUT / 'comparison.png', dpi=150)
    plt.close(fig)


def plot_predictions(group, predictions, spec):
    b = spec['folds'][0]['test_ids'][0]
    data, _, _ = load_bundle(BUNDLES / group)
    observed = data[data.battery_id.eq(b)].set_index('cycle')
    full_index = np.arange(observed.index.min(), observed.index.max()+1)
    observed = observed.reindex(full_index)
    truth = observed.soh.where(observed.is_observed.eq(True) & observed.capacity_ah.gt(0))
    fig, axes = plt.subplots(1, 1, figsize=(12, 4.5), squeeze=False)
    for ax, length in zip(axes.flat, CONTEXTS):
        selected = predictions[(predictions.battery_id == b) & (predictions.context_length == length)]
        # Actual measurements exist outside scored targets too. Show the full
        # measured trajectory; predictions alone break at unevaluated times.
        ax.plot(observed.index, truth, color='black', lw=1.3, label='All measured SOH')
        rejected = observed.quality_eligible.eq(False) & truth.notna()
        ax.scatter(observed.index[rejected], truth[rejected], color='#d75a28', s=18,
                   marker='x', zorder=5, label='Excluded / quarantined observation')
        for method, color in [('LastValue', '#888888'), ('zero-shot', '#2375ad'), ('head-only', '#934daf')]:
            values = selected[selected.method == method].set_index('target_cycle').reindex(full_index)
            ax.plot(values.index, values.prediction, lw=.9, color=color, label=method)
        ax.set(title=f'L={length} -> 1', xlabel='Retained cycle / observation index', ylabel='SOH')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'{group} | {b}\nFull measured curve; predictions shown only at evaluated times')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / group / 'representative.png', dpi=150)
    plt.close(fig)


def verify(plan):
    battery = pd.read_csv(OUT / 'battery_metrics.csv')
    summary = pd.read_csv(OUT / 'summary.csv')
    total = 0
    for group, spec in plan['groups'].items():
        df = pd.read_csv(OUT / group / 'predictions.csv.gz')
        ref = df[(df.context_length == 64) & (df.method == 'LastValue')].sort_values(['battery_id', 'target_cycle'])
        assert len(ref) == spec['test_windows'] and ref.battery_id.nunique() == spec['test_cells']
        if REVIEWED:
            cfg = ZeroShotConfig(**plan['config'])
            records, _, _, _ = prepare_reviewed_windows(BUNDLES / group, cfg)
            for record in records:
                if record['battery_id'] not in set(ref.battery_id):
                    continue
                w = record['windows']
                actual = ref[ref.battery_id.eq(record['battery_id'])]
                np.testing.assert_array_equal(actual.target_cycle,w['forecast_start'])
                np.testing.assert_allclose(actual.target,w['target'][:,0],rtol=1e-12)
                np.testing.assert_array_equal(actual.context_observed,w['context_observed'])
                np.testing.assert_array_equal(actual.last_observed_lag,w['last_observed_lag'])
                np.testing.assert_allclose(actual.prediction,baselines(w['context'])['LastValue'],rtol=1e-12)
        for (length, method), table in df.groupby(['context_length', 'method']):
            ordered = table.sort_values(['battery_id', 'target_cycle'])
            assert not ordered.duplicated(['battery_id', 'target_cycle']).any()
            pd.testing.assert_frame_equal(ordered[['battery_id', 'target_cycle', 'target']].reset_index(drop=True),
                                          ref[['battery_id', 'target_cycle', 'target']].reset_index(drop=True))
            for b, cell in table.groupby('battery_id'):
                for phase in ('all', 'early', 'middle', 'late'):
                    part = cell if phase == 'all' else cell[cell.phase == phase]
                    if part.empty:
                        continue
                    r = battery[(battery.dataset == group) & (battery.context_length == length) &
                                (battery.method == method) & (battery.battery_id == b) & (battery.phase == phase)].iloc[0]
                    err = part.prediction.to_numpy() - part.target.to_numpy()
                    np.testing.assert_allclose([r.mae, r.rmse], [abs(err).mean(), np.sqrt((err**2).mean())], rtol=1e-8, atol=1e-12)
        total += len(ref)
    expected = battery.groupby(['dataset', 'context_length', 'method', 'phase'])[['mae', 'rmse']].mean().sort_index()
    actual = summary.set_index(['dataset', 'context_length', 'method', 'phase'])[['mae', 'rmse']].sort_index()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    return dict(matched_targets_per_condition=total, independent_metric_recalculation=True,
                reviewed_data_and_context_masks_verified=REVIEWED,
                identical_targets_all_contexts=True, conditions=len(CONTEXTS)*len(METHODS), verified_at=now())


def main():
    global REVIEWED, OUT, WEIGHTS, BUNDLES
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'train', 'finish', 'report', 'all'], default='all')
    parser.add_argument('--reviewed', action='store_true', help='Use reviewed-native v3 and missing-context masks in a separate run')
    args = parser.parse_args()
    if args.reviewed:
        REVIEWED = True
        OUT = ROOT / 'results/c2-battery/one-step-64-reviewed-v3'
        WEIGHTS = ROOT / 'models/c2-head-one-step-64-reviewed-v3'
        BUNDLES = ROOT / 'data/processed/reviewed_native_v3'
    if args.stage == 'report':
        # Presentation-only refresh from completed immutable prediction tables.
        plan = read(OUT / 'protocol.json')
        if read(OUT / 'status.json')['status'] != 'complete':
            raise ValueError('Finish inference before rebuilding the report')
        before = {g: sha256(OUT / g / 'predictions.csv.gz') for g in plan['groups']}
        summarize(plan)
        for group, spec in plan['groups'].items():
            plot_predictions(group, pd.read_csv(OUT / group / 'predictions.csv.gz'), spec)
        assert before == {g: sha256(OUT / g / 'predictions.csv.gz') for g in plan['groups']}
        write_json(OUT / 'rendering.json', dict(rendered_at=now(), runner_sha256=sha256(__file__),
            prediction_sha256=before, prediction_tables_unchanged=True,
            note='Presentation-only: full measured SOH is visible; only prediction curves break across unevaluated times. Original execution source hashes remain in protocol.json.'))
        return
    cfg, plan, prepared = prepare()
    if args.stage == 'prepare':
        return
    state = read(OUT / 'status.json') if (OUT / 'status.json').exists() else dict(started_at=now())
    pipeline = load_model(cfg)
    initial = {n: p.detach().cpu().clone() for n, p in pipeline.model.named_parameters() if n.startswith('output_patch_embedding.')}
    update(state, status='loaded', device=str(pipeline.model.device), protocol_sha256=sha256(OUT / 'protocol.json'))
    try:
        if args.stage in ('train', 'all'):
            fit_all(pipeline, initial, plan, prepared, state)
        if args.stage in ('finish', 'all'):
            if 'all_heads_locked_at' not in state:
                raise ValueError('Lock all head choices before any test evaluation')
            score(pipeline, cfg, initial, plan, prepared, state)
    except Exception as error:
        update(state, status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
