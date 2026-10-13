"""16->1 follow-up with the frozen reviewed 64->1 labels and test timestamps."""
import argparse
import json
import os
import platform
from importlib.metadata import version, PackageNotFoundError
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if '--download-model' in sys.argv:
    os.environ['HF_HUB_OFFLINE'] = '0'

import numpy as np
import pandas as pd
from scripts import run_c2_one_step_context as base
from src.c2_one_step import crop_windows
from src.c2_context_comparison import METHODS, paired_metrics, summarize_paired

PRIOR = ROOT / 'results/c2-battery/one-step-64-reviewed-v3'
OUT = ROOT / 'results/c2-battery/one-step-16-reviewed-v3'
BUNDLES = ROOT / 'data/processed/reviewed_native_v3'
WEIGHTS = ROOT / 'models/c2-head-one-step-16-reviewed-v3'


def cropped(windows, length):
    result = crop_windows(windows, length)
    finite = np.isfinite(result['context'])
    if not finite.any(axis=1).all():
        raise ValueError('A matched target has no observed input in its shorter history')
    result['context_observed'] = finite.sum(axis=1)
    result['last_observed_lag'] = length-np.where(finite, np.arange(length), -1).max(axis=1)
    return result


def persistence(context):
    context = np.asarray(context)
    finite = np.isfinite(context)
    if context.ndim != 2 or np.isinf(context).any() or not finite.any(axis=1).all():
        raise ValueError('A finite past observation is required')
    last = np.where(finite, np.arange(context.shape[1]), -1).max(axis=1)
    return {'LastValue': context[np.arange(len(context)), last]}


def configure():
    # Keep the frozen 64-run source and outputs intact; reuse its training engine.
    base.OUT, base.WEIGHTS, base.BUNDLES = OUT, WEIGHTS, BUNDLES
    base.CONTEXTS, base.METHODS, base.REVIEWED = (16,), METHODS, True
    base.crop_windows, base.baselines = cropped, persistence
    base.plot_sweep = lambda summary: None
    base.write_reviewed_report, base.verify = write_report, verify


def check_frozen_prior():
    prior = base.read(PRIOR / 'protocol.json')
    release = base.read(PRIOR / 'release_verification.json')
    if base.read(PRIOR / 'status.json')['status'] != 'complete':
        raise ValueError('The 64->1 reference must be complete')
    assert base.sha256(PRIOR / 'protocol.json') == release['protocol_sha256']
    for path, digest in prior['source_sha256'].items():
        assert base.sha256(ROOT / path) == digest, path
    for group, digest in release['predictions_sha256'].items():
        assert base.sha256(PRIOR / group / 'predictions.csv.gz') == digest, group
    prior32 = ROOT / 'results/c2-battery/one-step-32-reviewed-v3'
    release32 = base.read(prior32 / 'release_verification.json')
    if base.read(prior32 / 'status.json')['status'] != 'complete':
        raise ValueError('The 32->1 reference must be complete')
    for group, digest in release32['predictions_sha256'].items():
        assert base.sha256(prior32 / group / 'predictions.csv.gz') == digest, group
    for name, digest in release32['reports_sha256'].items():
        assert base.sha256(prior32 / name) == digest, name
    return prior, release


def prepare(device="cuda"):
    prior, release = check_frozen_prior()
    master = base.ZeroShotConfig(**prior['config'])
    cfg = replace(master, context_length=16, device=device)
    plan = deepcopy(prior)
    plan.update(schema_version=2, config=asdict(cfg), master_window_config=asdict(master),
                contexts=[16], methods=list(METHODS), reference_context_length=64,
                reference_protocol_sha256=release['protocol_sha256'],
                reference_predictions_sha256=release['predictions_sha256'],
                matching='Exact 64-run target times, folds, label indices and label values. Input is the final 16 positions of each 64-position window; preserve NaN masks and original positions.',
                selection='Same validation-only loss and crossing guard as 64->1; step zero allowed. Fresh original C2 initialization for each fold/LR. Lock all 13 heads before test inference.',
                aggregation='Ordinary test R2 across one-step targets within each battery record, then equal-record mean; retain negative values and count undefined records. Report alongside MAE/RMSE in SOH ratio units and per-record LastValue-relative MSE skill.',
                bundle_root=str(BUNDLES),
                hardware_comparison='The prior 64/32 runs used XPU; this run records its device and package versions. Hardware/backend differences may affect numerical values and checkpoint selection.')
    sources = set(prior['source_sha256']) | {
        'scripts/run_c2_context16.py', 'scripts/report_c2_context16.py', 'src/c2_context_comparison.py',
        'src/c2_partial_tuning.py', 'src/battery_fewshot_tuning.py', 'src/c2_head_protocol.py'}
    plan['source_sha256'] = {p: base.sha256(ROOT/p) for p in sorted(sources)}
    prepared, coverage, budgets = {}, [], []
    for group, spec in plan['groups'].items():
        assert base.sha256(BUNDLES/group/'manifest.json') == spec['bundle_sha256']
        assert base.sha256(BUNDLES/group/'soh.csv') == spec['data_sha256']
        records, _, data, _ = base.prepare_reviewed_windows(BUNDLES/group, master)
        by_id = {r['battery_id']: r for r in records}
        source = {b for f in spec['folds'] for k in ('train_ids','validation_ids') for b in f[k]}
        test = {b for f in spec['folds'] for b in f['test_ids']}
        old = pd.read_csv(PRIOR/group/'predictions.csv.gz')
        old = old[old.method.eq('LastValue')]
        for b in sorted(source | test):
            w = cropped(by_id[b]['windows'], 16)
            if b in test:
                ref = old[old.battery_id.eq(b)].sort_values('target_cycle')
                np.testing.assert_array_equal(ref.target_cycle, w['target_cycle'][:,0])
                np.testing.assert_allclose(ref.target, w['target'][:,0], rtol=1e-12)
                np.testing.assert_allclose(ref.prediction, persistence(w['context'])['LastValue'], rtol=1e-12)
            coverage.append(dict(dataset=group,battery_id=b,role='held_out_test' if b in test else 'source_or_validation',
                                 context_length=16,n_targets=len(w['target']),
                                 min_context_observed=int(w['context_observed'].min()),
                                 targets_with_missing_context=int((w['context_observed']<16).sum())))
        for fold in spec['folds']:
            base.check_physical_split(fold, {b:r['physical_battery_id'] for b,r in by_id.items()})
            for label in fold['label_budget']:
                w = cropped(by_id[label['battery_id']]['windows'],16)
                i = label['index']
                assert int(w['target_cycle'][i,0]) == label['target_cycle']
                budgets.append(dict(**label,context_length=16,context_start=int(w['context_cycle'][i,0]),
                                    context_end=int(w['context_cycle'][i,-1]),
                                    observed_inputs=int(w['context_observed'][i])))
        assert len(old) == spec['test_windows']
        prepared[group] = dict(by_id=by_id,data=data)
        print(f"PLAN {group}: 16->1, {spec['train_windows']}/{spec['validation_windows']} train/val, {len(old)} matched test targets",flush=True)
    # Match the persisted JSON representation (dataclass seed tuples become lists).
    plan = json.loads(json.dumps(plan))
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists() and base.read(OUT/'protocol.json') != plan:
        raise ValueError('Locked protocol changed; use a new output directory')
    base.write_json(OUT/'protocol.json',plan)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    budget = pd.DataFrame(budgets)
    old_budget = pd.read_csv(PRIOR/'label_budget.csv')
    label_keys = ['dataset','fold_id','split','battery_id','index','target_cycle']
    pd.testing.assert_frame_equal(budget[label_keys].reset_index(drop=True),old_budget[label_keys].reset_index(drop=True))
    budget.to_csv(OUT/'label_budget.csv',index=False)
    runtime=OUT/'.runtime';runtime.mkdir(exist_ok=True)
    (runtime/'executed_runner.py').write_bytes(Path(__file__).read_bytes())
    return cfg,plan,prepared


def write_report(plan, summary):
    from scripts.report_c2_context16 import write_report as render
    render(OUT, plan, summary)


def verify(plan):
    check_frozen_prior()
    for path,digest in plan['source_sha256'].items():assert base.sha256(ROOT/path)==digest,path
    battery=pd.read_csv(OUT/'battery_metrics.csv')
    paired=pd.read_csv(OUT/'lastvalue_battery_metrics.csv')
    total=0
    for group,spec in plan['groups'].items():
        assert base.sha256(BUNDLES/group/'soh.csv')==spec['data_sha256']
        p=pd.read_csv(OUT/group/'predictions.csv.gz')
        assert set(p.method)==set(METHODS) and set(p.context_length)=={16}
        old=pd.read_csv(PRIOR/group/'predictions.csv.gz')
        old=old[old.method.eq('LastValue')].sort_values(['battery_id','target_cycle'])
        ref=p[p.method.eq('LastValue')].sort_values(['battery_id','target_cycle'])
        for col in ('battery_id','target_cycle','target','prediction','phase'):
            np.testing.assert_array_equal(ref[col].to_numpy(),old[col].to_numpy())
        records,_,_,_=base.prepare_reviewed_windows(BUNDLES/group,base.ZeroShotConfig(**plan['master_window_config']))
        for record in records:
            b=record['battery_id']
            if b not in set(ref.battery_id):continue
            w=record['windows'];x=w['context'][:,-16:]
            last=np.where(np.isfinite(x),np.arange(16),-1).max(axis=1)
            assert (last>=0).all()
            for method in METHODS:
                a=p[p.battery_id.eq(b)&p.method.eq(method)].sort_values('target_cycle')
                np.testing.assert_array_equal(a.target_cycle,w['target_cycle'][:,0])
                np.testing.assert_allclose(a.target,w['target'][:,0],rtol=1e-12)
                np.testing.assert_array_equal(a.context_observed,np.isfinite(x).sum(axis=1))
                np.testing.assert_array_equal(a.last_observed_lag,16-last)
                if method=='LastValue':np.testing.assert_allclose(a.prediction,x[np.arange(len(x)),last],rtol=1e-12)
                for phase in ('all','early','middle','late'):
                    part=a if phase=='all' else a[a.phase.eq(phase)]
                    if part.empty:continue
                    baseline=ref[ref.battery_id.eq(b)].set_index('target_cycle').loc[part.target_cycle]
                    e=part.prediction.to_numpy()-part.target.to_numpy()
                    be=baseline.prediction.to_numpy()-baseline.target.to_numpy()
                    s=battery[battery.dataset.eq(group)&battery.battery_id.eq(b)&battery.method.eq(method)&battery.phase.eq(phase)].iloc[0]
                    np.testing.assert_allclose([s.mae,s.rmse],[np.mean(abs(e)),np.sqrt(np.mean(e**2))],rtol=1e-9,atol=1e-12)
                    y=part.target.to_numpy()
                    sst=np.square(y-y.mean()).sum()
                    expected_r2=1-np.square(e).sum()/sst if len(y)>1 and sst>0 else np.nan
                    np.testing.assert_allclose(s.temporal_r2,expected_r2,rtol=1e-9,atol=1e-12,equal_nan=True)
                    r=paired[paired.dataset.eq(group)&paired.context_length.eq(16)&paired.battery_id.eq(b)&paired.method.eq(method)&paired.phase.eq(phase)].iloc[0]
                    expected=1-np.sum(e**2)/np.sum(be**2) if np.sum(be**2)>0 else np.nan
                    np.testing.assert_allclose(r.mse_skill_lastvalue,expected,rtol=1e-8,atol=1e-12)
        total+=len(ref)
    summary=pd.read_csv(OUT/'summary.csv').set_index(['dataset','context_length','method','phase']).sort_index()
    expected=battery.groupby(['dataset','context_length','method','phase'])[['mae','rmse','temporal_r2']].mean().sort_index()
    np.testing.assert_allclose(summary[expected.columns],expected,rtol=1e-10,atol=1e-12,equal_nan=True)
    comparison=pd.read_csv(OUT/'lastvalue_comparison.csv').set_index(['dataset','context_length','method','phase']).sort_index()
    expected=paired.groupby(['dataset','context_length','method','phase'])[['mse_skill_lastvalue','mae_skill_lastvalue','mse_skill_zero']].mean().sort_index()
    np.testing.assert_allclose(comparison[expected.columns],expected,rtol=1e-10,atol=1e-12,equal_nan=True)
    assert total==97868
    return dict(matched_targets_per_condition=total,conditions=3,context_length=16,prediction_length=1,
                identical_targets_to_context64=True,identical_lastvalue_to_context64=True,
                identical_source_labels_to_context64=True,reviewed_data_and_context_masks_verified=True,
                independent_metric_recalculation=True,linear_excluded=True,verified_at=base.now())


def runtime_info(device):
    torch = base.torch
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this desktop run. Activate the CUDA-enabled desktop environment; CPU/XPU fallback is disabled.')
    index = int(device.split(':')[1]) if ':' in device else 0
    info = dict(host=platform.node(), python=platform.python_version(),
                platform=platform.platform(), device=device, gpu=torch.cuda.get_device_name(index),
                cuda_runtime=torch.version.cuda, packages={})
    for package in ('torch','chronos-forecasting','transformers','numpy','pandas','accelerate','huggingface-hub'):
        try:
            info['packages'][package] = version(package)
        except PackageNotFoundError:
            info['packages'][package] = 'not installed'
    return info


def load_cuda_model(cfg, cache, download):
    from chronos import Chronos2Pipeline
    if download:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=cfg.model_id, revision=cfg.revision,
                          cache_dir=str(cache), allow_patterns=['*.json','*.safetensors','*.bin'])
    base.torch.backends.cuda.matmul.allow_tf32 = False
    base.torch.backends.cudnn.allow_tf32 = False
    pipeline = Chronos2Pipeline.from_pretrained(
        cfg.model_id, revision=cfg.revision, device_map=cfg.device,
        dtype=base.torch.float32, cache_dir=str(cache), local_files_only=True)
    if pipeline.model.device.type != 'cuda':
        raise RuntimeError('Model did not load on CUDA')
    pipeline.model.eval()
    return pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare','check','train','finish','report','all'], default='all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model-cache', type=Path, default=ROOT/'models/huggingface')
    parser.add_argument('--download-model', action='store_true', help='Fetch the pinned official amazon/chronos-2 revision if absent')
    args = parser.parse_args()
    configure()
    cfg, plan, prepared = prepare(args.device)
    if args.stage == 'prepare':
        print('PREPARED: same 97,868 targets; 16-position inputs; model training has NOT started.', flush=True)
        return
    if args.stage == 'report':
        base.summarize(plan)
        base.write_json(OUT/'verification.json', verify(plan))
        return
    state = base.read(OUT/'status.json') if (OUT/'status.json').exists() else dict(started_at=base.now())
    try:
        runtime = runtime_info(cfg.device)
        if (OUT/'runtime.json').exists() and base.read(OUT/'runtime.json') != runtime:
            raise RuntimeError('Runtime changed within a run. Keep device/environment fixed or use a fresh run directory.')
        base.write_json(OUT/'runtime.json', runtime)
        print('CUDA:', runtime['gpu'], '| torch:', runtime['packages']['torch'], flush=True)
        pipeline = load_cuda_model(cfg, args.model_cache.resolve(), args.download_model)
        group = next(iter(plan['groups']))
        battery = plan['groups'][group]['folds'][0]['test_ids'][0]
        x = cropped(prepared[group]['by_id'][battery]['windows'],16)['context'][:2]
        point, quantiles = base.predict_quantiles(pipeline,x,cfg)
        if not np.isfinite(point).all() or not np.isfinite(quantiles).all():
            raise RuntimeError('CUDA smoke inference returned nonfinite predictions')
        print('CUDA smoke inference passed:', point.shape, flush=True)
        if args.stage == 'check':
            return
        initial = {n:p.detach().cpu().clone() for n,p in pipeline.model.named_parameters() if n.startswith('output_patch_embedding.')}
        base.update(state,status='loaded',device=str(pipeline.model.device),protocol_sha256=base.sha256(OUT/'protocol.json'))
        if args.stage in ('train','all'):
            base.fit_all(pipeline,initial,plan,prepared,state)
        if args.stage in ('finish','all'):
            if 'all_heads_locked_at' not in state:
                raise ValueError('Lock all heads before testing')
            base.score(pipeline,cfg,initial,plan,prepared,state)
    except Exception as error:
        base.update(state,status='failed',error=repr(error))
        raise


if __name__ == '__main__':
    main()


