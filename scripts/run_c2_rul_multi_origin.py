"""Matched t64 evaluation: single versus three source origins, fresh XPU heads."""
import argparse
from pathlib import Path
import sys
import time
import platform
from importlib.metadata import version
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts import run_c2_rul_head as engine
from scripts.run_c2_rul_pilot import sha, write_json, now, REVISION, QUANTILES, forecasts_to_frame
from scripts.report_c2_rul_context_sweep import trajectory_diagnostics
from src.c2_rul import make_tasks, eligible_mask, score_trajectory, summarize_rul
from src.c2_rul_head import cache_features, validate_cache_equivalence, source_trajectories
from src.c2_one_step import restore_head
from src.c2_partial_tuning import configure_subset
from src.soh_data import load_bundle

OUT = ROOT/'results/c2-battery/rul-nasa-multi-origin-xpu-v1'
PRIOR = ROOT/'results/c2-battery/rul-nasa-l64-h256-head-v1'
CONDITIONS = {'single': (64,), 'multi': (64, 80, 96)}


def build_source(data, metadata, origins):
    """Origin admission uses only past eligibility and observed past EOL events."""
    _, candidates = make_tasks(data, metadata, context_length=64)
    source = {}
    for t in candidates:
        b, origin = t['battery_id'], t['origin_cycle']
        if origin not in origins:
            continue
        raw = data[data.battery_id.eq(b)].sort_values('cycle').set_index('cycle')
        y = raw.soh.where(eligible_mask(raw)).reindex(range(origin+1, origin+257)).to_numpy(dtype=np.float32)
        if not np.isfinite(y).any():
            raise ValueError('A requested source origin has no observed target')
        source[f'{b}@{origin}'] = dict(battery_id=b, origin_cycle=origin,
            context=np.asarray(t['context'], dtype=np.float32), target=y)
    # Equal window counts preserve equal battery weights under the existing mean loss.
    # Abort instead of silently changing weighting if another dataset lacks an origin.
    for b in sorted(data.battery_id.unique()):
        assert {r['origin_cycle'] for r in source.values() if r['battery_id']==b} == set(origins), b
    return source


def expand_fold(fold, source):
    expanded = dict(fold)
    expanded['train_ids'] = [k for k, r in source.items() if r['battery_id'] in fold['train_ids']]
    expanded['validation_ids'] = [f'{b}@64' for b in fold['validation_ids']]
    assert all(source[k]['battery_id'] not in fold['test_ids']+fold['validation_ids'] for k in expanded['train_ids'])
    assert all(source[k]['battery_id'] not in fold['test_ids'] for k in expanded['validation_ids'])
    return expanded


def prepare():
    prior = engine.read(PRIOR/'protocol.json')
    assert engine.read(PRIOR/'status.json')['status']=='complete'
    for name, digest in prior['source_sha256'].items():
        assert sha(ROOT/name)==digest, name
    for name, digest in prior['data_sha256'].items():
        assert sha(engine.BUNDLE/name)==digest, name
    data, meta, _ = load_bundle(engine.BUNDLE)
    audit, tasks = make_tasks(data, meta)
    tasks = [t for t in tasks if t['origin_cycle']==64]
    assert len(tasks)==4
    sources = {c: build_source(data, meta, origins) for c, origins in CONDITIONS.items()}
    old_source = source_trajectories(data)
    for b, r in old_source.items():
        for key in ['context', 'target']:
            np.testing.assert_array_equal(r[key], sources['single'][f'{b}@64'][key])
            np.testing.assert_array_equal(r[key], sources['multi'][f'{b}@64'][key])
    labels, budgets = [], []
    for condition, source in sources.items():
        for fold in prior['folds']:
            engine.check_physical_split(fold, {b:b for b in old_source})
            expanded = expand_fold(fold, source)
            for split, key in [('train','train_ids'), ('validation','validation_ids')]:
                unique = set()
                exposures = 0
                for k in expanded[key]:
                    r = source[k]
                    cycles = (np.flatnonzero(np.isfinite(r['target']))+r['origin_cycle']+1).tolist()
                    labels.append(dict(condition=condition, fold_id=fold['fold_id'], split=split,
                        battery_id=r['battery_id'], origin_cycle=r['origin_cycle'],
                        context_start=r['origin_cycle']-63, context_end=r['origin_cycle'],
                        n_labels=len(cycles), label_cycles=cycles))
                    unique.update((r['battery_id'], cycle) for cycle in cycles)
                    exposures += len(cycles)
                budgets.append(dict(condition=condition, fold_id=fold['fold_id'], split=split,
                    n_windows=len(expanded[key]), unique_soh_labels=len(unique),
                    label_exposures_per_full_batch=exposures,
                    repeated_label_exposures_per_full_batch=exposures-len(unique)))
    files = sorted(set(engine.SOURCES)|{'scripts/run_c2_rul_multi_origin.py',
        'scripts/report_c2_rul_multi_origin.py','scripts/report_c2_rul_context_sweep.py'})
    plan = dict(schema_version=1, model_id=prior['model_id'], revision=REVISION,
        context_length=64, prediction_length=256, primary_origin=64, seed=42,
        learning_rates=[1e-6,1e-5], steps_per_lr=100, validation_every=20,
        conditions={k:list(v) for k,v in CONDITIONS.items()}, folds=prior['folds'],
        selection=prior['selection'].replace('four heads','eight heads'),
        loss='Existing native normalized pinball loss: mean over valid labels per window, mean over windows. Exactly three origins per cell gives equal cell weight in multi; single has one per cell.',
        control='Both conditions freshly trained on same XPU runtime, same pretrained initialization. Validation and test origins fixed t64. Validation has one disjoint cell; training two; test one.',
        source_admission='Fixed origins 64/80/96; eligible current observation and no observed past EOL. No future lifetime fraction used. Equal origin count required.',
        label_policy=prior['labels'], eol_definition=prior['eol_definition'], censoring=prior['censoring'],
        label_budget=labels, budget_summary=budgets,
        limitation='Exploratory NASA four-cell single-seed study with three observed events; prior test results motivated this intervention. Overlapping windows repeat labels, not new independent batteries or unique future labels. No future measured input, smoothing or monotonic projection.',
        data_sha256=prior['data_sha256'], source_sha256={p:sha(ROOT/p) for p in files},
        reference_protocol_sha256=sha(PRIOR/'protocol.json'),
        tasks=[{k:v for k,v in t.items() if k!='context'} for t in tasks])
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        assert engine.read(OUT/'protocol.json')==plan, 'Locked protocol changed'
    write_json(OUT/'protocol.json', plan)
    pd.DataFrame(labels).to_csv(OUT/'label_windows.csv', index=False)
    pd.DataFrame(budgets).to_csv(OUT/'label_budget.csv', index=False)
    audit.to_csv(OUT/'eol_audit.csv', index=False)
    return plan, data, tasks, sources


def verify(plan, tasks, selected):
    for p, digest in plan['source_sha256'].items(): assert sha(ROOT/p)==digest, p
    for p, digest in plan['data_sha256'].items(): assert sha(engine.BUNDLE/p)==digest, p
    assert sha(PRIOR/'protocol.json')==plan['reference_protocol_sha256']
    locked = engine.read(OUT/'heads_locked.json')
    assert len(locked['selected'])==8
    assert locked['locked_at']<=engine.read(OUT/'status.json')['test_started_at']
    scores = pd.read_csv(OUT/'rul_predictions.csv')
    summary = pd.read_csv(OUT/'summary.csv')
    assert len(scores)==12
    with np.load(OUT/'forecast_quantiles.npz') as saved:
        np.testing.assert_array_equal(saved['context'], np.stack([t['context'] for t in tasks]))
        for method in ['zero-shot','single','multi']:
            q = saved[method]
            assert q.shape==(4,256,9) and np.isfinite(q).all()
            recomputed = pd.DataFrame([dict(method=method, **score_trajectory(t,v[:,4])) for t,v in zip(tasks,q)])
            observed = scores[scores.method.eq(method)].reset_index(drop=True)
            for k in ['predicted_rul','error_cycles','absolute_error_lower_bound_cycles','contradicts_censored_followup']:
                np.testing.assert_allclose(observed[k].astype(float),recomputed[k].astype(float),rtol=1e-10,atol=1e-12,equal_nan=True)
            for k,v in summarize_rul(recomputed).items():
                np.testing.assert_allclose(summary[summary.method.eq(method)].iloc[0][k],np.nan if v is None else v,rtol=1e-10,atol=1e-12,equal_nan=True)
        for condition, records in selected.items():
            for fold, s in zip(plan['folds'], records):
                assert sha(ROOT/s['weights_file'])==s['weights_sha256']
                assert s['frozen_backbone_verified'] and s['selected_head_reload_verified'] and s['masked_loss_native_parity']
                h = pd.read_csv(OUT/condition/fold['fold_id']/'validation_history.csv')
                best = h[h.eligible].sort_values(['validation_loss','step','learning_rate']).iloc[0]
                assert s['step']==int(best.step) and s['learning_rate']==best.learning_rate
                if s['step']==0:
                    i = [t['battery_id'] for t in tasks].index(fold['test_ids'][0])
                    np.testing.assert_allclose(saved[condition][i],saved['zero-shot'][i],rtol=2e-5,atol=1e-5)
    return dict(status='passed', verified_at=now(), heads=8, forecasts=12,
        same_runtime_control=True, physical_battery_splits_checked=True,
        saved_rul_and_summary_recomputed=True, source_data_hashes_checked=True,
        weights_hash_and_reload_checked=True, validation_selection_recomputed=True,
        heads_locked_before_test=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if (OUT/'status.json').exists():
        raise ValueError('Existing execution is immutable; use a new output directory')
    plan, data, tasks, sources = prepare()
    if args.prepare_only:return
    torch=engine.torch
    assert torch.xpu.is_available(), 'XPU unavailable'
    state=dict(status='loading',started_at=now(),protocol_sha256=sha(OUT/'protocol.json'))
    write_json(OUT/'status.json',state);started=time.perf_counter()
    try:
        write_json(OUT/'runtime.json',dict(host=platform.node(),python=platform.python_version(),device='xpu',
            device_name=torch.xpu.get_device_name(0),packages={p:version(p) for p in ['torch','chronos-forecasting','transformers','numpy','pandas']}))
        torch.manual_seed(42);np.random.seed(42)
        pipeline=engine.Chronos2Pipeline.from_pretrained(plan['model_id'],revision=REVISION,device_map='xpu',
            dtype=torch.float32,cache_dir=str(ROOT/'models/huggingface'),local_files_only=True)
        model=pipeline.model;model.eval();assert model.device.type=='xpu'
        names=configure_subset(model,'head',training=False)
        initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names}
        frozen={n:p._version for n,p in model.named_parameters() if n not in names}
        selected={};parity={}
        for condition, source in sources.items():
            restore_head(model,initial);ids=sorted(source)
            cache=cache_features(model,np.stack([source[k]['context'] for k in ids]),np.stack([source[k]['target'] for k in ids]))
            parity[condition]=validate_cache_equivalence(model,cache)
            engine.OUT=OUT/condition;engine.OUT.mkdir(exist_ok=True)
            engine.WEIGHTS=ROOT/f'models/c2-rul-multi-origin-xpu-v1/{condition}'
            selected[condition]=[]
            for fold in plan['folds']:
                state.update(status='training',condition=condition,fold=fold['fold_id']);write_json(OUT/'status.json',state)
                selected[condition].append(engine.train_fold(model,initial,cache,ids,expand_fold(fold,source),plan))
            del cache
        write_json(OUT/'cache_parity.json',parity)
        write_json(OUT/'heads_locked.json',dict(locked_at=now(),selected=[dict(condition=c,**s) for c,ss in selected.items() for s in ss]))
        state.update(status='test_inference',test_started_at=now());write_json(OUT/'status.json',state)
        x=np.stack([t['context'] for t in tasks]);restore_head(model,initial)
        arrays={'context':x,'zero-shot':engine.predict(pipeline,x)}
        for condition, records in selected.items():
            values=np.empty_like(arrays['zero-shot'])
            for fold,s in zip(plan['folds'],records):
                restore_head(model,torch.load(ROOT/s['weights_file'],map_location='cpu',weights_only=True))
                i=[t['battery_id'] for t in tasks].index(fold['test_ids'][0])
                values[i]=engine.predict(pipeline,x[i:i+1])[0]
            arrays[condition]=values
        torch.xpu.synchronize();state['compute_elapsed_seconds']=time.perf_counter()-started
        np.savez_compressed(OUT/'forecast_quantiles.npz',**arrays)
        from scripts.report_c2_rul_multi_origin import build_report
        build_report(OUT,plan,data,tasks,arrays)
        params=dict(model.named_parameters());assert all(params[n]._version==v and params[n].grad is None for n,v in frozen.items())
        verification=verify(plan,tasks,selected)
        verification['artifacts_sha256']={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['verification.json','status.json']}
        write_json(OUT/'verification.json',verification)
        state.update(status='complete',finished_at=now(),total_elapsed_seconds=time.perf_counter()-started)
        write_json(OUT/'status.json',state);print('COMPLETE',state,flush=True)
    except Exception as e:
        state.update(status='failed',error=repr(e));write_json(OUT/'status.json',state);raise

if __name__=='__main__':main()
