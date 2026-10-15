"""Independent post-run audit of the locked NASA few-shot experiments."""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.battery_zeroshot import sha256
from src.c2_rul_context_nominal import load
from scripts.run_c2_nasa_fewshot_improvement import arrays

HEAD = ROOT / 'results/c2-battery/rul-nasa-fewshot-improvement-v1'
FULL = ROOT / 'results/c2-battery/rul-nasa-fewshot-full-v1'
OUT = ROOT / 'results/c2-battery/rul-nasa-fewshot-report-v1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def audit():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    cells, data_hashes, _ = load(ROOT)
    protocol = read(HEAD / 'protocol.json')
    locked = read(HEAD / 'selection_lock.json')
    assert locked['protocol_sha256'] == sha256(HEAD / 'protocol.json')
    assert protocol['data_artifact_sha256'] == data_hashes
    for name, digest in protocol['source_sha256'].items():
        assert sha256(ROOT / name) == digest, name
    cache = torch.load(HEAD / '.runtime/features.pt', weights_only=False, map_location='cpu')
    assert cache['protocol_sha256'] == locked['protocol_sha256']
    pred = pd.read_csv(HEAD / 'predictions.csv')
    assert sha256(HEAD / 'predictions.csv') == read(HEAD / 'verification.json')['predictions_sha256']
    assert len(pred) == 2097
    search = read(HEAD / 'validation_search.json')
    for s in search:
        assert sha256(HEAD / s['weights_path']) == s['weights_sha256']
        if s['config']['name'] != 'historical':
            history = pd.read_csv((HEAD / s['weights_path']).parent / 'history.csv')
            best = history.loc[history.validation_mae.idxmin()]
            assert int(best.step) == s['step']
            assert np.isclose(best.validation_mae, s['validation_mae'])

    def feats(a, length):
        return np.stack([cache['records'][f'{b}/{length}/{t}'] for b, t in zip(a['battery'], a['cycle'])])

    checks = []
    for m in locked['models']:
        assert sha256(HEAD / m['weights_path']) == m['weights_sha256']
        roles = [set(m[k]) for k in ['train_ids', 'validation_ids', 'test_ids']]
        assert [len(r) for r in roles] == [2, 1, 1]
        assert sum(len(r) for r in roles) == len(set.union(*roles))
        a = arrays(cells, m, m['train_ids'], '16')
        assert len(a['target']) == 32 and all(np.sum(a['battery'] == b) == 16 for b in roles[0])
        state = torch.load(HEAD / m['weights_path'], weights_only=False, map_location='cpu')
        x = feats(a, m['length']).astype(np.float64)
        # Independent fit: the budget is exactly 16 per cell, hence uniform weights.
        center, scale = np.zeros(771), np.ones(771)
        mode = state['scaler']['mode']
        indices = np.arange(771) if mode == 'all' else (np.arange(768, 771) if mode == 'metadata' else [])
        if len(indices):
            mean, std = x.mean(0), x.std(0)
            std[std < 1e-6] = 1
            center[indices], scale[indices] = mean[indices], std[indices]
        np.testing.assert_allclose(center.astype('float32'), state['scaler']['center'], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(scale.astype('float32'), state['scaler']['scale'], rtol=1e-6, atol=1e-6)
        assert state['scale'] == max(1, np.sort(a['target'])[15])
        assert np.isclose(state['constants']['source_mean'], a['target'].mean())
        head = nn.Sequential(nn.LayerNorm(771), nn.Linear(771, 64), nn.GELU(), nn.Linear(64, 1)).eval()
        head.load_state_dict(state['head'])
        assert sum(p.numel() for p in head.parameters()) == 51015

        def infer(a):
            z = (feats(a, m['length']) - state['scaler']['center']) / state['scaler']['scale']
            with torch.no_grad():
                return (torch.nn.functional.softplus(head(torch.tensor(z)).squeeze(-1)) * state['scale']).numpy()

        val = arrays(cells, m, m['validation_ids'])
        val_mae = float(np.mean(np.abs(infer(val) - val['target'])))
        assert np.isclose(val_mae, m['validation_mae'], rtol=2e-4, atol=.01)
        test = arrays(cells, m, m['test_ids'])
        actual = pred[(pred.method == m['method']) & (pred.seed == m['seed']) & pred.battery_id.isin(m['test_ids'])].sort_values('origin')
        np.testing.assert_array_equal(actual.origin, test['cycle'])
        np.testing.assert_array_equal(actual.true_rul, test['target'])
        for b, t, context, y in zip(test['battery'], test['cycle'], test['context'], test['target']):
            assert t >= 32 and t < cells['nasa_aging', b]['eol']
            np.testing.assert_allclose(context, cells['nasa_aging', b]['values'][t-m['length']:t], equal_nan=True)
            assert y == cells['nasa_aging', b]['eol'] - t
        delta = float(np.max(abs(actual.predicted_rul.to_numpy() - infer(test))))
        assert delta < .01, (m['key'], delta)
        if m['seed'] == 42 and m['method'] != 'historical_L32':
            eligible = [s for s in search if s['fold'] == m['fold'] and (m['method'] == 'tuned_context' or s['length'] == 32)]
            assert m['key'] == min(eligible, key=lambda s: s['validation_mae'])['key']
        checks.append(dict(method=m['method'], fold=m['fold'], seed=m['seed'], length=m['length'], train_labels=32,
                           validation_labels=len(val['target']), test_origins=len(test['target']),
                           validation_mae_reloaded=val_mae, max_prediction_difference=delta))

    old = pd.read_csv(ROOT / 'results/c2-battery/rul-context-nominal80-v1/predictions.csv')
    old = old[(old.dataset == 'nasa_aging') & (old.method == 'head') & (old.length == 32)].copy()
    current = pred[(pred.method == 'historical_L32') & pred.original_stride16].copy()
    merged = current.merge(old, left_on=['seed', 'battery_id', 'origin'], right_on=['seed', 'battery_id', 'origin_cycle'], validate='1:1', suffixes=('_new', '_old'))
    assert len(merged) == len(current) == len(old) == 48
    reference_delta = float(max(abs(merged.predicted_rul_new - merged.predicted_rul_old)))
    assert reference_delta < .01
    np.testing.assert_array_equal(merged.true_rul_new, merged.true_rul_old)

    # Verify full-model training records and remote reload evidence, not a claim of local weight reload.
    fp = read(FULL / 'protocol.json')
    fl = read(FULL / 'selection_lock.json')
    fv = read(FULL / 'verification.json')
    full = pd.read_csv(FULL / 'predictions.csv')
    assert fp['head_protocol_sha256'] == sha256(HEAD / 'protocol.json')
    assert fp['head_selection_lock_sha256'] == sha256(HEAD / 'selection_lock.json')
    assert fl['protocol_sha256'] == sha256(FULL / 'protocol.json')
    for name, digest in fp['source_sha256'].items():
        assert sha256(ROOT / name) == digest, name
    assert fv['status'] == 'passed' and len(full) == 699 and len(fl['models']) == 12
    assert sha256(FULL / 'predictions.csv') == fv['predictions_sha256']
    remote = read(FULL / 'remote_weights_inventory.json')
    remote = {r['case']: r for r in remote}
    full_checks = []
    for m in fl['models']:
        history = pd.read_csv(FULL / 'checkpoints' / m['case'] / 'history.csv')
        best = history.loc[history.validation_mae.idxmin()]
        assert best.step == m['step'] and np.isclose(best.validation_mae, m['validation_mae'])
        assert m['validation_mae'] <= m['baseline_validation_mae'] + .01
        assert all(all(g[k] for k in ['head_grad', 'backbone_grad', 'unused_output_frozen']) for g in m['gradient_audit'])
        assert m['train_labels'] == 32 and not m['architecture_changed']
        if m['weights_path']:
            assert remote[m['case']]['sha256'] == m['weights_sha256'] and remote[m['case']]['bytes'] > 100000000
        else:
            assert m['step'] == 0
            ref = m['reference']
            a = full[(full.seed == ref['seed']) & full.battery_id.isin(ref['test_ids'])].sort_values('origin')
            b = pred[(pred.method == 'tuned_L32') & (pred.seed == ref['seed']) & pred.battery_id.isin(ref['test_ids'])].sort_values('origin')
            np.testing.assert_allclose(a.predicted_rul, b.predicted_rul, rtol=2e-4, atol=.01)
        full_checks.append(dict(case=m['case'], selected_step=m['step'], baseline_validation_mae=m['baseline_validation_mae'], validation_mae=m['validation_mae']))
    combined = pd.concat([pred, full], ignore_index=True)
    assert not combined.duplicated(['method', 'seed', 'battery_id', 'origin']).any()
    assert np.isfinite(combined.predicted_rul).all() and (combined.predicted_rul >= 0).all()
    common = None
    for _, a in combined.groupby(['method', 'seed']):
        keys = set(zip(a.battery_id, a.origin, a.true_rul))
        if common is None:
            common = keys
        assert keys == common and len(keys) == 233
    combined.to_csv(OUT / 'predictions.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT / 'head_reload_checks.csv', index=False)
    pd.DataFrame(full_checks).to_csv(OUT / 'full_selection_checks.csv', index=False)
    audit = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                 data_hashes_match=True, source_hashes_match=True, unchanged_head_parameters=51015,
                 source_validation_test_disjoint=True, source_labels_per_model=32, validation_cell_labels_additional=True,
                 scaler_independently_refitted_on_training_only=True, selected_head_reload_models=len(checks),
                 max_head_prediction_difference=max(r['max_prediction_difference'] for r in checks),
                 old_stride16_prediction_max_difference=reference_delta, original_stride16_origins=16, dense_origins=233,
                 full_models=12, full_step_zero_fallbacks=sum(m['step'] == 0 for m in fl['models']),
                 full_weight_reload='Verified on home GPU by training runner; local metadata and hashes checked. Full weights remain on home GPU.',
                 full_weights_inventory_sha256=sha256(FULL / 'remote_weights_inventory.json'),
                 note='Historical test cells have been viewed before; exploratory follow-up, not an untouched external validation.')
    (OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    audit()
