"""NASA paired 2x2 scaling/loss ablation over every disjoint 2/1/1 partition."""
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src.battery_zeroshot import sha256
from src.c2_rul_context_nominal import load
from src.c2_rul_improvement import transform, new_head, output
from scripts.run_c2_nasa_fewshot_improvement import fit, arrays, read, write, status

HEAD = ROOT / 'results/c2-battery/rul-nasa-fewshot-improvement-v1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    cells, hashes, _ = load(ROOT)
    assert hashes == read(HEAD / 'protocol.json')['data_artifact_sha256']
    ids = sorted(b for g, b in cells if g == 'nasa_aging')
    assert ids == ['B0005', 'B0006', 'B0007', 'B0018']
    cases = [dict(fold=f'test_{test}__val_{val}', length=32, train_ids=[b for b in ids if b not in [test, val]],
                  validation_ids=[val], test_ids=[test]) for test in ids for val in ids if val != test]
    grid = [dict(name=f'{norm}_{loss}', normalization=norm, loss=loss, weight_decay=0.)
            for norm in ['none', 'metadata'] for loss in ['huber1', 'mae']]
    protocol = dict(task='NASA fixed-L32 scaling x loss factorial, all 12 disjoint partitions',
                    cases=cases, grid=grid, seeds=[42, 43, 44], lrs=[.001, .003], steps=1000,
                    batch_size=16, eval_every=50, source_labels=32,
                    source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in [Path(__file__), ROOT/'src/c2_rul_improvement.py', ROOT/'scripts/run_c2_nasa_fewshot_improvement.py']},
                    data_artifact_sha256=hashes, frozen_feature_protocol_sha256=sha256(HEAD/'protocol.json'),
                    frozen_features_sha256=sha256(HEAD/'.runtime/features.pt'),
                    comparison='Four separately reported methods; same pretrained frozen encoder and head architecture; 2 source cells x16 labels; identical sampled minibatches per seed. Only scaler and loss differ.',
                    selection='Each method independently chooses LR and checkpoint on the validation cell, for every seed/partition. All 144 selections locked before test inference. No test-based winner selection.',
                    limitation='Follow-up motivated by prior NASA results; reused historical cells, not external confirmation. Three partitions per test cell are correlated, not 12 independent test cells. Validation uses all eligible labels in its separate cell, beyond the 32 training examples.')
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json') == protocol
    else:
        write(out/'protocol.json', protocol)
    if args.prepare_only:
        status(out, 'prepared', fits=288, selected_models=144)
        return
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    record = torch.load(HEAD/'.runtime/features.pt', weights_only=False, map_location='cpu')
    assert record['protocol_sha256'] == protocol['frozen_feature_protocol_sha256']
    def get(a, length):
        return np.stack([record['records'][f'{b}/{length}/{int(t)}'] for b, t in zip(a['battery'], a['cycle'])])
    cache = SimpleNamespace(device='cuda', get=get)
    all_results = []
    selections = []
    if (out/'selection_lock.json').exists():
        selections = read(out/'selection_lock.json')['models']
    else:
        for c in cases:
            for config in grid:
                for seed in protocol['seeds']:
                    candidates = [fit(c, config, lr, seed, cells, cache, out, protocol) for lr in protocol['lrs']]
                    all_results.extend(candidates)
                    best = min(candidates, key=lambda r: r['validation_mae'])
                    selections.append(dict(**best, method=config['name'], train_ids=c['train_ids'], validation_ids=c['validation_ids'], test_ids=c['test_ids']))
                status(out, 'factorial_search', fits=len(all_results), total=288, fold=c['fold'], method=config['name'])
        write(out/'validation_search.json', all_results)
        write(out/'selection_lock.json', dict(models=selections, protocol_sha256=sha256(out/'protocol.json'), locked_at=datetime.now(timezone.utc).isoformat()))
    assert len(selections) == 144
    rows = []
    reload_checks = []
    for m in selections:
        state_path = out/m['weights_path']
        assert sha256(state_path) == m['weights_sha256']
        state = torch.load(state_path, weights_only=False, map_location='cpu')
        head = new_head(771, state['scale'], state['constants']['source_mean'], m['seed'], 'cuda')
        head.load_state_dict(state['head'])
        def predict(a):
            f = transform(get(a, 32), state['scaler'])
            with torch.no_grad():
                return output(head, torch.tensor(f, device='cuda'), state['scale']).cpu().numpy()
        v = arrays(cells, m, m['validation_ids'])
        vmae = float(np.mean(abs(predict(v) - v['target'])))
        assert np.isclose(vmae, m['validation_mae'], rtol=2e-4, atol=.01)
        reload_checks.append(dict(key=m['key'], validation_mae=vmae))
        a = arrays(cells, m, m['test_ids'])
        p = predict(a)
        assert np.isfinite(p).all()
        for b, t, y, yp in zip(a['battery'], a['cycle'], a['target'], p):
            assert b not in m['train_ids'] + m['validation_ids']
            rows.append(dict(method=m['method'], fold=m['fold'], validation_id=m['validation_ids'][0], seed=m['seed'],
                             battery_id=b, origin=int(t), true_rul=float(y), predicted_rul=float(yp), length=32,
                             original_stride16=(t-32)%16 == 0, life_fraction=float(t/cells['nasa_aging', b]['eol'])))
    pd.DataFrame(rows).to_csv(out/'predictions.csv', index=False)
    write(out/'verification.json', dict(status='passed', models=144, architecture_changed=False,
          source_cells=2, labels_per_source_cell=16, validation_labels_additional=True,
          source_test_disjoint=True, selection_locked_before_test=True, reload_checks=reload_checks,
          predictions_sha256=sha256(out/'predictions.csv')))
    status(out, 'completed', models=144, predictions=len(rows))


if __name__ == '__main__':
    main()
