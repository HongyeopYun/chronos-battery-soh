"""Source-data learning curves; splitting never ranks cells by observed lifetime."""
import numpy as np

from src.c2_v4_experiments import rul_windows, small_folds
from src.c2_head_protocol import natural_key

SEEDS = (42, 43, 44)
BUDGETS = ('4', '16', 'all')


def nested_indices(n, budget):
    if n < 4:
        raise ValueError('At least four distinct source origins are required')
    chosen = set(np.linspace(0, n - 1, 4, dtype=int).tolist())
    count = n if budget == 'all' else min(n, int(budget))
    if count < 4:
        raise ValueError('Budget must be at least four')
    # Keep original four labels and fill the largest gaps deterministically.
    while len(chosen) < count:
        distances = np.min(abs(np.arange(n)[:, None] - np.array(sorted(chosen))[None, :]), axis=1)
        chosen.add(int(np.argmax(distances)))
    return np.array(sorted(chosen), dtype=int)


def mit_split(ids):
    ids = sorted(ids, key=natural_key)
    if len(ids) != 43:
        raise ValueError('This protocol expects 43 MIT Batch 2 event cells')
    # Preserve the previous two source cells and validation cell; choose the
    # new common test by IDs and a fixed RNG, not EOL or previous errors.
    shuffled = np.random.default_rng(20261013).permutation(ids[3:]).tolist()
    return dict(train_pool=ids[:2] + shuffled[22:], validation_ids=ids[2:3] + shuffled[20:22],
                test_ids=sorted(shuffled[:20], key=natural_key))


def make_cases(cells):
    cases = []
    for group in ('nasa_aging', 'calce_cs2_1c', 'mit_batch2', 'xjtu_2c'):
        ids = [b for (g, b), c in cells.items() if g == group and c['eol'] is not None
               and len(rul_windows(c, 1)['cycle']) >= 4]
        if group == 'mit_batch2':
            split = mit_split(ids)
            for seed in SEEDS:
                pool = split['train_pool'][:2] + np.random.default_rng(seed).permutation(split['train_pool'][2:]).tolist()
                for count in (2, 4, 8, 16):
                    for budget in BUDGETS:
                        cases.append(dict(dataset=group, fold='fixed20', seed=seed, n_train_cells=count,
                            budget=budget, train_ids=pool[:count], validation_ids=split['validation_ids'],
                            test_ids=split['test_ids']))
        else:
            if len(ids) != 4:
                raise ValueError(f'Expected four event cells in {group}')
            for fold in small_folds(ids):
                for seed in SEEDS:
                    for budget in BUDGETS:
                        cases.append(dict(dataset=group, fold=fold['fold_id'], seed=seed, n_train_cells=2,
                            budget=budget, **{k: fold[k] for k in ('train_ids', 'validation_ids', 'test_ids')}))
    for case in cases:
        roles = [set(case[k]) for k in ('train_ids', 'validation_ids', 'test_ids')]
        assert not any(roles[i] & roles[j] for i, j in ((0, 1), (0, 2), (1, 2)))
        case['case_id'] = f"{case['dataset']}__{case['fold']}__s{case['seed']}__n{case['n_train_cells']}__b{case['budget']}"
    assert len(cases) == 144
    return cases


def assemble(cells, group, ids, budget='all', stride=1):
    pieces = []
    for battery in ids:
        w = rul_windows(cells[group, battery], stride)
        idx = np.arange(len(w['cycle'])) if budget == 'all' else nested_indices(len(w['cycle']), budget)
        pieces.append(dict(context=w['context'][idx], target=w['target'][idx], cycle=w['cycle'][idx],
                           battery=np.repeat(battery, len(idx))))
    return {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}


def cell_weights(ids):
    ids = np.asarray(ids)
    return np.array([1 / np.sum(ids == b) for b in ids], dtype=float) / len(np.unique(ids))


def constants(y, ids):
    w = cell_weights(ids)
    order = np.argsort(y)
    return dict(source_mean=float(np.sum(y * w)),
                source_median=float(y[order[np.searchsorted(np.cumsum(w[order]), .5)]]))


def simple_features(x):
    """Same SOH-only input as C2, without origin cycle or future information."""
    rows = []
    for row in x:
        pos = np.flatnonzero(np.isfinite(row))
        y = row[pos].astype(float)
        if not len(y):
            raise ValueError('All missing history')
        slope = np.polyfit(pos, y, 1)[0] if len(y) > 1 else 0.
        rows.append([y[-1], y[0], y.mean(), y.std(), slope, len(y) / len(row)])
    return np.asarray(rows)


def fit_ridge(train, validation):
    x, v = simple_features(train['context']), simple_features(validation['context'])
    w = cell_weights(train['battery'])
    center = np.sum(w[:, None] * x, axis=0)
    scale = np.sqrt(np.sum(w[:, None] * (x - center) ** 2, axis=0))
    scale[scale < 1e-8] = 1.
    a = np.column_stack([np.ones(len(x)), (x - center) / scale])
    b = np.column_stack([np.ones(len(v)), (v - center) / scale])
    best = None
    for alpha in (0.001, 0.01, 0.1, 1., 10.):
        penalty = np.eye(a.shape[1]) * alpha
        penalty[0, 0] = 0.
        coef = np.linalg.solve(a.T @ (w[:, None] * a) + penalty, a.T @ (w * train['target']))
        pred = np.maximum(0., b @ coef)
        score = float(np.sum(cell_weights(validation['battery']) * abs(pred - validation['target'])))
        candidate = dict(alpha=alpha, center=center.tolist(), scale=scale.tolist(), coef=coef.tolist(), validation_mae=score)
        if best is None or score < best['validation_mae']:
            best = candidate
    return best


def predict_ridge(state, x):
    a = np.column_stack([np.ones(len(x)), (simple_features(x) - state['center']) / state['scale']])
    return np.maximum(0., a @ state['coef'])
