import unittest
import numpy as np
from src.c2_rul_learning_curve import nested_indices, mit_split, cell_weights, constants, simple_features, fit_ridge, predict_ridge


class LearningCurveTests(unittest.TestCase):
    def test_nested_labels_and_short_cells(self):
        for n in (4, 9, 17, 321):
            a, b, c = [nested_indices(n, budget) for budget in ('4', '16', 'all')]
            self.assertTrue(set(a) <= set(b) <= set(c))
            self.assertEqual(len(b), min(n, 16))
            self.assertEqual(len(c), n)
            np.testing.assert_array_equal(a, np.linspace(0, n - 1, 4, dtype=int))

    def test_mit_roles_disjoint_and_order_independent(self):
        ids = [f'b2c{i}' for i in range(43)]
        a, b = mit_split(ids), mit_split(ids[::-1])
        self.assertEqual(a, b)
        self.assertEqual([len(a[k]) for k in ('train_pool', 'validation_ids', 'test_ids')], [20, 3, 20])
        self.assertEqual(a['train_pool'][:2], ['b2c0', 'b2c1'])
        self.assertEqual(len(set(a['train_pool'] + a['validation_ids'] + a['test_ids'])), 43)

    def test_long_cells_do_not_dominate_constants(self):
        ids = np.array(['a', 'a', 'a', 'b'])
        y = np.array([0., 0., 0., 10.])
        np.testing.assert_allclose(cell_weights(ids), [1/6, 1/6, 1/6, .5])
        self.assertAlmostEqual(constants(y, ids)['source_mean'], 5.)

    def test_features_keep_original_missing_positions(self):
        x = np.array([[1., np.nan, .8], [.9, np.nan, np.nan]])
        f = simple_features(x)
        self.assertAlmostEqual(f[0, 4], -.1)
        self.assertEqual(f[1, 4], 0.)
        self.assertAlmostEqual(f[0, -1], 2/3)

    def test_ridge_is_finite_with_constant_or_missing_features(self):
        x = np.array([[1., 1., 1.], [1., np.nan, 1.], [.9, .9, .9], [.9, np.nan, .9]])
        a = dict(context=x, target=np.array([20., 19., 5., 4.]), battery=np.array(['a', 'a', 'b', 'b']))
        state = fit_ridge(a, a)
        pred = predict_ridge(state, x)
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue((pred >= 0).all())
        self.assertLess(np.mean(abs(pred - a['target'])), 2.)


if __name__ == '__main__':
    unittest.main()
