"""Validation-fitted interval width around an unchanged LoRA median.

This is empirical calibration, not a conformal coverage guarantee: validation
windows also selected the model, and time points within a window are correlated.
"""
import numpy as np


def fit_scale(target, lora_median, foundation_quantiles, coverage=.8):
    y, m, q = map(lambda x: np.asarray(x, dtype=np.float64), (target, lora_median, foundation_quantiles))
    if y.shape != m.shape or q.shape != (*y.shape, 9):
        raise ValueError("Mismatched validation shapes")
    lower_width, upper_width = q[...,4]-q[...,0], q[...,-1]-q[...,4]
    if not (np.isfinite(y).all() and np.isfinite(m).all() and np.isfinite(q).all()):
        raise ValueError("Nonfinite validation values")
    width = np.where(y < m, lower_width, upper_width)
    if np.any(width <= 0):
        raise ValueError("Foundation interval needs positive half widths")
    required = abs(y-m)/width
    # A scalar from the original 10 validation windows only; no test labels.
    return max(1., float(np.quantile(required, coverage, method="higher")))


def apply_scale(lora_median, foundation_quantiles, scale):
    m, q = np.asarray(lora_median, dtype=np.float64), np.asarray(foundation_quantiles, dtype=np.float64)
    if q.shape != (*m.shape, 9) or not np.isfinite(scale) or scale < 1:
        raise ValueError("Invalid interval calibration")
    result = m[...,None]+float(scale)*(q-q[...,4:5])
    np.testing.assert_array_equal(result[...,4],m)
    return result
