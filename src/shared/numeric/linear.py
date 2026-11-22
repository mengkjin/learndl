"""Pure NumPy weights and per-component contributions for linear combinations."""
from __future__ import annotations

import numpy as np


def linear_weights(value, size: int, *, normalize: bool = False) -> np.ndarray:
    """Equal means 1/N; explicit coefficients are literal unless normalize is set."""
    if size < 1:
        raise ValueError('Linear inputs must not be empty')
    if value is None or (isinstance(value, str) and value == 'equal'):
        return np.full(size, 1.0 / size)
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError('Linear weights must be equal or a numeric vector')
    raw = np.asarray(value, dtype=object)
    if raw.shape != (size,) or any(isinstance(v, (bool, np.bool_, str)) for v in raw):
        raise ValueError(f'Linear weights must contain exactly {size} finite numbers')
    weights = np.asarray(value, dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError('Linear weights must be finite numbers')
    if normalize:
        if weights.sum() == 0:
            raise ValueError('Normalized linear weights must have a nonzero sum')
        weights = weights / weights.sum()
    return weights


def weighted_components(values: np.ndarray, weights, *, axis: int = -1,
                        normalize_weights: bool = False) -> np.ndarray:
    """Keep component columns; callers own alignment, missing values and reduction."""
    axis %= values.ndim
    coefficients = linear_weights(weights, values.shape[axis], normalize=normalize_weights)
    shape = [1] * values.ndim
    shape[axis] = len(coefficients)
    return values * coefficients.reshape(shape)
