"""Deterministic tie-preserving groups for factor performance evaluation."""
from __future__ import annotations

import numpy as np

__all__ = ['GROUPING_VERSION', 'factor_groups']
GROUPING_VERSION = 'ties-minimum-1of50-v1'


def factor_groups(values, group_num=10, direction=1):
    """Return ordered 1-based groups; nonfinite observations remain NaN.

    Never split equal values. Merge groups smaller than ceil(N/50) into
    the adjacent group with closest mean (lower neighbour wins ties).
    """
    if group_num < 1 or direction not in (-1, 1):
        raise ValueError('Expected positive group count and direction +/-1')
    values = np.asarray(values, dtype=float) * direction
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    unique, inverse, counts = np.unique(values[valid], return_inverse=True, return_counts=True)
    if not len(unique):
        return result
    cumulative = counts.cumsum()
    boundaries = cumulative[:-1]
    cuts = sorted({int(np.argmin(abs(boundaries - cumulative[-1] * j / group_num))) + 1
                   for j in range(1, group_num)}) if len(boundaries) else []
    groups = [list(g) for g in np.split(np.arange(len(unique)), cuts)]
    minimum = int(np.ceil(cumulative[-1] / 50))
    while len(groups) > 1:
        small = next((i for i, g in enumerate(groups) if counts[g].sum() < minimum), None)
        if small is None:
            break
        means = [float(np.average(unique[g], weights=counts[g])) for g in groups]
        neighbours = [i for i in (small - 1, small + 1) if 0 <= i < len(groups)]
        small_mean = means[small]
        neighbour = min(neighbours, key=lambda i: (abs(means[i] - small_mean), i))
        lo, hi = sorted((small, neighbour))
        groups[lo] += groups[hi]
        del groups[hi]
    labels = np.empty(len(unique))
    for i, group in enumerate(groups, 1):
        labels[group] = i
    result[valid] = labels[inverse]
    return result
