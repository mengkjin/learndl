"""Tie-preserving grouping invariants without project database initialization."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np

__all__ = []

spec = importlib.util.spec_from_file_location('factor_groups', Path(__file__).resolve().parents[1] / 'src/func/factor_groups.py')
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_ties_minimum_and_permutation():
    values = np.r_[np.zeros(70), np.ones(29), 100.]
    groups = module.factor_groups(values)
    assert len(set(groups)) == 2
    assert groups[-1] == groups[-2]
    assert np.unique(groups, return_counts=True)[1].min() >= 2
    order = np.random.default_rng(17).permutation(len(values))
    np.testing.assert_array_equal(module.factor_groups(values[order]), groups[order])


def test_continuous_direction_and_missing():
    values = np.r_[np.arange(100.), np.nan, np.inf]
    groups = module.factor_groups(values)
    assert np.isnan(groups[-2:]).all()
    assert len(set(groups[:-2])) == 10
    np.testing.assert_array_equal(module.factor_groups(values, direction=-1)[:-2], 11 - groups[:-2])


def test_constant():
    np.testing.assert_array_equal(module.factor_groups(np.ones(100)), np.ones(100))
    assert np.isnan(module.factor_groups([np.nan])).all()
