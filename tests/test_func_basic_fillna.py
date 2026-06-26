"""Regression tests for forward_fillna loop/vector paths and inplace."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from src.func.basic import backward_fillna, forward_fillna, forward_fillna_np
from tests.helpers.legacy_fillna import legacy_fillna_torch


def _random_nan(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    arr = rng.standard_normal(shape)
    arr[rng.random(shape) < 0.35] = np.nan
    return arr


class TestForwardFillna(unittest.TestCase):
    SHAPES = ((12,), (5, 8), (4, 7, 3), (3, 6, 2, 5), (8, 20, 1, 10))

    def test_auto_loop_matches_reference(self) -> None:
        rng = np.random.default_rng(0)
        for shape in self.SHAPES:
            raw = _random_nan(rng, shape)
            t = torch.tensor(raw, dtype=torch.float64)
            for method in ('auto', 'loop', 'vector'):
                with self.subTest(shape=shape, method=method):
                    out = forward_fillna(t, 1 if t.ndim > 1 else 0, method=method).numpy()
                    ref = forward_fillna_np(raw, 1 if raw.ndim > 1 else 0)
                    np.testing.assert_allclose(out, ref, equal_nan=True)

    def test_loop_matches_legacy_vector_on_panel(self) -> None:
        rng = np.random.default_rng(1)
        raw = _random_nan(rng, (8, 20, 1, 10))
        t = torch.tensor(raw, dtype=torch.float64)
        loop = forward_fillna(t, 1, method='loop').numpy()
        legacy = legacy_fillna_torch(t, 1).numpy()
        np.testing.assert_allclose(loop, legacy, equal_nan=True)

    def test_inplace_preserves_storage(self) -> None:
        rng = np.random.default_rng(2)
        t = torch.tensor(_random_nan(rng, (6, 15, 1, 4)), dtype=torch.float64)
        base = t.clone()
        out = forward_fillna(t, 1, method='loop', inplace=True)
        self.assertIs(out, t)
        expected = forward_fillna(base, 1, method='loop', inplace=False)
        torch.testing.assert_close(out, expected, equal_nan=True)

    def test_inplace_matches_copy_on_numpy(self) -> None:
        rng = np.random.default_rng(3)
        arr = _random_nan(rng, (5, 12, 2))
        base = arr.copy()
        out = forward_fillna(arr, 1, method='loop', inplace=True)
        self.assertIs(out, arr)
        np.testing.assert_allclose(out, forward_fillna_np(base, 1), equal_nan=True)

    def test_force_value_inplace(self) -> None:
        t = torch.tensor([[float('nan'), 1.0, float('nan')], [2.0, float('nan'), float('nan')]])
        forward_fillna(t, 1, force_value=0.0, inplace=True)
        self.assertTrue(torch.isnan(t[0, 0]))
        self.assertEqual(t[0, 2].item(), 0.0)
        self.assertEqual(t[1, 2].item(), 0.0)

    def test_backward_loop(self) -> None:
        rng = np.random.default_rng(4)
        raw = _random_nan(rng, (3, 5, 4))
        t = torch.tensor(raw, dtype=torch.float64)
        out = backward_fillna(t, 1, method='loop').numpy()
        ref = legacy_fillna_torch(t, 1, backward=True).numpy()
        np.testing.assert_allclose(out, ref, equal_nan=True)


if __name__ == '__main__':
    unittest.main()
