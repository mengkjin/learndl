"""Undefined group IR must not break the monotony report or be replaced with zero."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.res.factor.util.plot.factor import Plotter
from src.res.factor.util.stats.factor import calc_ic_monotony

__all__ = ['MonotonyPlotTest']


class MonotonyPlotTest(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def draw(self, frame):
        fig = plt.figure()
        plotter = Plotter()
        plotter.plot_iter = SimpleNamespace(
            set_args=Mock(), iter=lambda: iter([(frame, fig)]), figs={'test': fig},
        )
        plotter.plot_ic_monotony(frame)
        return fig

    def test_one_observation_per_group_drops_ir_but_plot_survives(self):
        factor = Mock()
        factor.within.return_value = factor
        factor.eval_group_perf.return_value = pd.DataFrame({
            'factor_name': ['example', 'example'], 'group': [1, 2],
            'start': [20261117, 20261117], 'end': [20261118, 20261118],
            'group_ret': [.01, .02],
        })
        stats = calc_ic_monotony(factor)
        self.assertNotIn('grp_ir', stats.stats_name.tolist())
        fig = self.draw(stats)
        self.assertEqual(len(fig.axes[0].patches), 2)
        self.assertTrue(np.isnan(fig.axes[1].lines[0].get_ydata()).all())
        self.assertIn('IR unavailable', [t.get_text() for t in fig.axes[1].texts])

    def test_valid_ir_unchanged(self):
        frame = pd.DataFrame({
            'group': [1, 2, 1, 2],
            'stats_name': ['grp_ret', 'grp_ret', 'grp_ir', 'grp_ir'],
            'stats_value': [.1, .2, .5, 1.5],
        })
        fig = self.draw(frame)
        np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), [.5, 1.5])
        self.assertFalse(fig.axes[1].texts)
