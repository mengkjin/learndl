"""Check that ranking and hidden-aggregation parameters reach the native engines."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from src.res.algo.boost.booster.general import GeneralBoostModel
from src.res.algo.boost.util import BoostInput


class BoostParameterTest(unittest.TestCase):
    def data(self, seed):
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(256, 5, 4, generator=generator)
        y = x[:, :, 0] + 0.2 * x[:, :, 1]
        return BoostInput.from_tensor(x, y, secid=np.arange(256), date=np.arange(5))

    def test_rank_target_reaches_lightgbm_training_and_evaluation(self):
        model = GeneralBoostModel('lgbm', params={
            'num_boost_round': 3, 'early_stopping': 0,
        }, override_boost={'param': {'objective': 'rank', 'rank_target_size': 50}},
            cuda=False, seed=42)
        model.fit(self.data(1), self.data(2), silent=True)
        native = model.boost.model
        self.assertEqual(native.params['lambdarank_truncation_level'], 50)
        self.assertNotIn('lambdarank_truncation_target', native.params)
        self.assertIn('ndcg@50', model.boost.evals_result['valid'])
        self.assertNotIn('ndcg@100', model.boost.evals_result['valid'])
        self.assertTrue(np.isfinite(model.predict('valid').pred.numpy()).all())

    def test_mse_does_not_send_ranking_parameters(self):
        model = GeneralBoostModel('lgbm', params={
            'num_boost_round': 2, 'early_stopping': 0,
        }, cuda=False, seed=42)
        model.fit(self.data(1), self.data(2), silent=True)
        self.assertNotIn('lambdarank_truncation_level', model.boost.model.params)
        self.assertTrue(np.isfinite(model.predict('valid').pred.numpy()).all())

    def test_xgboost_schedule_parameters_survive_filtering(self):
        root = Path(__file__).resolve().parents[1]
        schedule = yaml.safe_load((root / 'configs/schedule/current/gru_day_xgboost.yaml').read_text())
        params = {key: value[0] for key, value in schedule['algo']['xgboost'].items()}
        params.update(num_boost_round=3, early_stopping=0)
        model = GeneralBoostModel('xgboost', params=params,
            override_boost=schedule['train']['boost'], cuda=False, seed=42)
        model.fit(self.data(1), self.data(2), silent=True)
        native = json.loads(model.boost.model.save_config())
        values = {}

        def collect(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(child, (str, int, float)):
                        values.setdefault(key, []).append(child)
                    else:
                        collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(native)
        for key in ('min_child_weight', 'max_bin', 'max_depth', 'subsample',
                    'colsample_bytree', 'learning_rate', 'reg_lambda', 'reg_alpha'):
            with self.subTest(parameter=key):
                self.assertTrue(any(np.isclose(float(v), params[key]) for v in values.get(key, [])))
        self.assertIn('hist', values['tree_method'])
        self.assertTrue(np.isfinite(model.predict('valid').pred.numpy()).all())


if __name__ == '__main__':
    unittest.main()
