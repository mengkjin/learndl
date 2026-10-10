from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.res.rl_experimental import training
from src.res.rl_experimental.data import Standardizer, date_split, make_synthetic_panel
from src.res.rl_experimental.diagnostics import compare_metrics, comparison_figure


class TrainingLoggingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.panel = make_synthetic_panel(70, 24, 4, seed=5)
        self.config = training.ExperimentConfig(
            candidate_count=8, max_holdings=4, max_weight=0.3,
            n_envs=2, rollout_steps=8, batch_size=8, update_epochs=1,
            hidden_dim=16, total_timesteps=32,
        )

    def test_records_completed_updates_and_selected_checkpoint(self) -> None:
        observed = []
        original_evaluate = training.evaluate_model

        def observe(model, env):
            observed.append((model._n_updates, id(env)))
            return original_evaluate(model, env)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(training, "evaluate_model", side_effect=observe), patch.object(
                training, "replay_baseline", wraps=training.replay_baseline,
            ) as replay:
                metrics = training.train_experiment(self.panel, output, self.config)
            self.assertEqual(replay.call_count, 2)  # validation once, test once
            self.assertEqual([x[0] for x in observed[:3]], [0, 1, 2])
            self.assertEqual(len({x[1] for x in observed[:3]}), 1)
            self.assertNotEqual(observed[-1][1], observed[0][1])  # test only after training
            with (output / "update_history.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(x["environment_steps"]) for x in rows], [16, 32])
            self.assertEqual([int(x["update"]) for x in rows], [1, 2])
            self.assertNotIn("rollout/frozen_holdings_mean", rows[0])
            self.assertIn("rollout/reward_mean", rows[0])
            self.assertIn("rollout/reward_component/log_growth_mean", rows[0])
            events = EventAccumulator(str(output / "tensorboard/PPO_1")).Reload()
            losses = events.Scalars("train/value_loss")
            self.assertEqual([x.step for x in losses], [16, 32])
            np.testing.assert_allclose([x.value for x in losses], [float(x["train/value_loss"]) for x in rows], rtol=1e-6)
            self.assertEqual([x.step for x in events.Scalars("validation/ppo/final_nav")], [0, 16, 32])
            self.assertEqual([x.step for x in events.Scalars("test/ppo/final_nav")], [32])
            self.assertIn("test/comparison/curves", events.Tags()["images"])
            self.assertTrue((output / "test_comparison.png").is_file())
            self.assertEqual(metrics["completed_updates"], 2)
            expected_best = max(rows, key=lambda row: float(row["validation/ppo/final_nav"]))
            self.assertEqual(metrics["best_update"], int(expected_best["update"]))
            splits = date_split(self.panel)
            normalized = Standardizer.fit(self.panel, *splits["train"]).transform(self.panel)
            env = training._env(normalized, splits["valid"], self.config, random_start=False)
            model = PPO.load(output / "best_model.zip", env=env)
            actual, _ = original_evaluate(model, env)
            self.assertAlmostEqual(actual["final_nav"], metrics["validation"]["final_nav"])
            env.close()

    def test_discounted_reward_can_select_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metrics = training.train_experiment(
                self.panel, output, replace(self.config, selection_metric="discounted_reward"),
            )
            with (output / "validation_updates.csv").open() as handle:
                rows = list(csv.DictReader(handle))[1:]  # initial policy cannot be selected
            expected = max(rows, key=lambda row: float(row["validation/ppo/discounted_reward"]))
            self.assertEqual(metrics["selection_metric"], "discounted_reward")
            self.assertEqual(metrics["best_update"], int(expected["update"]))
            self.assertAlmostEqual(
                metrics["validation"]["discounted_reward"],
                float(expected["validation/ppo/discounted_reward"]),
            )

    def test_positive_interval_rounds_up_and_always_validates_final_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            training.train_experiment(self.panel, output, replace(self.config, total_timesteps=48, eval_freq=30))
            with (output / "validation_updates.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(x["environment_steps"]) for x in rows], [0, 32, 48])
            events = EventAccumulator(str(output / "tensorboard/PPO_1")).Reload()
            self.assertEqual([x.step for x in events.Scalars("train/value_loss")], [16, 32, 48])

    def test_comparison_units_and_date_alignment(self) -> None:
        base = dict(total_return=0.1, final_nav=1.1, max_drawdown=-0.2,
                    annualized_volatility=0.2, mean_turnover=0.1, total_fees=0.001)
        result = compare_metrics(base | {"total_return": 0.2, "final_nav": 1.2}, base)
        self.assertAlmostEqual(result["return_difference_pp"], 10.0)
        self.assertAlmostEqual(result["relative_nav_return"], 1.2 / 1.1 - 1)
        first = [{"decision_date": 1, "return_end_date": 2}]
        second = [{"decision_date": 1, "return_end_date": 3}]
        with self.assertRaisesRegex(ValueError, "dates do not match"):
            comparison_figure({"ppo": first, "equal_weight": second})


if __name__ == "__main__":
    unittest.main()
