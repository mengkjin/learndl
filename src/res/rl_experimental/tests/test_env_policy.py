from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env

from src.res.rl_experimental.data import make_synthetic_panel
from src.res.rl_experimental.env import EpisodeRange, PortfolioEnv
from src.res.rl_experimental.policy import MaskedDiagonalGaussian, MaskedStockActorCriticPolicy
from src.res.rl_experimental.portfolio import PortfolioConstraints
from src.res.rl_experimental.training import ExperimentConfig, evaluate_checkpoint, train_experiment


class EnvironmentAndPolicyTest(unittest.TestCase):
    def _env(self) -> PortfolioEnv:
        panel = make_synthetic_panel(50, 24, 4, seed=3)
        return PortfolioEnv(panel, EpisodeRange(0, 40), PortfolioConstraints(8, 4, 0.3))

    def test_gym_contract_and_legal_steps(self) -> None:
        env = self._env()
        check_env(env, warn=False)
        obs, _ = env.reset(seed=2)
        self.assertEqual(obs["features"].shape, (12, 4))
        for _ in range(4):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            self.assertTrue(np.isfinite(reward))
            self.assertFalse(truncated)
            self.assertLessEqual(np.count_nonzero(env.full_weights > 1e-8), 4)
            self.assertLessEqual(float(np.max(info["target_weights"], initial=0)), 0.3 + 1e-7)
            self.assertFalse(terminated)

    def test_masked_gaussian_ignores_invalid_actions(self) -> None:
        mean = torch.zeros((1, 3, 2))
        distribution = MaskedDiagonalGaussian(mean, torch.zeros((1, 1, 2)), torch.tensor([[1.0, 0.0, 1.0]]))
        first = torch.zeros((1, 3, 2))
        second = first.clone()
        second[:, 1, :] = 100.0
        torch.testing.assert_close(distribution.log_prob(first), distribution.log_prob(second))

    def test_policy_is_permutation_equivariant(self) -> None:
        env = self._env()
        obs, _ = env.reset(seed=1)
        policy = MaskedStockActorCriticPolicy(env.observation_space, env.action_space, lambda _: 3e-4, hidden_dim=16)
        tensor_obs = {key: torch.as_tensor(value).unsqueeze(0) for key, value in obs.items()}
        mean, value = policy.network(tensor_obs)
        permutation = np.arange(env.constraints.max_slots)[::-1].copy()
        permuted = {
            key: (value[:, permutation] if key != "account" else value)
            for key, value in tensor_obs.items()
        }
        perm_mean, perm_value = policy.network(permuted)
        torch.testing.assert_close(mean[:, permutation], perm_mean)
        torch.testing.assert_close(value, perm_value)

    def test_future_execution_data_is_not_in_observation(self) -> None:
        panel = make_synthetic_panel(20, 12, 3, seed=23)
        shape = panel.alpha.shape
        execution = dict(
            execution_dates=panel.return_end_dates.copy(),
            overnight_returns=np.zeros(shape, dtype=np.float32),
            intraday_returns=np.nan_to_num(panel.forward_returns, nan=0.0),
            can_buy=np.ones(shape, dtype=bool),
            can_sell=np.ones(shape, dtype=bool),
            valuation_stale=np.zeros(shape, dtype=bool),
        )
        first = replace(panel, **execution)
        changed_buy = execution["can_buy"].copy()
        changed_buy[5:] = False
        second = replace(panel, **(execution | {"can_buy": changed_buy}))
        env_a = PortfolioEnv(first, EpisodeRange(0, 15), PortfolioConstraints(6, 3, 0.4))
        env_b = PortfolioEnv(second, EpisodeRange(0, 15), PortfolioConstraints(6, 3, 0.4))
        obs_a, _ = env_a.reset(seed=1)
        obs_b, _ = env_b.reset(seed=1)
        for key in obs_a:
            np.testing.assert_array_equal(obs_a[key], obs_b[key])

    def test_short_sb3_training_and_reload(self) -> None:
        panel = make_synthetic_panel(70, 24, 4, seed=5)
        config = ExperimentConfig(
            candidate_count=8,
            max_holdings=4,
            max_weight=0.3,
            n_envs=2,
            rollout_steps=8,
            batch_size=8,
            update_epochs=1,
            hidden_dim=16,
            total_timesteps=32,
            eval_freq=16,
        )
        with tempfile.TemporaryDirectory() as directory:
            metrics = train_experiment(panel, directory, config)
            self.assertTrue((Path(directory) / "best_model.zip").exists())
            self.assertTrue((Path(directory) / "test_history.csv").exists())
            self.assertTrue(np.isfinite(metrics["test"]["final_nav"]))
            self.assertGreater(metrics["parameter_change_l2"], 0.0)
            export = Path(directory) / "export"
            reloaded = evaluate_checkpoint(panel, Path(directory) / "best_model.zip", export)
            self.assertAlmostEqual(reloaded["final_nav"], metrics["test"]["final_nav"], places=7)
            self.assertTrue((export / "deterministic_weights.csv").exists())
            changed_alpha = panel.alpha.copy()
            changed_alpha[0, 0] += 0.1
            with self.assertRaisesRegex(ValueError, "panel contract"):
                evaluate_checkpoint(replace(panel, alpha=changed_alpha), Path(directory) / "best_model.zip", export)


if __name__ == "__main__":
    unittest.main()
