from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data import PanelData
from .portfolio import PortfolioConstraints, execute_open_close, project_action, select_slots, settle_period
from .reward import RewardConfig, RewardContext, RewardEvaluator


@dataclass(frozen=True)
class EpisodeRange:
    start: int
    end: int

    def validate(self, n_steps: int) -> None:
        if not 0 <= self.start < self.end <= n_steps:
            raise ValueError("episode range must be within the panel")


class PortfolioEnv(gym.Env):
    """Daily portfolio environment with dynamic identities in fixed slots."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: PanelData,
        episode_range: EpisodeRange,
        constraints: PortfolioConstraints | None = None,
        *,
        reward_scale: float = 100.0,
        reward_config: RewardConfig | dict[str, Any] | None = None,
        random_start: bool = False,
        minimum_episode_steps: int = 32,
    ) -> None:
        super().__init__()
        episode_range.validate(panel.n_steps)
        self.panel = panel
        self.episode_range = episode_range
        self.constraints = constraints or PortfolioConstraints()
        self.constraints.validate()
        self.reward_scale = float(reward_scale)
        self.reward_evaluator = RewardEvaluator(reward_config, self.reward_scale)
        self.random_start = bool(random_start)
        self.minimum_episode_steps = max(1, int(minimum_episode_steps))
        s = self.constraints.max_slots
        f = panel.observation_feature_count
        self.observation_space = spaces.Dict(
            {
                "features": spaces.Box(-np.inf, np.inf, shape=(s, f), dtype=np.float32),
                "alpha": spaces.Box(-np.inf, np.inf, shape=(s,), dtype=np.float32),
                "current_weights": spaces.Box(0.0, 1.0, shape=(s,), dtype=np.float32),
                "present_mask": spaces.Box(0.0, 1.0, shape=(s,), dtype=np.float32),
                "investable_mask": spaces.Box(0.0, 1.0, shape=(s,), dtype=np.float32),
                "account": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
            }
        )
        # SB3 stores Box actions flattened; pairs are restored inside step().
        self.action_space = spaces.Box(-10.0, 10.0, shape=(s * 2,), dtype=np.float32)
        self._time = episode_range.start
        self._episode_end = episode_range.end
        self._weights = np.zeros(panel.n_stocks, dtype=np.float32)
        self._cash = 1.0
        self._nav = 1.0
        self._slots = np.full(s, -1, dtype=np.int64)
        self._present = np.zeros(s, dtype=bool)
        self._can_invest = np.zeros(s, dtype=bool)
        self._slot_alpha = np.zeros(s, dtype=np.float32)
        self.history: list[dict[str, Any]] = []

    def _choose_start(self, options: dict[str, Any] | None) -> int:
        if options and "start_index" in options:
            start = int(options["start_index"])
            if not self.episode_range.start <= start < self.episode_range.end:
                raise ValueError("requested start_index is outside episode range")
            return start
        if not self.random_start:
            return self.episode_range.start
        last = max(self.episode_range.start, self.episode_range.end - self.minimum_episode_steps)
        return int(self.np_random.integers(self.episode_range.start, last + 1))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._time = self._choose_start(options)
        self._episode_end = self.episode_range.end
        self._weights.fill(0.0)
        self._cash = 1.0
        self._nav = 1.0
        self.history = []
        obs = self._observation()
        return obs, self._base_info()

    def _alpha_zscore(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(values, dtype=np.float32)
        if mask.any():
            active = values[mask].astype(np.float64)
            std = active.std()
            result[mask] = ((active - active.mean()) / max(std, 1e-6)).astype(np.float32)
        return result

    def _observation(self) -> dict[str, np.ndarray]:
        t = min(self._time, self.episode_range.end - 1)
        self._slots, self._present, self._can_invest = select_slots(
            self.panel.alpha[t],
            self.panel.investable[t],
            self._weights,
            self.panel.stock_ids,
            self.constraints,
        )
        s = self.constraints.max_slots
        features = np.zeros((s, self.panel.observation_feature_count), dtype=np.float32)
        alpha = np.zeros(s, dtype=np.float32)
        current = np.zeros(s, dtype=np.float32)
        if self._present.any():
            idx = self._slots[self._present]
            features[self._present] = self.panel.observation_features(t, idx)
            alpha[self._present] = self.panel.alpha[t, idx]
            current[self._present] = self._weights[idx]
        alpha = self._alpha_zscore(alpha, self._can_invest)
        self._slot_alpha = alpha
        remaining = (self.episode_range.end - t) / max(1, self.episode_range.end - self.episode_range.start)
        return {
            "features": features,
            "alpha": alpha,
            "current_weights": current,
            "present_mask": self._present.astype(np.float32),
            "investable_mask": self._can_invest.astype(np.float32),
            "account": np.asarray([self._cash, remaining], dtype=np.float32),
        }

    def _base_info(self) -> dict[str, Any]:
        return {
            "time_index": self._time,
            "decision_date": self.panel.decision_dates[min(self._time, self.panel.n_steps - 1)].item(),
            "nav": self._nav,
            "cash_weight": self._cash,
            "stock_ids": self.panel.stock_ids[self._slots[self._present]].copy(),
        }

    def step(self, action: np.ndarray):
        if self._time >= self._episode_end:
            raise RuntimeError("step called after termination; reset the environment")
        decision_observation = {key: value.copy() for key, value in self._observation().items()}
        for value in decision_observation.values():
            value.flags.writeable = False
        old_weights = self._weights.copy()
        slot_alpha = self._slot_alpha.copy()
        slot_ids = np.full(self.constraints.max_slots, np.iinfo(np.int64).max, dtype=np.int64)
        if self._present.any():
            idx = self._slots[self._present]
            slot_ids[self._present] = self.panel.stock_ids[idx]
        slot_weights, target_cash, chosen_slots = project_action(
            action,
            slot_alpha,
            self._can_invest,
            slot_ids,
            self.constraints,
        )
        target = np.zeros(self.panel.n_stocks, dtype=np.float32)
        if self._present.any():
            target[self._slots[self._present]] = slot_weights[self._present]
        previous_nav = self._nav
        if self.panel.is_execution_panel:
            execution_dates, overnight, intraday, can_buy, can_sell, valuation_stale = self.panel.execution_arrays()
            execution = execute_open_close(
                self._weights,
                self._cash,
                target,
                overnight[self._time],
                intraday[self._time],
                can_buy[self._time],
                can_sell[self._time],
                fee_rate=self.constraints.fee_rate,
                max_holdings=self.constraints.max_holdings,
                max_weight=self.constraints.max_weight,
            )
            new_weights = execution.end_weights
            new_cash = execution.end_cash
            nav_multiplier = execution.nav_multiplier
            turnover = execution.turnover
            cost_fraction = execution.cost_fraction
            execution_info = {
                "execution_date": execution_dates[self._time].item(),
                "overnight_return": execution.overnight_multiplier - 1.0,
                "buy_notional": execution.buy_notional,
                "sell_notional": execution.sell_notional,
                "frozen_holdings": execution.frozen_count,
                "rejected_buys": execution.rejected_buy_count,
                "rejected_sells": execution.rejected_sell_count,
                "stale_valuations": int(np.count_nonzero(valuation_stale[self._time] & (new_weights > 1e-8))),
            }
        else:
            new_weights, new_cash, nav_multiplier, turnover, cost_fraction = settle_period(
                self._weights,
                target,
                target_cash,
                self.panel.forward_returns[self._time],
                self.constraints.fee_rate,
            )
            execution_info = {}
        self._weights = new_weights
        self._cash = float(new_cash)
        self._nav *= nav_multiplier
        fee = previous_nav * cost_fraction
        reward_weights = {
            "current": old_weights.copy(), "target": target.copy(), "end": new_weights.copy(),
        }
        for value in reward_weights.values():
            value.flags.writeable = False
        reward_result = self.reward_evaluator(RewardContext(
            decision_date=int(self.panel.decision_dates[self._time]),
            return_end_date=int(self.panel.return_end_dates[self._time]),
            observation=MappingProxyType(decision_observation),
            current_weights=reward_weights["current"],
            target_weights=reward_weights["target"],
            end_weights=reward_weights["end"],
            end_cash=float(new_cash),
            nav_multiplier=float(nav_multiplier),
            net_return=float(nav_multiplier - 1.0),
            turnover=float(turnover),
            fee_fraction=float(cost_fraction),
            execution=MappingProxyType({
                key: value for key, value in execution_info.items() if isinstance(value, (int, float, np.number))
            }),
        ))
        reward = reward_result.reward
        record = {
            "time_index": self._time,
            "decision_date": self.panel.decision_dates[self._time].item(),
            "return_end_date": self.panel.return_end_dates[self._time].item(),
            "nav": self._nav,
            "period_return": nav_multiplier - 1.0,
            "reward": reward,
            "reward_unscaled": reward / self.reward_scale,
            "turnover": turnover,
            "fee_estimate": fee,
            "cash_weight": self._cash,
            "holdings": int(np.count_nonzero(self._weights > 1e-8)),
            "selected_stock_ids": self.panel.stock_ids[self._slots[chosen_slots]].copy(),
            "target_weights": slot_weights[chosen_slots].copy(),
            "held_stock_ids": self.panel.stock_ids[self._weights > 1e-8].copy(),
            "held_weights": self._weights[self._weights > 1e-8].copy(),
        } | execution_info | {f"reward_component/{key}": value for key, value in reward_result.components.items()}
        self.history.append(record)
        self._time += 1
        terminated = self._time >= self._episode_end
        obs = self._observation()
        info = self._base_info() | record
        return obs, reward, terminated, False, info

    @property
    def full_weights(self) -> np.ndarray:
        return self._weights.copy()

    def render(self):
        return self._base_info()
