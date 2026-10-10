"""Stateless, auditable reward functions for the portfolio environment."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, TypeAlias

import numpy as np


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class RewardConfig:
    """Serializable reward definition. Zero penalties reproduce the legacy reward."""

    turnover_penalty: float = 0.0
    downside_penalty: float = 0.0
    concentration_penalty: float = 0.0
    custom_function: str | None = None
    custom_params: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: "RewardConfig | Mapping[str, Any] | None") -> "RewardConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**dict(value))

    def validate(self) -> None:
        penalties = (self.turnover_penalty, self.downside_penalty, self.concentration_penalty)
        if any(not np.isfinite(value) or value < 0 for value in penalties):
            raise ValueError("reward penalty coefficients must be finite and nonnegative")
        try:
            json.dumps(self.custom_params, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("custom reward parameters must be JSON serializable") from error
        if self.custom_function is not None and self.custom_function.count(":") != 1:
            raise ValueError("custom_function must use module:function")


@dataclass(frozen=True)
class RewardContext:
    """Only the decision state and realized one-step execution feedback."""

    decision_date: int
    return_end_date: int
    observation: Mapping[str, np.ndarray]
    current_weights: np.ndarray
    target_weights: np.ndarray
    end_weights: np.ndarray
    end_cash: float
    nav_multiplier: float
    net_return: float
    turnover: float
    fee_fraction: float
    execution: Mapping[str, float | int]


@dataclass(frozen=True)
class RewardResult:
    reward: float
    components: Mapping[str, float]


RewardFunction: TypeAlias = Callable[[RewardContext, Mapping[str, JsonValue]], RewardResult]


def builtin_reward(context: RewardContext, params: Mapping[str, JsonValue]) -> RewardResult:
    """Log growth plus optional turnover, downside and concentration penalties."""
    config = RewardConfig.from_value(params)
    config.validate()
    components = {
        "log_growth": float(np.log(context.nav_multiplier)),
        "turnover_penalty": -config.turnover_penalty * context.turnover,
        "downside_penalty": -config.downside_penalty * min(context.net_return, 0.0) ** 2,
        "concentration_penalty": -config.concentration_penalty * float(np.square(context.end_weights).sum()),
    }
    return RewardResult(float(sum(components.values())), MappingProxyType(components))


def example_drawdown_aware_reward(context: RewardContext, params: Mapping[str, JsonValue]) -> RewardResult:
    """Runnable custom example; still stateless and based on realized one-step feedback."""
    downside = float(params.get("downside_penalty", 2.0))
    turnover = float(params.get("turnover_penalty", 0.001))
    components = {
        "log_growth": float(np.log(context.nav_multiplier)),
        "custom_downside": -downside * min(context.net_return, 0.0) ** 2,
        "custom_turnover": -turnover * context.turnover,
    }
    return RewardResult(float(sum(components.values())), MappingProxyType(components))


def _load_function(entrypoint: str) -> RewardFunction:
    module_name, function_name = entrypoint.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"custom reward {entrypoint} is not callable")
    return function


class RewardEvaluator:
    def __init__(self, config: RewardConfig | Mapping[str, Any] | None, scale: float) -> None:
        self.config = RewardConfig.from_value(config)
        self.config.validate()
        self.scale = float(scale)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        self.entrypoint = self.config.custom_function or "src.res.rl_experimental.reward:builtin_reward"
        self.function = _load_function(self.entrypoint)

    @property
    def metadata(self) -> dict[str, Any]:
        source = inspect.getsourcefile(self.function)
        digest = None
        if source and Path(source).is_file():
            digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        return {
            "config": asdict(self.config),
            "entrypoint": self.entrypoint,
            "source_sha256": digest,
            "scale": self.scale,
        }

    def __call__(self, context: RewardContext) -> RewardResult:
        params: Mapping[str, JsonValue]
        if self.config.custom_function:
            params = MappingProxyType(dict(self.config.custom_params))
        else:
            params = MappingProxyType(asdict(self.config))
        result = self.function(context, params)
        if not isinstance(result, RewardResult):
            raise TypeError(f"custom reward {self.entrypoint} must return RewardResult")
        components = {str(key): float(value) for key, value in result.components.items()}
        if not components or any(not np.isfinite(value) for value in components.values()):
            raise FloatingPointError(f"reward components from {self.entrypoint} must be finite and nonempty")
        unscaled = float(result.reward)
        if not np.isfinite(unscaled) or not np.isclose(unscaled, sum(components.values()), rtol=1e-7, atol=1e-10):
            raise ValueError("RewardResult.reward must be finite and equal the sum of components")
        return RewardResult(self.scale * unscaled, MappingProxyType({key: self.scale * value for key, value in components.items()}))
