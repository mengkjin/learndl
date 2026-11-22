"""Small independent continuous-market and sparse-event contribution branches."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["MarketGate", "EventGate"]


class _LowRankGate(nn.Module):
    """Zero output initialization preserves the parent model, not both matrices."""

    def __init__(self, input_dim: int, factor_count: int, rank: int) -> None:
        super().__init__()
        if factor_count < 1 or rank < 1:
            raise ValueError("Positive factor count and context rank required")
        self.project = nn.Linear(input_dim, rank, bias=False)
        self.output = nn.Linear(rank, factor_count, bias=False)
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def records(market: Tensor) -> tuple[Tensor, Tensor]:
        # Repaired order: three (event, age) pairs, market_up, eight risk metrics;
        # each numeric channel is immediately followed by its validity mask.
        if market.ndim != 3 or tuple(market.shape) != (1, 1, 30):
            raise ValueError("Expected current shared repaired market [1, 1, 30]")
        packed = market[0, 0].reshape(15, 2)
        value, mask = packed[:, 0], packed[:, 1]
        valid = (mask == 1) & torch.isfinite(value)
        return torch.where(valid, value, torch.zeros_like(value)), valid

    def response(self, features: Tensor, ready: Tensor) -> Tensor:
        logits = self.output(torch.tanh(self.project(features)))
        return torch.where(ready, logits, torch.zeros_like(logits))


class MarketGate(_LowRankGate):
    """Only market_up and eight standardized risks; event channels cannot enter."""

    def __init__(self, factor_count: int, rank: int = 2) -> None:
        super().__init__(18, factor_count, rank)

    def encode(self, market: Tensor) -> tuple[Tensor, Tensor]:
        value, valid = self.records(market)
        value, valid = value[6:], valid[6:]
        # The repaired market_up channel is binary, not a continuous return.
        valid = valid & torch.cat(
            ((value[:1] == 0) | (value[:1] == 1), torch.ones_like(valid[1:]))
        )
        value = torch.where(valid, value, torch.zeros_like(value))
        features = torch.stack((value, valid.to(value.dtype)), dim=-1).flatten()
        return features, valid.any()

    def forward(self, market: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features, ready = self.encode(market)
        return self.response(features, ready), features, ready


class EventGate(_LowRankGate):
    """Three event flags and exp(-age / tau), with separate source/age validity.

    Unknown age is not age zero. The stored log age is capped at 252 sessions;
    at that cap the decay is explicitly zero rather than a perpetual tail.
    """

    def __init__(
        self, factor_count: int, rank: int = 1, decay_days: float = 20.0
    ) -> None:
        super().__init__(12, factor_count, rank)
        if not math.isfinite(decay_days) or decay_days <= 0:
            raise ValueError("Positive finite event decay required")
        self.decay_days = decay_days

    def encode(self, market: Tensor) -> tuple[Tensor, Tensor]:
        value, valid = self.records(market)
        events, masks = value[:6].reshape(3, 2), valid[:6].reshape(3, 2)
        flag, age = events[:, 0], events[:, 1]
        flag_valid = masks[:, 0] & ((flag == 0) | (flag == 1))
        age_valid = flag_valid & masks[:, 1] & (age >= 0) & (age <= 1)
        flag = torch.where(flag_valid, flag, torch.zeros_like(flag))
        # Clamp only for safe arithmetic; invalid inputs retain false validity.
        sessions = torch.expm1(age.clamp(0, 1) * math.log1p(252))
        decay = torch.where(
            age_valid & (age < 1),
            torch.exp(-sessions / self.decay_days),
            torch.zeros_like(age),
        )
        # No known current/recent event => exactly no event response even after
        # training. This branch cannot turn ordinary no-event dates into a bias.
        ready = ((flag == 1) | (decay > 0)).any()
        features = torch.stack(
            (flag, flag_valid.to(flag.dtype), decay, age_valid.to(flag.dtype)), dim=-1
        ).flatten()
        return features, ready

    def forward(self, market: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features, ready = self.encode(market)
        intensity = features.reshape(3, 4)[:, (0, 2)].amax()
        # Validity inputs must not create a constant response as an event ages.
        return intensity * self.response(features, ready), features, ready
