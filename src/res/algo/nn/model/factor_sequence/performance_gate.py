"""Causal, mask-aware performance summaries and a shared bounded factor gate."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["PerformanceGate"]


class PerformanceGate(nn.Module):
    """Read [factor IC, IC valid, spread, spread valid] in frozen factor order.

    Each factor uses the same MLP; there are no learned temporal kernels or factor
    embeddings. Means normalize over observed weights, while coverage normalizes
    over all calendar weights. Spread is already in percentage points upstream.
    """

    half_lives: Tensor

    def __init__(
        self,
        factor_count: int,
        hidden_dim: int = 8,
        half_lives: tuple[float, ...] = (5.0, 20.0, 60.0),
        scale: float = 0.2,
        min_observations: int = 20,
    ) -> None:
        super().__init__()
        if (
            factor_count < 1
            or hidden_dim < 1
            or len(half_lives) < 2
            or any(not math.isfinite(h) or h <= 0 for h in half_lives)
            or any(a >= b for a, b in zip(half_lives, half_lives[1:]))
            or not 0 < scale < 1
            or min_observations < 1
        ):
            raise ValueError("Invalid performance gate parameters")
        self.factor_count = factor_count
        self.scale = scale
        self.min_observations = min_observations
        self.register_buffer(
            "half_lives", torch.tensor(half_lives, dtype=torch.float32)
        )
        # Per metric: means at each scale, weighted coverages, short-long mean.
        self.feature_count = 4 * len(half_lives) + 2
        self.hidden = nn.Linear(self.feature_count, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def summarize(self, performance: Tensor) -> tuple[Tensor, Tensor]:
        if (
            performance.ndim != 3
            or performance.shape[0] != 1
            or performance.shape[1] < 1
            or performance.shape[2] != 4 * self.factor_count
        ):
            raise ValueError("Expected shared performance [1, days, 4 * factors]")
        records = performance[0].reshape(-1, self.factor_count, 2, 2)
        values, masks = records[..., 0], records[..., 1]
        valid = (masks == 1) & torch.isfinite(values)
        values = torch.where(valid, values, torch.zeros_like(values))
        age = torch.arange(
            len(records) - 1, -1, -1, device=performance.device, dtype=performance.dtype
        )
        weights = torch.exp2(-age[:, None] / self.half_lives.to(performance.dtype))
        observed_weights = torch.einsum("tfm,th->fmh", valid.to(values.dtype), weights)
        numerator = torch.einsum("tfm,th->fmh", values, weights)
        means = numerator / observed_weights.clamp_min(torch.finfo(values.dtype).tiny)
        coverage = observed_weights / weights.sum(0)
        features = torch.cat(
            (means, coverage, means[..., :1] - means[..., -1:]), dim=-1
        ).reshape(self.factor_count, self.feature_count)
        # Require usable history for BOTH IC and spread. No learned missing-data gate.
        ready = (valid.sum(0) >= self.min_observations).all(-1)
        return features, ready

    def encode(self, performance: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return masked logits so independent context streams can be added safely."""
        features, ready = self.summarize(performance)
        logits = self.output(torch.nn.functional.gelu(self.hidden(features))).squeeze(
            -1
        )
        return torch.where(ready, logits, torch.zeros_like(logits)), features, ready

    def forward(self, performance: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits, features, ready = self.encode(performance)
        delta = self.scale * torch.tanh(logits)
        return 1 + delta, features, ready
