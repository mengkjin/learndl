"""Bounded market adjustments to the v4 Graph prediction contributions."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .contextual_v4 import FactorContextualV4
from .market_gate import EventGate, MarketGate

__all__ = ["FactorContextualV5", "V5_VARIANTS"]

V5_VARIANTS = ("market", "market_events")


class FactorContextualV5(FactorContextualV4):
    def __init__(self, input_dim, variant="market", hidden_dim=32, **kwargs):
        dims = [input_dim] if isinstance(input_dim, int) else list(input_dim)
        if len(dims) != 3 or dims[-1] != 30 or variant not in V5_VARIANTS:
            raise ValueError(
                "v5 requires stock, performance and repaired market streams"
            )
        super().__init__(dims[:2], variant="graph", hidden_dim=hidden_dim, **kwargs)
        self.market_gate = MarketGate(self.k, rank=2)
        self.stock_market_gate = nn.Linear(2, hidden_dim, bias=False)
        nn.init.zeros_(self.stock_market_gate.weight)
        self.event_gate = EventGate(self.k) if variant == "market_events" else None

    def forward(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 3:
            raise ValueError("v5 requires three ordered streams")
        market = x[2].squeeze(2) if x[2].ndim == 4 else x[2]
        features, ready = self.market_gate.encode(market)
        state = torch.tanh(self.market_gate.project(features)) * ready
        market_logits = self.market_gate.output(state)
        stock_offset = 0.2 * self.stock_market_gate(state).tanh()
        market_offset = 0.2 * market_logits.tanh()

        graph_score, other = self._forward_with_performance(
            x[:2], retain_factor_contributions=True
        )
        contributions = other.pop("factor_contributions")
        market_factor = (contributions * market_offset).sum(-1, keepdim=True)
        market_stock = F.linear(other["hidden"] * stock_offset, self.output.weight)
        event_factor = torch.zeros_like(market_factor)
        total_offset = market_offset
        penalty = market_offset.square().mean() + stock_offset.square().mean()
        if self.event_gate is not None:
            event_logits, _, _ = self.event_gate(market)
            total_offset = 0.2 * (market_logits + event_logits).tanh()
            event_factor = (contributions * (total_offset - market_offset)).sum(
                -1, keepdim=True
            )
            penalty = penalty + (0.2 * event_logits.tanh()).square().mean()
            values, valid = self.event_gate.records(market)
            other.update(event_values=values[:6], event_validity=valid[:6])

        correction = market_factor + market_stock + event_factor
        other.update(
            graph_score=graph_score,
            market_factor_correction=market_factor,
            market_stock_correction=market_stock,
            event_factor_correction=event_factor,
            market_ready=ready,
            market_factor_multiplier=1 + total_offset,
            market_stock_multiplier=1 + stock_offset,
            residual_score=other["residual_score"] + correction,
            loss_context_shrinkage=other["loss_context_shrinkage"]
            + self.context_penalty * penalty,
        )
        return graph_score + correction, other
