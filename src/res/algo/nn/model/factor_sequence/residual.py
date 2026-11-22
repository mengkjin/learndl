"""Compact residual models, with legacy FiLM or v4 bounded contribution gates."""

from __future__ import annotations

__all__ = ["FactorResidual"]

import torch
from torch import nn
from .performance_gate import PerformanceGate
from .market_gate import EventGate, MarketGate


class FactorResidual(nn.Module):
    mode = "snapshot"

    def __init__(
        self,
        input_dim,
        mode="snapshot",
        hidden_dim=32,
        dropout=0.1,
        rnn_layers=2,
        context_layers=1,
        gate_hidden_dim=8,
        gate_half_lives=(5.0, 20.0, 60.0),
        gate_scale=0.2,
        gate_min_observations=20,
        gate_penalty=0.01,
        market_rank=2,
        event_rank=1,
        event_decay_days=20.0,
        **kwargs,
    ):
        super().__init__()
        dims = [input_dim] if isinstance(input_dim, int) else list(input_dim)
        v4_modes = ("contextual_v4", "contextual_v4_market", "contextual_v4_events")
        self.is_v4 = mode in v4_modes
        self.uses_market = mode in v4_modes[1:]
        expected_streams = (
            3 if mode == "contextual" or self.uses_market else 2 if self.is_v4 else 1
        )
        if (
            mode not in ("snapshot", "temporal", "contextual", *v4_modes)
            or len(dims) != expected_streams
            or dims[0] % 4
        ):
            raise ValueError("Invalid compact residual streams/mode")
        self.mode = mode
        self.k = dims[0] // 4
        self.linear = nn.Linear(self.k, 1)
        width = dims[0]
        if mode != "snapshot":
            self.project = nn.Linear(dims[0], hidden_dim)
            self.stock = nn.GRU(
                hidden_dim,
                hidden_dim,
                rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0,
            )
            width += hidden_dim
        if mode == "contextual":
            self.performance = nn.GRU(
                dims[1], hidden_dim, context_layers, batch_first=True
            )
            self.market = nn.GRU(dims[2], hidden_dim, context_layers, batch_first=True)
            self.film = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.head = nn.Sequential(
            nn.Linear(width, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if self.is_v4:
            if dims[1] != 4 * self.k or not 0 <= gate_penalty < float("inf"):
                raise ValueError(
                    "v4 requires repaired IC/spread with validity and a nonnegative penalty"
                )
            self.performance_gate = PerformanceGate(
                self.k,
                gate_hidden_dim,
                tuple(gate_half_lives),
                gate_scale,
                gate_min_observations,
            )
            self.gate_penalty = gate_penalty
        if self.uses_market:
            if dims[2] != 30:
                raise ValueError("v4 market requires the 30-channel repaired contract")
            self.market_gate = MarketGate(self.k, market_rank)
        if mode == "contextual_v4_events":
            self.event_gate = EventGate(self.k, event_rank, event_decay_days)

    def forward(self, x):
        streams = list(x) if isinstance(x, (list, tuple)) else [x]
        streams = [v.squeeze(2) if v.ndim == 4 else v for v in streams]
        if self.is_v4 and (
            len(streams) != (3 if self.uses_market else 2)
            or streams[0].ndim != 3
            or streams[0].shape[-1] != self.k * 4
            or streams[1].ndim != 3
            or streams[1].shape[1] < streams[0].shape[1]
        ):
            raise ValueError("v4 requires stock and aligned shared performance streams")
        stock = streams[0]
        parts = [stock[:, -1]]
        if self.mode != "snapshot":
            z = self.project(stock)
            if self.mode == "contextual":
                p, m = streams[1:]
                if (
                    p.shape[0] != 1
                    or m.shape[0] != 1
                    or p.shape[1] != m.shape[1]
                    or p.shape[1] < stock.shape[1]
                ):
                    raise ValueError("Invalid shared context calendars")
                c = torch.cat((self.performance(p)[0], self.market(m)[0]), -1)
                gamma, beta = self.film(c[:, -stock.shape[1] :]).chunk(2, -1)
                z = z * (1 + gamma) + beta
            parts.append(self.stock(z)[0][:, -1])
        hidden = self.head(torch.cat(parts, -1))
        linear = self.linear(stock[:, -1, : self.k])
        residual = self.output(hidden)
        other = dict(hidden=hidden, linear_score=linear, residual_score=residual)
        if self.is_v4:
            logits, summaries, perf_ready = self.performance_gate.encode(streams[1])
            ready = perf_ready
            other.update(performance_logits=logits, performance_ready=perf_ready)
            # Independence is essential: missing performance never disables valid
            # market/event information, and missing market never disables events.
            branch_penalty = logits.new_zeros(())
            if self.uses_market:
                market_logits, market_features, market_ready = self.market_gate(
                    streams[2]
                )
                logits = logits + market_logits
                ready = ready | market_ready
                branch_penalty = (
                    (self.performance_gate.scale * torch.tanh(market_logits))
                    .square()
                    .mean()
                )
                other.update(
                    market_logits=market_logits,
                    market_features=market_features,
                    market_ready=market_ready,
                )
            if self.mode == "contextual_v4_events":
                event_logits, event_features, event_ready = self.event_gate(streams[2])
                logits = logits + event_logits
                ready = ready | event_ready
                branch_penalty = (
                    branch_penalty
                    + (self.performance_gate.scale * torch.tanh(event_logits))
                    .square()
                    .mean()
                )
                other.update(
                    event_logits=event_logits,
                    event_features=event_features,
                    event_ready=event_ready,
                )
            # A single bound after combination, rather than multiplying gates.
            gates = 1 + self.performance_gate.scale * torch.tanh(logits)
            values = stock[:, -1, : self.k]
            values = torch.where(
                (stock[:, -1, self.k : 2 * self.k] == 1) & torch.isfinite(values),
                values,
                torch.zeros_like(values),
            )
            correction = torch.nn.functional.linear(
                values * (gates - 1), self.linear.weight
            )
            other.update(
                base_linear_score=linear,
                temporal_score=linear + residual,
                context_correction=correction,
                factor_gate=gates,
                gate_summaries=summaries,
                gate_ready=ready,
                loss_context_shrinkage=self.gate_penalty
                * (
                    (gates - 1).square().sum() / ready.sum().clamp_min(1)
                    + branch_penalty
                ),
                linear_score=linear + correction,
            )
            linear = linear + correction
        return linear + residual, other
