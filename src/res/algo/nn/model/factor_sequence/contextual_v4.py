"""Shared factor-performance sequences and point-in-time FiLM corrections."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .residual import FactorResidual

__all__ = ["FactorContextualV4", "V4_VARIANTS"]

V4_VARIANTS = ("shared", "private", "graph")


def _masked_mean_square(value: Tensor, ready: Tensor) -> Tensor:
    masked = torch.where(ready[:, None], value, 0.0)
    return masked.square().sum() / (ready.sum().clamp_min(1) * value.shape[-1])


class _FactorGraph(nn.Module):
    def __init__(self, factor_count: int):
        super().__init__()
        self.identity = nn.Parameter(torch.empty(factor_count, 4))
        nn.init.normal_(self.identity, std=0.02)
        self.identity_project = nn.Linear(4, 16, bias=False)
        self.norm = nn.LayerNorm(16)
        self.query = nn.Linear(16, 8, bias=False)
        self.key = nn.Linear(16, 8, bias=False)
        self.value = nn.Linear(16, 16, bias=False)
        self.output = nn.Linear(16, 16, bias=False)

    def forward(
        self, state: Tensor, ready: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        identity_delta = 0.1 * torch.tanh(self.identity_project(self.identity))
        nodes = self.norm(state + identity_delta)
        scores = self.query(nodes) @ self.key(nodes).T / math.sqrt(8)
        eligible = ready[:, None] & ready[None, :]
        eligible = eligible & ~torch.eye(
            len(ready), dtype=torch.bool, device=ready.device
        )
        # For empty rows use finite logits during softmax, then zero every edge.
        scores = scores.masked_fill(~eligible, -torch.inf)
        scores = torch.where(eligible.any(-1, keepdim=True), scores, 0.0)
        weights = scores.softmax(-1).masked_fill(~eligible, 0.0)
        graph_delta = 0.1 * torch.tanh(self.output(weights @ self.value(nodes)))
        return (
            state + identity_delta + graph_delta,
            identity_delta,
            graph_delta,
            weights,
        )


class FactorContextualV4(FactorResidual):
    """One-date stock batch and one shared, ordered 60-day performance stream."""

    def __init__(
        self,
        input_dim,
        variant="shared",
        hidden_dim=32,
        dropout=0.1,
        rnn_layers=2,
        context_dropout=0.1,
        min_observations=20,
        context_penalty=0.01,
        **kwargs,
    ):
        dims = [input_dim] if isinstance(input_dim, int) else list(input_dim)
        if len(dims) != 2 or dims[0] <= 0 or dims[0] % 4 or dims[0] != dims[1]:
            raise ValueError(
                "v4 needs compact stock and paired IC/spread in matching factor order"
            )
        if variant not in V4_VARIANTS:
            raise ValueError(f"Unknown v4 variant: {variant}")
        if (
            not 1 <= min_observations <= 60
            or not math.isfinite(context_penalty)
            or context_penalty < 0
        ):
            raise ValueError("Invalid v4 performance maturity or penalty")
        super().__init__(
            dims[0],
            mode="temporal",
            hidden_dim=hidden_dim,
            dropout=dropout,
            rnn_layers=rnn_layers,
        )
        self.variant = variant
        self.min_observations = min_observations
        self.context_penalty = context_penalty
        self.shared_gru = nn.GRU(4, 16, batch_first=True)
        if variant == "private":
            self.private_grus = nn.ModuleList(
                nn.GRU(4, 4, batch_first=True) for _ in range(self.k)
            )
            self.private_project = nn.Linear(4, 16, bias=False)
        if variant == "graph":
            self.factor_graph = _FactorGraph(self.k)
        self.context_dropout = nn.Dropout(context_dropout)
        self.value_encoder = nn.Linear(4, 8)
        self.factor_film = nn.Linear(16, 16)
        nn.init.zeros_(self.factor_film.weight)
        nn.init.zeros_(self.factor_film.bias)
        self.readout = nn.Linear(hidden_dim + 8, 8)
        self.readout_output = nn.Linear(8, 1, bias=False)

    def _encode_performance(
        self, performance: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        records = performance[0].reshape(60, self.k, 2, 2)
        values = records[..., 0]
        valid = (records[..., 1] == 1) & torch.isfinite(values)
        ready = (valid.sum(0) >= self.min_observations).all(-1)
        packed = torch.stack(
            (torch.where(valid, values, 0.0), valid.to(values.dtype)), dim=-1
        )
        sequences = packed.flatten(2).transpose(0, 1).contiguous()
        state = self.shared_gru(sequences)[1][0]
        details: dict[str, Tensor] = {}
        if self.variant == "private":
            private = torch.cat(
                [
                    gru(sequences[i : i + 1])[1][0]
                    for i, gru in enumerate(self.private_grus)
                ],
                dim=0,
            )
            correction = 0.2 * torch.tanh(self.private_project(private))
            state = state + correction
            details["private_delta"] = correction
        elif self.variant == "graph":
            state, identity, correction, attention = self.factor_graph(state, ready)
            details.update(
                identity_delta=identity,
                graph_delta=correction,
                factor_attention=attention,
            )
        return state, ready, details

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("v4 requires exactly stock and shared performance streams")
        stock, performance = (v.squeeze(2) if v.ndim == 4 else v for v in x)
        if (
            stock.ndim != 3
            or stock.shape[1:] != (60, 4 * self.k)
            or performance.shape != (1, 60, 4 * self.k)
        ):
            raise ValueError("v4 expects [stocks,60,4*factors] and [1,60,4*factors]")
        temporal, other = super().forward(stock)
        state, ready, details = self._encode_performance(performance)
        gamma, beta = self.factor_film(self.context_dropout(state)).tanh().chunk(2, -1)
        gamma = torch.where(ready[:, None], 0.2 * gamma, 0.0)
        beta = torch.where(ready[:, None], 0.1 * beta, 0.0)

        current = stock[:, -1].reshape(-1, 4, self.k).transpose(1, 2)
        valid = (current[..., 1] == 1) & torch.isfinite(current[..., 0])
        current = torch.where(valid[..., None], current, 0.0)
        value = F.gelu(self.value_encoder(current))
        modulated = (1 + gamma) * value + beta
        # Split the same affine readout into stock and factor terms so we never
        # materialize two concatenated [stocks, factors, hidden_dim + 8] inputs.
        hidden = other["hidden"]
        stock_weight, value_weight = self.readout.weight.split(
            (hidden.shape[-1], 8), dim=-1
        )
        stock_term = F.linear(hidden, stock_weight, self.readout.bias)[:, None, :]
        before = F.gelu(stock_term + F.linear(value, value_weight))
        after = F.gelu(stock_term + F.linear(modulated, value_weight))
        by_factor = self.readout_output(after - before).squeeze(-1)
        correction = torch.where(valid & ready, by_factor, 0.0).mean(
            dim=1, keepdim=True
        )

        shrinkage = _masked_mean_square(torch.cat((gamma, beta), dim=-1), ready)
        for key in ("private_delta", "identity_delta", "graph_delta"):
            if key in details:
                shrinkage = shrinkage + _masked_mean_square(details[key], ready)
        other.update(
            temporal_score=temporal,
            context_correction=correction,
            residual_score=other["residual_score"] + correction,
            performance_ready=ready,
            film_gamma=gamma,
            film_beta=beta,
            loss_context_shrinkage=self.context_penalty * shrinkage,
            **details,
        )
        return temporal + correction, other
