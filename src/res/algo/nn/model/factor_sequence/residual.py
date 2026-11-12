"""Compact residual factor models; context conditions the temporal encoder only."""
from __future__ import annotations

__all__ = ['FactorResidual']

import torch
from torch import nn


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
        **kwargs,
    ):
        super().__init__()
        dims = [input_dim] if isinstance(input_dim, int) else list(input_dim)
        if (
            mode not in ("snapshot", "temporal", "contextual")
            or len(dims) != (3 if mode == "contextual" else 1)
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

    def forward(self, x):
        streams = list(x) if isinstance(x, (list, tuple)) else [x]
        streams = [v.squeeze(2) if v.ndim == 4 else v for v in streams]
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
        return linear + residual, dict(
            hidden=hidden, linear_score=linear, residual_score=residual
        )
