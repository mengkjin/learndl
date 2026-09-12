"""
Transformer-GRU model.
Encode intra-day bars with a Transformer, then process day sequence with GRU.
"""

from __future__ import annotations

import torch
from torch import nn, Tensor

from .. import layer as Layer
from .Attention import TimeWiseAttention, mod_transformer
from .RNN import mod_gru


class transformer_gru(nn.Module):
    """GRU with intra-day Transformer encoder.  Registry key: ``'transformer_gru'``.

    ``enc_in_dim`` must be divisible by 8 (``num_heads = enc_in_dim // 8``, ``head_dim=8``).
    Intra-day bar length is taken from the input tensor shape.
    The same Transformer encodes each day independently; GRU alone models
    the sequence of days, matching the ResNet-GRU baseline.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=2**6,
        dropout=0.1,
        enc_in_dim=None,
        enc_att=False,
        rnn_layers=2,
        enc_layers=2,
        rnn_type='gru',
        **kwargs,
    ):
        super().__init__()
        if rnn_type != 'gru':
            raise ValueError(f'only gru is supported, got {rnn_type}')
        if enc_in_dim is None:
            enc_in_dim = hidden_dim
        assert enc_in_dim % 8 == 0, f'enc_in_dim must be divisible by 8, got {enc_in_dim}'

        # Shared intra-day encoder: [bs, bars, feat] → [bs, bars, enc_in_dim].
        self.fc_enc_in = mod_transformer(
            input_dim=input_dim,
            output_dim=enc_in_dim,
            dropout=dropout,
            num_layers=enc_layers,
        )

        self.fc_rnn = mod_gru(
            input_dim=enc_in_dim,
            output_dim=hidden_dim,
            num_layers=rnn_layers,
            dropout=dropout,
        )
        self.fc_enc_att = (
            TimeWiseAttention(hidden_dim, hidden_dim, dropout=dropout) if enc_att else None
        )
        self.fc_hid_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim))
        self.fc_map_out = nn.Sequential(Layer.MeanPool(), nn.BatchNorm1d(1))

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        """
        in:  [bs x days x bars x feat]
        out: [bs x 1], {'hidden': [bs x hidden_dim]}
        """
        if x.ndim != 4 or x.size(1) == 0 or x.size(2) == 0:
            raise ValueError(f'expected [batch, days, bars, features] with nonempty days/bars, got {tuple(x.shape)}')
        # Each day's 16 bars attend only to that day. The final bar's
        # contextual representation summarizes all bars (non-causal attention).
        # Keep days out of the attention batch to avoid stocks * days CUDA grids.
        x = torch.stack([
            self.fc_enc_in(day.contiguous())[:, -1]
            for day in x.unbind(dim=1)
        ], dim=1)  # [bs, 30, enc_in_dim] in chronological order

        x = self.fc_rnn(x)
        x = self.fc_enc_att(x) if self.fc_enc_att is not None else x[:, -1]
        x = self.fc_hid_out(x)
        o = self.fc_map_out(x)
        return o, {'hidden': x}
