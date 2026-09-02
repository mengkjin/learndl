"""
Transformer-GRU model.
Encode intra-day bars with a Transformer, then process day sequence with GRU.
"""

from __future__ import annotations

from torch import nn, Tensor

from .. import layer as Layer
from .Attention import TimeWiseAttention, mod_transformer
from .RNN import mod_gru


class transformer_gru(nn.Module):
    """GRU with intra-day Transformer encoder.  Registry key: ``'transformer_gru'``.

    ``enc_in_dim`` must be divisible by 8 (Transformer ``num_heads=8``).
    Intra-day bar length is taken from the input tensor shape.
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
        **kwargs,
    ):
        super().__init__()
        if enc_in_dim is None:
            enc_in_dim = hidden_dim
        assert enc_in_dim % 8 == 0, f'enc_in_dim must be divisible by 8, got {enc_in_dim}'

        # Intra-day: [bs*days, bars, feat] → [bs*days, bars, enc_in_dim] → last bar
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
        # shape: (bs, days, bars, feat) → (bs*days, bars, feat)
        bs, days = x.shape[:2]
        x = x.reshape(bs * days, *x.shape[2:])
        x = self.fc_enc_in(x)[:, -1]  # shape: (bs*days, enc_in_dim)
        x = x.reshape(bs, days, -1)  # shape: (bs, days, enc_in_dim)

        x = self.fc_rnn(x)
        x = self.fc_enc_att(x) if self.fc_enc_att is not None else x[:, -1]
        x = self.fc_hid_out(x)
        o = self.fc_map_out(x)
        return o, {'hidden': x}
