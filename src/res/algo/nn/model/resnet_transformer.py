"""
ResNet-Transformer model.
Encode intra-day bars with ResNet-1D, then process day sequence with Transformer.
"""

from __future__ import annotations

from torch import nn, Tensor

from .. import layer as Layer
from .Attention import TimeWiseAttention, mod_transformer
from .CNN import mod_resnet_1d


class resnet_transformer(nn.Module):
    """Transformer with intra-day ResNet-1D encoder.  Registry key: ``'resnet_transformer'``.

    Requires ``inday_dim`` in kwargs.  ``hidden_dim`` must be divisible by 8.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=2**6,
        dropout=0.1,
        enc_in_dim=None,
        enc_att=False,
        rnn_layers=2,
        **kwargs,
    ):
        super().__init__()
        assert hidden_dim % 8 == 0, f'hidden_dim must be divisible by 8, got {hidden_dim}'
        if enc_in_dim is None:
            enc_in_dim = hidden_dim

        res_kwargs = {k: v for k, v in kwargs.items() if k != 'seq_len'}
        self.fc_enc_in = mod_resnet_1d(kwargs['inday_dim'], input_dim, enc_in_dim, **res_kwargs)

        self.fc_day = mod_transformer(
            input_dim=enc_in_dim,
            output_dim=hidden_dim,
            dropout=dropout,
            num_layers=rnn_layers,
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
        x = self.fc_enc_in(x)  # shape: (bs, days, enc_in_dim)
        x = self.fc_day(x)  # shape: (bs, days, hidden_dim)
        x = self.fc_enc_att(x) if self.fc_enc_att is not None else x[:, -1]
        x = self.fc_hid_out(x)
        o = self.fc_map_out(x)
        return o, {'hidden': x}
