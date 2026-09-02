"""
ResNet-MLP model.
Encode intra-day bars with ResNet-1D, then mix the day sequence with a temporal MLP.
"""

from __future__ import annotations

from torch import nn, Tensor

from .. import layer as Layer
from .Attention import TimeWiseAttention
from .CNN import mod_resnet_1d


class _temporal_mlp(nn.Module):
    """Time-mix then feature-MLP over a day sequence.

    Time mix: Linear over the day axis (fixed ``seq_len``).
    Feature MLP: ``num_layers`` Linear+Act+Dropout blocks to ``output_dim``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        num_layers: int = 2,
        act_type: str = 'leaky',
    ):
        super().__init__()
        self.time_mix = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            Layer.Act.get_activation_fn(act_type),
            nn.Dropout(dropout),
        )
        layers: list[nn.Module] = []
        d_in = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(d_in, output_dim),
                Layer.Act.get_activation_fn(act_type),
                nn.Dropout(dropout),
            ])
            d_in = output_dim
        self.feat_mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # shape: (bs, days, feat) → time mix on days → feature MLP
        x = self.time_mix(x.transpose(1, 2)).transpose(1, 2)
        return self.feat_mlp(x)


class resnet_mlp(nn.Module):
    """Temporal MLP with intra-day ResNet-1D encoder.  Registry key: ``'resnet_mlp'``.

    Requires ``inday_dim`` and ``seq_len`` (number of days) in kwargs.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=2**6,
        dropout=0.1,
        act_type='leaky',
        enc_in_dim=None,
        enc_att=False,
        rnn_layers=2,
        **kwargs,
    ):
        super().__init__()
        if enc_in_dim is None:
            enc_in_dim = hidden_dim

        seq_len = kwargs['seq_len']
        res_kwargs = {k: v for k, v in kwargs.items() if k != 'seq_len'}
        self.fc_enc_in = mod_resnet_1d(kwargs['inday_dim'], input_dim, enc_in_dim, **res_kwargs)

        self.fc_day = _temporal_mlp(
            input_dim=enc_in_dim,
            output_dim=hidden_dim,
            seq_len=seq_len,
            dropout=dropout,
            num_layers=rnn_layers,
            act_type=act_type,
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
