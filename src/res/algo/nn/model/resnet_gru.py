"""
ResNet-GRU model.
Use resnet to encode the input features, and then use gru to process the features.
"""

from __future__ import annotations

__all__ = [
    'resnet_gru',
    'resnet_gru_new',
    'gru_gru',
    'gru_gru_new',
]
from torch import nn , Tensor, cat

from .. import layer as Layer
from ..layer.checkpoint import ActivationCheckpointMixin
from .Attention import TimeWiseAttention
from .CNN import mod_resnet_1d
from .RNN import mod_gru

class resnet_gru(nn.Module):
    """GRU with intra-day ResNet-1D encoder.  Registry key: ``'resnet_gru'``.

    Requires ``inday_dim`` (intra-day bar count) in kwargs.
    """
    def __init__(
        self,
        input_dim ,
        hidden_dim      = 2**6,
        dropout         = 0.1,
        act_type        = 'leaky',
        enc_in          = None,
        enc_in_dim      = None ,
        enc_att         = False,
        rnn_type        = 'gru',
        rnn_layers      = 2,
        loss_corr_lamb = 0.1,
        hidden_as_factors = True,
        hidden_mean_pool = False,
        **kwargs
    ):
        super().__init__()
        assert rnn_type == 'gru' , f'only gru is supported : {rnn_type}'
        self.loss_corr_lamb = loss_corr_lamb

        if enc_in_dim is None:
            enc_in_dim = hidden_dim
        
        res_kwargs = {k:v for k,v in kwargs.items() if k != 'seq_len'}
        self.fc_enc_in = mod_resnet_1d(kwargs['inday_dim'] , input_dim , enc_in_dim, act_type=act_type, **res_kwargs)

        self.fc_rnn = mod_gru(
            input_dim = enc_in_dim,
            output_dim = hidden_dim,
            num_layers = rnn_layers,
            dropout = dropout,
        )

        if enc_att:
            self.fc_enc_att = TimeWiseAttention(hidden_dim,hidden_dim,dropout=dropout) 
        else:
            self.fc_enc_att = None

        self.output = Layer.basic.OutputLayer(hidden_dim, hidden_as_factors, hidden_mean_pool)
        
    def forward(self , x : Tensor) -> tuple[Tensor,dict]:
        """
        in: [bs x seq_len x inday_dim x input_dim]
        out:[bs x 1] , [bs x hidden_dim]
        """
        x = self.fc_enc_in(x)
        x = self.fc_rnn(x)
        x = self.fc_enc_att(x) if self.fc_enc_att is not None else x[:,-1]
        x, o = self.output(x)
        return o , {'hidden' : x}


class resnet_gru_new(nn.Module):
    """Optional per-bar projection, daily ResNet encoding and cross-day GRU.

    enc_in: True/'linear' enables Linear + Tanh; False/None disables it.
    enc_in_dim: projection width (default hidden_dim // 4, minimum 1).
    inday_resnet_dim: daily ResNet output and GRU input width (default hidden_dim).
    hidden_dim: cross-day GRU output width. dim_med controls the ResNet
    bottleneck independently (None uses inday_resnet_dim // 4, minimum 1).
    Input is [stocks, days, inday_dim, input_dim]. ResNet BatchNorm retains
    full stock/day statistics; no chunking or recomputation is applied here.
    """

    def __init__(self, input_dim, hidden_dim=64, dropout=.1, act_type='leaky',
                 enc_in: bool | str | None = True, enc_in_dim=None, inday_resnet_dim=None,
                 enc_att=False, rnn_type='gru', rnn_layers=2,
                 loss_corr_lamb=.1, hidden_as_factors=True, hidden_mean_pool=False,
                 *, inday_dim, resnet_blocks=3, dim_med=None, clip_value=None, **kwargs):
        super().__init__()
        if rnn_type != 'gru':
            raise ValueError('resnet_gru_new supports only rnn_type=gru')
        if enc_in is not None and enc_in is not True and enc_in is not False and enc_in != 'linear':
            raise ValueError("enc_in must be True, 'linear', False or None")
        enc_in_dim = max(hidden_dim // 4, 1) if enc_in_dim is None else enc_in_dim
        inday_resnet_dim = hidden_dim if inday_resnet_dim is None else inday_resnet_dim
        for name, value in dict(input_dim=input_dim, hidden_dim=hidden_dim,
                                enc_in_dim=enc_in_dim, inday_resnet_dim=inday_resnet_dim,
                                rnn_layers=rnn_layers).items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if rnn_layers > 3:
            raise ValueError('rnn_layers must be <= 3 (mod_gru limit)')
        self.loss_corr_lamb = loss_corr_lamb
        project = enc_in is True or enc_in == 'linear'
        self.fc_enc_in = (nn.Sequential(nn.Linear(input_dim, enc_in_dim), nn.Tanh())
                          if project else nn.Identity())
        self.inday_resnet = mod_resnet_1d(
            inday_dim, enc_in_dim if project else input_dim, inday_resnet_dim,
            resnet_blocks=resnet_blocks, dim_med=dim_med, act_type=act_type,
            clip_value=clip_value,
        )
        self.fc_rnn = mod_gru(inday_resnet_dim, hidden_dim, num_layers=rnn_layers, dropout=dropout)
        self.fc_enc_att = TimeWiseAttention(hidden_dim, hidden_dim, dropout=dropout) if enc_att else None
        self.output = Layer.basic.OutputLayer(hidden_dim, hidden_as_factors, hidden_mean_pool)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        x = self.inday_resnet(self.fc_enc_in(x))
        x = self.fc_rnn(x)
        x = self.fc_enc_att(x) if self.fc_enc_att is not None else x[:, -1]
        hidden, prediction = self.output(x)
        return prediction, {'hidden': hidden}


class gru_gru(ActivationCheckpointMixin, nn.Module):
    """GRU with intra-day GRU encoder.  Registry key: ``'gru_gru'``.

    Requires ``inday_dim`` (intra-day bar count) in kwargs.
    Both GRU stages support activation checkpointing; the BatchNorm output
    head stays outside recomputation. Parameter names and batch layout remain
    compatible with existing weights.
    """
    activation_checkpoint_regions = ('encode_intraday_chunk', 'encode_days')
    intraday_chunk_size = 12500

    def __init__(
        self,
        input_dim ,
        hidden_dim      = 2**6,
        dropout         = 0.1,
        act_type        = 'leaky',
        enc_in          = None,
        enc_in_dim      = None ,
        enc_att         = False,
        rnn_type        = 'gru',
        rnn_layers      = 2,
        loss_corr_lamb  = 0.1,
        hidden_as_factors = True,
        hidden_mean_pool = False,
        **kwargs
    ):
        super().__init__()
        self.loss_corr_lamb = loss_corr_lamb

        if enc_in_dim is None:
            enc_in_dim = hidden_dim
        
        self.fc_enc_in = mod_gru(input_dim , enc_in_dim, num_layers=rnn_layers, dropout=dropout)

        self.fc_rnn = mod_gru(
            input_dim = enc_in_dim,
            output_dim = hidden_dim,
            num_layers = rnn_layers,
            dropout = dropout,
        )

        if enc_att:
            self.fc_enc_att = TimeWiseAttention(hidden_dim,hidden_dim,dropout=dropout) 
        else:
            self.fc_enc_att = None
        
        self.output = Layer.basic.OutputLayer(hidden_dim, hidden_as_factors, hidden_mean_pool)
        
    def encode_intraday_chunk(self, x: Tensor) -> Tensor:
        """Return compact last-step outputs without detaching their gradients."""
        return self.fc_enc_in(x)[:, -1].clone()

    def encode_intraday(self, x: Tensor) -> Tensor:
        """Bound each GRU call; checkpoint each chunk independently when active."""
        bs, days, bars, features = x.shape
        flat = x.reshape(-1, bars, features)
        encoded = cat([
            self.checkpoint_region('encode_intraday_chunk', part)
            for part in flat.split(self.intraday_chunk_size)
        ], dim=0)
        return encoded.reshape(bs, days, -1)

    def encode_days(self, x: Tensor) -> Tensor:
        """Encode days and pool before the stateful output head."""
        x = self.fc_rnn(x)
        return self.fc_enc_att(x) if self.fc_enc_att is not None else x[:, -1]

    def forward(self , x : Tensor) -> tuple[Tensor,dict]:
        """
        in: [bs x seq_len x inday_dim x input_dim]
        out:[bs x 1] , [bs x hidden_dim]
        """
        x = self.encode_intraday(x)
        x = self.checkpoint_region('encode_days', x)
        x, o = self.output(x)
        return o , {'hidden' : x}

class gru_gru_new(ActivationCheckpointMixin, nn.Module):
    """GRU with intra-day GRU encoder.  Registry key: ``'gru_gru'``.

    Requires ``inday_dim`` (intra-day bar count) in kwargs.
    Both GRU stages support activation checkpointing; the BatchNorm output
    head stays outside recomputation. Parameter names and batch layout remain
    compatible with existing weights.
    """
    activation_checkpoint_regions = ('encode_intraday_chunk', 'encode_days')
    intraday_chunk_size = 12500

    def __init__(
        self,
        input_dim ,
        hidden_dim      = 2**6,
        dropout         = 0.1,
        act_type        = 'leaky',
        enc_in          = None,
        enc_in_dim      = None ,
        inday_rnn_dim   = None,
        rnn_layers      = 2,
        loss_corr_lamb  = 0.1,
        hidden_as_factors = True,
        hidden_mean_pool = False,
        **kwargs
    ):
        super().__init__()
        self.loss_corr_lamb = loss_corr_lamb

        if enc_in_dim is None:
            enc_in_dim = max(hidden_dim // 4 , 1)

        if inday_rnn_dim is None:
            inday_rnn_dim = enc_in_dim

        if enc_in == 'linear' or enc_in is True:
            self.fc_enc_in = nn.Sequential(nn.Linear(input_dim, enc_in_dim),nn.Tanh())
        else:
            enc_in_dim = input_dim
            self.fc_enc_in = nn.Sequential()
        
        self.inday_rnn = mod_gru(enc_in_dim , inday_rnn_dim, num_layers=rnn_layers, dropout=dropout)

        self.fc_rnn = mod_gru(
            input_dim = enc_in_dim,
            output_dim = hidden_dim,
            num_layers = rnn_layers,
            dropout = dropout,
        )
        
        self.output = Layer.basic.OutputLayer(hidden_dim, hidden_as_factors, hidden_mean_pool)

    def encode_in(self, x: Tensor) -> Tensor:
        """Encode the input features."""
        return self.fc_enc_in(x)
        
    def encode_intraday_chunk(self, x: Tensor) -> Tensor:
        """Include feature projection in the independently recomputed chunk."""
        return self.inday_rnn(self.encode_in(x))[:, -1].clone()

    def encode_intraday(self, x: Tensor) -> Tensor:
        """Preserve stock/day ordering while bounding each intraday GRU call."""
        bs, days, bars, features = x.shape
        flat = x.reshape(-1, bars, features)
        encoded = cat([
            self.checkpoint_region('encode_intraday_chunk', part)
            for part in flat.split(self.intraday_chunk_size)
        ], dim=0)
        return encoded.reshape(bs, days, -1)

    def encode_days(self, x: Tensor) -> Tensor:
        """Encode days and pool before the stateful output head."""
        x = self.fc_rnn(x)
        return x[:, -1]

    def forward(self , x : Tensor) -> tuple[Tensor,dict]:
        """
        in: [bs x seq_len x inday_dim x input_dim]
        out:[bs x 1] , [bs x hidden_dim]
        """
        x = self.encode_intraday(x)
        x = self.checkpoint_region('encode_days', x)
        x, o = self.output(x)
        return o , {'hidden' : x}