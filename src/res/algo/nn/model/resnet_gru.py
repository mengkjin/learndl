"""
ResNet-GRU model.
Use resnet to encode the input features, and then use gru to process the features.
"""

from __future__ import annotations

__all__ = [
    'resnet_gru',
    'resnet_gru_new',
    'resnet_gru_simp',
    'gru_gru',
    'gru_gru_new',
]
from torch import nn , Tensor, cat

from .. import layer as Layer
from ..layer.checkpoint import ActivationCheckpointMixin
from .Attention import TimeWiseAttention
from .CNN import mod_resnet_1d, _validate_resnet_dims, _validate_resnet_input
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
        res_kwargs.setdefault('resnet_pool', 'linear')
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

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # The original network stored its flatten projection and both output heads directly.
        names = {'fc_hid_out.': 'output.fc_hid_out.', 'fc_map_out.': 'output.fc_map_out.'}
        if self.fc_enc_in.resnet_pool == 'flatten':
            names['fc_enc_in.fc_out.'] = 'fc_enc_in.fc_out.net.'
        for old, new in names.items():
            for key in list(state_dict):
                if key.startswith(prefix + old) and not key.startswith(prefix + new):
                    target = prefix + new + key.removeprefix(prefix + old)
                    if target not in state_dict:
                        state_dict[target] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        
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
            resnet_pool=kwargs.get('resnet_pool', 'linear'),
            resnet_pool_attn_dim=kwargs.get('resnet_pool_attn_dim'),
            resnet_pool_conv_layers=kwargs.get('resnet_pool_conv_layers', 1),
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


class _IdentityIntradayResidual(nn.Module):
    """Two same-shape convolutions with a strictly identity skip connection."""

    def __init__(self, channels, kernel_size=3, act_type='leaky'):
        super().__init__()
        _validate_resnet_dims(channels=channels, resnet_kernel_size=kernel_size)
        if kernel_size % 2 != 1:
            raise ValueError('resnet_kernel_size must be odd')
        self.channels = channels
        self.branch = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2,
                      padding_mode='replicate'),
            Layer.Act.get_activation_fn(act_type),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2,
                      padding_mode='replicate'),
        )
        self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _validate_resnet_input(x, (None, self.channels, None))
        return self.shortcut(x) + self.branch(x)


class resnet_gru_simp(nn.Module):
    """Shared optional bar projection -> identity residual -> daily Linear -> GRU.

    Both convolutions preserve the projected input space and replicate boundary
    values. Flatten/Linear occurs AFTER residual addition. The last GRU output
    feeds a factor head (Linear + BN), then equal weighting + BN by default.
    Input [stocks, days, bars, features]; output (prediction, {'hidden': factors}).
    """

    def __init__(self, input_dim, hidden_dim=64, dropout=.1, act_type='leaky',
                 enc_in: bool | str | None = False, enc_in_dim=16,
                 inday_resnet_dim=128, rnn_layers=1, rnn_type='gru',
                 hidden_as_factors=True, hidden_mean_pool=True,
                 *, inday_dim, resnet_kernel_size=3, **kwargs):
        super().__init__()
        if rnn_type != 'gru':
            raise ValueError('resnet_gru_simp supports only rnn_type=gru')
        if enc_in is not None and enc_in is not True and enc_in is not False and enc_in != 'linear':
            raise ValueError("enc_in must be True, 'linear', False or None")
        _validate_resnet_dims(input_dim=input_dim, hidden_dim=hidden_dim,
                              inday_dim=inday_dim, inday_resnet_dim=inday_resnet_dim,
                              rnn_layers=rnn_layers)
        if rnn_layers > 3:
            raise ValueError('rnn_layers must be <= 3')
        if not 0 <= dropout <= 1:
            raise ValueError('dropout must be between 0 and 1')
        project = enc_in is True or enc_in == 'linear'
        if project:
            _validate_resnet_dims(enc_in_dim=enc_in_dim)
        channels = enc_in_dim if project else input_dim
        self.input_dim = input_dim
        self.inday_dim = inday_dim
        self.fc_enc_in = (nn.Sequential(nn.Linear(input_dim, channels), nn.Tanh())
                          if project else nn.Identity())
        self.inday_resnet = _IdentityIntradayResidual(channels, resnet_kernel_size, act_type)
        self.daily_projection = nn.Linear(inday_dim * channels, inday_resnet_dim)
        self.fc_rnn = mod_gru(inday_resnet_dim, hidden_dim, num_layers=rnn_layers,
                             dropout=dropout if rnn_layers > 1 else 0.)
        self.output = Layer.basic.OutputLayer(hidden_dim, hidden_as_factors, hidden_mean_pool)

    def encode_intraday(self, x: Tensor) -> Tensor:
        _validate_resnet_input(x, (None, None, self.inday_dim, self.input_dim))
        batch, days = x.shape[:2]
        x = self.fc_enc_in(x)
        x = x.reshape(batch * days, self.inday_dim, -1).transpose(1, 2)
        x = self.inday_resnet(x)
        return self.daily_projection(x.flatten(1)).reshape(batch, days, -1)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        x = self.fc_rnn(self.encode_intraday(x))[:, -1]
        factors, prediction = self.output(x)
        return prediction, {'hidden': factors}


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
