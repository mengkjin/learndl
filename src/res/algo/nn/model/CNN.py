"""Convolutional building blocks: Temporal Convolutional Network (TCN) and
1-D / 2-D ResNet blocks for intra-day feature extraction.
"""
from __future__ import annotations
import torch
from math import isfinite
from numbers import Integral, Real

from torch import nn , Tensor
from torch.nn.utils.parametrizations import weight_norm

from src.proj import Logger
from .. import layer as Layer

__all__ = ['mod_tcn' , 'mod_resnet_1d' , 'mod_resnet_2d']
# 1-d conv resnet
class _tcn_block(nn.Module):
    """Single dilated causal convolution residual block for TCN.

    Architecture: ``WeightNorm-Conv1d → chomp → ReLU → Dropout → (same again)``
    with an optional 1×1 residual projection when ``input_dim != output_dim``.

    Args:
        input_dim:   Number of input channels.
        output_dim:  Number of output channels.
        dilation:    Dilation factor for the causal convolution.
        dropout:     Dropout rate after each activation.
        kernel_size: Convolution kernel size (default ``3``).

    Shapes:
        Input:  ``[bs, input_dim, seq_len]`` (channels-first)
        Output: ``[bs, output_dim, seq_len]``
    """
    def __init__(self, input_dim , output_dim , dilation, dropout=0.0 , kernel_size=3):
        super().__init__()
        padding = (kernel_size-1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(input_dim , output_dim, kernel_size, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(output_dim, output_dim, kernel_size, padding=padding, dilation=dilation))
        self.conv1.weight.data.normal_(0, 0.01) # type: ignore
        self.conv2.weight.data.normal_(0, 0.01) # type: ignore
        
        self.net = nn.Sequential(self.conv1, self._chomp(padding), nn.ReLU(), nn.Dropout(dropout), 
                                 self.conv2, self._chomp(padding), nn.ReLU(), nn.Dropout(dropout))
        
        if input_dim != output_dim:
            self.residual = nn.Conv1d(input_dim , output_dim, 1)
            self.residual.weight.data.normal_(0, 0.01)
        else:
            self.residual = nn.Sequential()
        self.relu = nn.ReLU()

    def forward(self, x : Tensor) -> Tensor:
        output = self.net(x)
        return self.relu(output + self.residual(x))
    
    class _chomp(nn.Module):
        """Remove causal padding from the right end of the time dimension.

        After a causal ``Conv1d`` with ``padding = (kernel-1) * dilation``,
        the output has ``padding`` extra time steps on the right.  This module
        removes them so that the output length matches the input length.
        """
        def __init__(self, padding):
            super().__init__()
            self.padding = padding
        def forward(self, x):
            return x[:, :, :-self.padding] # .contiguous()

class mod_tcn(nn.Module):
    """Temporal Convolutional Network (TCN) sub-module.

    Stacks ``num_layers`` ``_tcn_block`` blocks with exponentially increasing
    dilation (``2^0, 2^1, ..., 2^(num_layers-1)``), giving a receptive field
    of ``O(2^num_layers)``.

    Args:
        input_dim:   Number of input features.
        output_dim:  Number of output features.
        dropout:     Dropout rate.
        num_layers:  Number of TCN blocks (min 2).
        kernel_size: Convolution kernel size (default ``3``).

    Shapes:
        Input:  ``[bs, seq_len, input_dim]``  (channels-last, permuted internally)
        Output: ``[bs, seq_len, output_dim]``
    """
    def __init__(self, input_dim , output_dim , dropout=0.0 , num_layers = 2 , kernel_size = 3):
        super().__init__()
        if kernel_size is None: 
            kernel_size = 3
        num_layers = max(2 , num_layers)
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            inp_d , out_dim = (input_dim , output_dim) if i == 0 else (output_dim , output_dim)
            layers += [_tcn_block(inp_d, out_dim, dilation=dilation, dropout=dropout , kernel_size = kernel_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, x : Tensor) -> Tensor:
        return self.net(x.permute(0,2,1)).permute(0,2,1)
    
def _validate_resnet_dims(**dims):
    for name, value in dims.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer, got {value!r}')


def _resnet_bottleneck_dim(dim_in, dim_out, dim_med, clip_value):
    _validate_resnet_dims(dim_in=dim_in, dim_out=dim_out)
    if dim_med is None:
        dim_med = max(1, dim_out // 4)
    _validate_resnet_dims(dim_med=dim_med)
    if clip_value is not None and (
        isinstance(clip_value, bool) or not isinstance(clip_value, Real)
        or not isfinite(clip_value) or clip_value <= 0
    ):
        raise ValueError(f'clip_value must be None or a finite positive number, got {clip_value!r}')
    return dim_med


def _validate_resnet_input(x, expected_shape):
    if not isinstance(x, Tensor):
        raise TypeError('ResNet input must be a torch.Tensor')
    if x.ndim != len(expected_shape) or any(
        actual < 1 or (expected is not None and actual != expected)
        for actual, expected in zip(x.shape, expected_shape)
    ):
        raise ValueError(f'Expected nonempty ResNet input shape {expected_shape}, got {tuple(x.shape)}')
    if not x.is_floating_point():
        raise TypeError(f'ResNet input must be floating point, got {x.dtype}')


class _resnet_block_1d(nn.Module):
    """Post-activation 1-D bottleneck residual block.

    Branch: ``1×1 Conv → BN → Act → 3×1 Conv → BN → Act → 1×1 Conv → BN``.
    Output: ``Act(shortcut + branch)``. The shortcut is identity when channel
    counts match, otherwise a ``1×1 Conv → BN`` projection.

    ``dim_med=None`` selects ``max(1, dim_out // 4)``. ``clip_value=None``
    leaves the shortcut unchanged; a positive value enables shortcut clipping.
    ``act_type`` accepts the keys/callable factories supported by ``Layer.Act``.
    Input: ``[bs, dim_in, seq_len]``; output: ``[bs, dim_out, seq_len]``.
    """
    def __init__(self, dim_in, dim_out=64, dim_med=None, clip_value=None, act_type='leaky', **kwargs):
        super().__init__()
        dim_med = _resnet_bottleneck_dim(dim_in, dim_out, dim_med, clip_value)
        self.clip_value = clip_value
        self.dim_in = dim_in
        self.dim_med = dim_med
        self.dim_out = dim_out
        self.downsample = nn.Identity() if dim_in == dim_out else nn.Sequential(
            nn.Conv1d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm1d(dim_out),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(dim_in, dim_med, 1, bias=False),
            nn.BatchNorm1d(dim_med),
            Layer.Act.get_activation_fn(act_type),
            nn.Conv1d(dim_med, dim_med, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(dim_med),
            Layer.Act.get_activation_fn(act_type),
            nn.Conv1d(dim_med, dim_out, 1, bias=False),
            nn.BatchNorm1d(dim_out),
        )
        self.act = Layer.Act.get_activation_fn(act_type)

    def forward(self, x: Tensor) -> Tensor:
        _validate_resnet_input(x, (None, self.dim_in, None))
        shortcut = self.downsample(x)
        if self.clip_value is not None:
            shortcut = torch.clamp(shortcut, -self.clip_value, self.clip_value)
        return self.act(shortcut + self.conv(x))


class mod_resnet_1d(nn.Module):
    """Stack 1-D residual blocks, then flatten and linearly project each day.

    Input: ``[bs, n_days, seq_len, feat_len]``; output: ``[bs, n_days, dim_out]``.
    Input axes must follow this order; layouts are never guessed. Training BN
    statistics are shared across samples, days and spatial positions.

    Args:
        seq_len: Positive number of intra-day bars.
        feat_len: Positive number of input features per bar.
        dim_out: Positive output width (default 64).
        clip_value: Optional finite positive shortcut limit; None disables clipping.
        resnet_blocks: Positive number of residual blocks (default 3).
        dim_med: Positive bottleneck width; None selects max(1, dim_out // 4).
        act_type: Activation key or module factory (default 'leaky').

    Extra kwargs are accepted for shared model configuration compatibility.
    """
    def __init__(self, seq_len, feat_len, dim_out=64, clip_value=None, resnet_blocks=3,
                 dim_med=None, act_type='leaky', **kwargs):
        super().__init__()
        _validate_resnet_dims(seq_len=seq_len, feat_len=feat_len, resnet_blocks=resnet_blocks)
        dim_med = _resnet_bottleneck_dim(feat_len, dim_out, dim_med, clip_value)
        self.seq_len = seq_len
        self.dim_in = feat_len
        self.dim_out = dim_out
        self.dim_med = dim_med
        self.clip_value = clip_value
        self.blocks = nn.Sequential()
        d_in = feat_len
        for _ in range(resnet_blocks):
            self.blocks.append(_resnet_block_1d(
                d_in, dim_out, dim_med=dim_med, clip_value=clip_value, act_type=act_type,
            ))
            d_in = dim_out
        self.fc_out = nn.Linear(seq_len * dim_out, dim_out)

    def forward(self, x: Tensor) -> Tensor:
        _validate_resnet_input(x, (None, None, self.seq_len, self.dim_in))
        output = self.blocks(x.reshape(-1, self.seq_len, self.dim_in).permute(0, 2, 1))
        output = self.fc_out(output.flatten(start_dim=1))
        return output.reshape(x.shape[0], x.shape[1], self.dim_out)


class _resnet_block_2d(nn.Module):
    """Post-activation 2-D bottleneck residual block.

    Branch: ``1×1 Conv → BN → Act → 3×3 Conv → BN → Act → 1×1 Conv → BN``.
    Output: ``Act(shortcut + branch)``. The shortcut is identity when channel
    counts match, otherwise a ``1×1 Conv → BN`` projection.

    ``dim_med=None`` selects ``max(1, dim_out // 4)``. ``clip_value=None``
    leaves the shortcut unchanged; a positive value enables shortcut clipping.
    ``act_type`` accepts the keys/callable factories supported by ``Layer.Act``.
    Input: ``[bs, dim_in, seq_len, feat_len]``; output: ``[bs, dim_out, seq_len, feat_len]``.
    """
    def __init__(self, dim_in, dim_out=64, dim_med=None, clip_value=None, act_type='leaky', **kwargs):
        super().__init__()
        dim_med = _resnet_bottleneck_dim(dim_in, dim_out, dim_med, clip_value)
        self.clip_value = clip_value
        self.dim_in = dim_in
        self.dim_med = dim_med
        self.dim_out = dim_out
        self.downsample = nn.Identity() if dim_in == dim_out else nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_med, 1, bias=False),
            nn.BatchNorm2d(dim_med),
            Layer.Act.get_activation_fn(act_type),
            nn.Conv2d(dim_med, dim_med, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_med),
            Layer.Act.get_activation_fn(act_type),
            nn.Conv2d(dim_med, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
        )
        self.act = Layer.Act.get_activation_fn(act_type)

    def forward(self, x: Tensor) -> Tensor:
        _validate_resnet_input(x, (None, self.dim_in, None, None))
        shortcut = self.downsample(x)
        if self.clip_value is not None:
            shortcut = torch.clamp(shortcut, -self.clip_value, self.clip_value)
        return self.act(shortcut + self.conv(x))


class mod_resnet_2d(nn.Module):
    """Stack 2-D residual blocks, then flatten and linearly project each day.

    Input: ``[bs, n_days, seq_len, feat_len]``; output: ``[bs, n_days, dim_out]``.
    Input axes must follow this order; layouts are never guessed. Training BN
    statistics are shared across samples, days and spatial positions.

    Args:
        seq_len: Positive number of intra-day bars.
        feat_len: Positive number of input features per bar.
        dim_out: Positive output width (default 64).
        clip_value: Optional finite positive shortcut limit; None disables clipping.
        resnet_blocks: Positive number of residual blocks (default 3).
        dim_med: Positive bottleneck width; None selects max(1, dim_out // 4).
        act_type: Activation key or module factory (default 'leaky').

    Extra kwargs are accepted for shared model configuration compatibility.
    """
    def __init__(self, seq_len, feat_len, dim_out=64, clip_value=None, resnet_blocks=3,
                 dim_med=None, act_type='leaky', **kwargs):
        super().__init__()
        _validate_resnet_dims(seq_len=seq_len, feat_len=feat_len, resnet_blocks=resnet_blocks)
        dim_med = _resnet_bottleneck_dim(1, dim_out, dim_med, clip_value)
        self.seq_len = seq_len
        self.dim_in = feat_len
        self.dim_out = dim_out
        self.dim_med = dim_med
        self.clip_value = clip_value
        self.blocks = nn.Sequential()
        d_in = 1
        for _ in range(resnet_blocks):
            self.blocks.append(_resnet_block_2d(
                d_in, dim_out, dim_med=dim_med, clip_value=clip_value, act_type=act_type,
            ))
            d_in = dim_out
        self.fc_out = nn.Linear(feat_len * seq_len * dim_out, dim_out)

    def forward(self, x: Tensor) -> Tensor:
        _validate_resnet_input(x, (None, None, self.seq_len, self.dim_in))
        output = self.blocks(x.reshape(-1, 1, self.seq_len, self.dim_in))
        output = self.fc_out(output.flatten(start_dim=1))
        return output.reshape(x.shape[0], x.shape[1], self.dim_out)


if __name__ == '__main__':

    import torch
    import torch.nn as nn
    # pip install pytimedinput -i https://pypi.tuna.tsinghua.edu.cn/simple
    from src.res.algo.nn.model.RNN import mod_gru
    from src.res.algo.nn.model.CNN import mod_resnet_1d, mod_resnet_2d
        
    class resnet1d_gru(nn.Module):
        def __init__(self, seq_len , feat_len , dim_res = 16 , dim_rnn = 64 , **kwargs) -> None:
            super().__init__()
            self.resnet = mod_resnet_1d(seq_len, feat_len, dim_res, resnet_blocks=3)
            self.gru    = mod_gru(dim_res , dim_rnn , 0.1 , 2)
        def forward(self , x):
            hidden = self.resnet(x)
            Logger.stdout("hidden shape :" , hidden.shape)
            output = self.gru(hidden)[:,-1,:]
            Logger.stdout("output shape :" , output.shape)
            return output
        
    class resnet2d_gru(nn.Module):
        def __init__(self, seq_len , feat_len , dim_res = 16 , dim_rnn = 64 , **kwargs) -> None:
            super().__init__()
            self.resnet = mod_resnet_2d(seq_len, feat_len, dim_res, resnet_blocks=3)
            self.gru    = mod_gru(dim_res , dim_rnn , 0.1 , 2)
        def forward(self , x):
            hidden = self.resnet(x)
            output = self.gru(hidden)[:,-1,:]
            Logger.stdout("hidden shape :" , hidden.shape)
            Logger.stdout("output shape :" , output.shape)
            return output
        

    batch_n  = 2
    seq_day  = 30
    seq_inday  = 8
    feat_len = 5
    dim_out  = 64
    x = torch.randn(batch_n,seq_day,seq_inday,feat_len)

    net_1d = resnet1d_gru(seq_inday , feat_len , dim_res = dim_out // 4 , dim_rnn = dim_out)
    net_2d = resnet2d_gru(seq_inday , feat_len , dim_res = dim_out // 4 , dim_rnn = dim_out)
    y1 = net_1d(x)
    y2 = net_2d(x)
