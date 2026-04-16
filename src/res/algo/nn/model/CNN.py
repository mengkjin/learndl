"""Convolutional building blocks: Temporal Convolutional Network (TCN) and
1-D / 2-D ResNet blocks for intra-day feature extraction.
"""
import torch

from torch import nn , Tensor
from torch.nn.utils.parametrizations import weight_norm

from src.proj import Logger
from .. import layer as Layer

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
    
class _resnet_block_1d(nn.Module):
    """Bottleneck 1-D ResNet block with optional clipping.

    Architecture: ``1×1 Conv → BN → Act → 3×1 Conv → BN → Act → 1×1 Conv → BN → Act``
    with a residual 1×1 projection when ``dim_in != dim_out``.

    Args:
        dim_in:     Number of input channels.
        dim_out:    Number of output channels (default ``64``).
        dim_med:    Bottleneck intermediate channels (default ``dim_out // 4``).
        clip_value: Clamp residual shortcut to ``[-clip_value, clip_value]``
                    to prevent exploding activations (default ``10``).
        act_type:   Activation function key (default ``'leaky'``).

    Shapes:
        Input:  ``[bs, dim_in, seq_len]``  (channels-first)
        Output: ``[bs, dim_out, seq_len]``
    """
    def __init__(self, dim_in , dim_out = 64 , dim_med = 64 // 4 , clip_value = 10 , act_type = 'leaky' , **kwargs) -> None:
        super().__init__()
        self.clip_value = clip_value
        self.dim_in     = dim_in
        if dim_in == dim_out:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Conv1d(dim_in , dim_out , 1)

        self.conv = nn.Sequential(
            nn.Conv1d(dim_in , dim_med , 1) , 
            nn.BatchNorm1d(dim_med),
            Layer.Act.get_activation_fn(act_type) ,
            nn.Conv1d(dim_med, dim_med , kernel_size=3, stride=1, padding=1 , bias = False) , 
            nn.BatchNorm1d(dim_med),
            Layer.Act.get_activation_fn(act_type) ,
            nn.Conv1d(dim_med, dim_out , 1) , 
            nn.BatchNorm1d(dim_out),
            Layer.Act.get_activation_fn(act_type) ,
        )

    def forward(self , x : Tensor) -> Tensor:
        if x.shape[-2] != self.dim_in and x.shape[-1] == self.dim_in: 
            Logger.alert1('auto permute!')
            x = x.permute(0,2,1) 
        x1 = torch.clip(self.downsample(x) , -self.clip_value , self.clip_value)
        x2 = self.conv(x)
        return x1 + x2
    
class mod_resnet_1d(nn.Module):
    """Stacked 1-D ResNet blocks for intra-day feature extraction.

    Applies ``resnet_blocks`` bottleneck residual blocks to a per-sample
    feature sequence, then flattens and projects to ``dim_out``.  Used as an
    intra-day encoder in ``resnet_lstm``/``resnet_gru`` where each date's
    intra-day bar sequence is processed independently.

    Args:
        seq_len:       Intra-day sequence length (e.g. number of bars per day).
        feat_len:      Number of input features per bar.
        dim_out:       Output dimension (default ``64``).
        clip_value:    Residual clipping threshold (default ``10``).
        resnet_blocks: Number of residual blocks (default ``3``).

    Shapes:
        Input:  ``[bs, n_days, seq_len, feat_len]``
        Output: ``[bs, n_days, dim_out]``
    """
    def __init__(self, seq_len , feat_len , dim_out = 64 , clip_value = 10 , resnet_blocks = 3 , **kwargs) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.dim_in  = feat_len
        self.clip_value = clip_value
        self.blocks = nn.Sequential()
        d_in , d_out , d_med = feat_len , dim_out , dim_out//4
        for _ in range(resnet_blocks):
            self.blocks.append(_resnet_block_1d(d_in , d_out , d_med , clip_value))
            d_in = d_out
        self.fc_out = nn.Linear(d_out * seq_len , dim_out)

    def forward(self , x : Tensor) -> Tensor:
        x1 = x.reshape(-1,*x.shape[2:]).permute(0,2,1)
        output = self.blocks(x1)
        output = self.fc_out(output.flatten(start_dim=1))
        output = output.reshape(*x.shape[:2] , *output.shape[1:])
        return output
    
# 2-d conv resnet
class _resnet_block_2d(nn.Module):
    """Bottleneck 2-D ResNet block with optional clipping.

    Parallel to ``_resnet_block_1d`` but uses ``Conv2d`` layers.  Used for
    treating the ``(seq_len, feat_len)`` intra-day feature matrix as a 2-D
    input.

    Shapes:
        Input:  ``[bs, dim_in, seq_len, feat_len]``  (channels-first)
        Output: ``[bs, dim_out, seq_len, feat_len]``
    """
    def __init__(self, dim_in , dim_out = 64 , dim_med = 64 // 4 , clip_value = 10 , act_type = 'leaky' , **kwargs) -> None:
        super().__init__()
        self.clip_value = clip_value
        self.dim_in     = dim_in
        if dim_in == dim_out:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Conv2d(dim_in , dim_out , 1)

        self.conv = nn.Sequential(
            nn.Conv2d(dim_in , dim_med , 1) , 
            nn.BatchNorm2d(dim_med),
            Layer.Act.get_activation_fn(act_type) ,
            nn.Conv2d(dim_med, dim_med , kernel_size=3, stride=1, padding=1 , bias = False) , 
            nn.BatchNorm2d(dim_med),
            Layer.Act.get_activation_fn(act_type) ,
            nn.Conv2d(dim_med, dim_out , 1) , 
            nn.BatchNorm2d(dim_out),
            Layer.Act.get_activation_fn(act_type) ,
        )

    def forward(self , x : Tensor) -> Tensor:
        if x.shape[1] != self.dim_in and self.dim_in == 1: 
            Logger.alert1('auto add dim 1 in index 1!')
            x = x.unsqueeze(1)
        x1 = torch.clip(self.downsample(x) , -self.clip_value , self.clip_value)
        x2 = self.conv(x)
        return x1 + x2
    
class mod_resnet_2d(nn.Module):
    """Stacked 2-D ResNet blocks treating the intra-day feature grid as an image.

    Wraps the ``(seq_len, feat_len)`` feature matrix in a channel dimension
    (``dim_in=1``), applies ``resnet_blocks`` 2-D bottleneck residual blocks,
    then flattens and projects to ``dim_out``.

    Args:
        seq_len:       Intra-day sequence length.
        feat_len:      Number of features per bar.
        dim_out:       Output dimension (default ``64``).
        clip_value:    Residual clipping threshold (default ``10``).
        resnet_blocks: Number of residual blocks (default ``3``).

    Shapes:
        Input:  ``[bs, n_days, seq_len, feat_len]``
        Output: ``[bs, n_days, dim_out]``
    """
    def __init__(self, seq_len , feat_len , dim_out = 64 , clip_value = 10 , resnet_blocks = 3 , **kwargs) -> None:
        super().__init__()
        self.clip_value = clip_value
        self.blocks = nn.Sequential()
        d_in , d_out , d_med = 1 , dim_out , dim_out//4
        for _ in range(resnet_blocks):
            self.blocks.append(_resnet_block_2d(d_in , d_out , d_med , clip_value))
            d_in = d_out
        self.fc_out = nn.Linear(feat_len * seq_len * d_out , dim_out)

    def forward(self , x : Tensor) -> Tensor:
        x1 = x.reshape(-1,*x.shape[2:]).unsqueeze(-3)
        output = self.blocks(x1)
        output = self.fc_out(output.flatten(start_dim=1))
        output = output.reshape(*x.shape[:2] , *output.shape[1:])
        return output

if __name__ == '__main__':

    import torch
    import torch.nn as nn
    # pip install pytimedinput -i https://pypi.tuna.tsinghua.edu.cn/simple
    from src.res.algo.nn.model.RNN import mod_gru
    from src.res.algo.nn.model.CNN import mod_resnet_1d, mod_resnet_2d
        
    class resnet1d_gru(nn.Module):
        def __init__(self, seq_len , feat_len , dim_res = 16 , dim_rnn = 64 , **kwargs) -> None:
            super().__init__()
            self.resnet = mod_resnet_1d(seq_len , feat_len , dim_res , 10 , 3) 
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
            self.resnet = mod_resnet_2d(seq_len , feat_len , dim_res , 10 , 3) 
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
