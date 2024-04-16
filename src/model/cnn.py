import torch

from torch import nn , Tensor
from torch.nn.utils.parametrizations import weight_norm

from .. import layer as Layer

# 1-d conv resnet
class _tcn_block(nn.Module):
    def __init__(self, input_dim , output_dim , dilation, dropout=0.0 , kernel_size=3):
        super().__init__()
        padding = (kernel_size-1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(input_dim , output_dim, kernel_size, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(output_dim, output_dim, kernel_size, padding=padding, dilation=dilation))
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
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
        def __init__(self, padding):
            super().__init__()
            self.padding = padding
        def forward(self, x):
            return x[:, :, :-self.padding] # .contiguous()

class mod_tcn(nn.Module):
    def __init__(self, input_dim , output_dim , dropout=0.0 , num_layers = 2 , kernel_size = 3):
        super().__init__()
        if kernel_size is None: kernel_size = 3
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
            print('auto permute!')
            x = x.permute(0,2,1) 
        x1 = torch.clip(self.downsample(x) , -self.clip_value , self.clip_value)
        x2 = self.conv(x)
        return x1 + x2
    
class mod_resnet_1d(nn.Module):
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
            print('auto add dim 1 in index 1!')
            x = x.unsqueeze(1)
        x1 = torch.clip(self.downsample(x) , -self.clip_value , self.clip_value)
        x2 = self.conv(x)
        return x1 + x2
    
class mod_resnet_2d(nn.Module):
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
    pass
    """
    import torch
    import torch.nn as nn
    # pip install pytimedinput -i https://pypi.tuna.tsinghua.edu.cn/simple
    from scripts.nn.My import mod_gru
    from scripts.nn.ResNet import resnet_1d, resnet_2d
        
    class resnet1d_gru(nn.Module):
        def __init__(self, seq_len , feat_len , dim_res = 16 , dim_rnn = 64 , **kwargs) -> None:
            super().__init__()
            self.resnet = resnet_1d(seq_len , feat_len , dim_res , 10 , 3) 
            self.gru    = mod_gru(dim_res , dim_rnn , 0.1 , 2)
        def forward(self , x):
            hidden = self.resnet(x)
            print("hidden shape :" , hidden.shape)
            output = self.gru(hidden)[:,-1,:]
            print("output shape :" , output.shape)
            return output
        
    class resnet2d_gru(nn.Module):
        def __init__(self, seq_len , feat_len , dim_res = 16 , dim_rnn = 64 , **kwargs) -> None:
            super().__init__()
            self.resnet = resnet_2d(seq_len , feat_len , dim_res , 10 , 3) 
            self.gru    = mod_gru(dim_res , dim_rnn , 0.1 , 2)
        def forward(self , x):
            hidden = self.resnet(x)
            output = self.gru(hidden)[:,-1,:]
            print("hidden shape :" , hidden.shape)
            print("output shape :" , output.shape)
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
    """
