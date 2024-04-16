import torch
import torch.nn.functional as F

from torch import nn , Tensor

from .. import layer as Layer

__all__ = ['ModernTCN']
class ModernTCN(nn.Module):
    '''
    in:  [bs x seq_len x nvars]
    out: [bs x seq_len x nvars] for pretrain
         [bs x predict_steps] for prediction
    '''
    def __init__(
        self, 
        nvars : int , 
        seq_len: int , 
        d_model:int, 
        patch_len:int = 5, 
        stride:int|None = None, 
        kernel_size:int=3,expansion_factor:int=1,
        shared_embedding=True, shared_head = False, channel_mixer=True,
        revin:bool=True,n_layers:int=3, dropout:float=0., act:str='gelu', 
        pe:str='zeros', learn_pe:bool=True, head_dropout = 0, predict_steps:int = 1,
        head_type = 'prediction', **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction'], 'head type should be either pretrain, prediction, or regression'
        self.nvars = nvars
        self.head_type = head_type
        if stride is None: stride = patch_len // 2
        num_patch = max(seq_len + patch_len - stride, 0) // stride - 1

        # RevIN
        self.revin = Layer.RevIN.RevIN(num_features = nvars) if revin else None

        # Embedding
        self.embed = ModernTCNEmbed(nvars,d_model,patch_len,stride,shared=shared_embedding)

        # Backbone
        self.backbone = ModernTCNEncoder(nvars , num_patch, n_layers, d_model, kernel_size, expansion_factor , channel_mixer, 
                                         dropout=dropout, activation=act,pe=pe, learn_pe=learn_pe)
        
        # Head
        if head_type == 'pretrain':
            self.head = ModernTCNPretrainHead(d_model, num_patch, seq_len , head_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'prediction':
            self.head = ModernTCNPredictionHead(self.nvars, d_model, num_patch, predict_steps, head_dropout , shared = shared_head)

    def forward(self, x : Tensor) -> Tensor:  
        '''
        in:  [bs x seq_len x nvars]
        out: [bs x seq_len x nvars] for pretrain
             [bs x predict_steps] for prediction
        
        '''
        if self.revin is not None: 
            x = self.revin(x , 'norm')              # [bs x seq_len x nvars]
        x = self.embed(x)                           # [bs x nvars x num_patch x d_model]
        x = self.backbone(x)                        # [bs x nvars x num_patch x d_model]
        if self.revin is not None: 
            x = x.permute(0,2,3,1)                  # [bs x d_model x num_patch x nvars]
            x = self.revin(x , 'denorm')            # [bs x d_model x num_patch x nvars]
            x = x.permute(0,3,1,2)                  # [bs x nvars x d_model x num_patch]
            
        x = self.head(x)                            # [bs x seq_len x nvars] | [bs x predict_steps]
        return x
    
class ModernTCNEmbed(nn.Module):
    '''
    1D卷积同时完成patch和embedding
    in : [bs x nvars x seq_len]
    out: [bs x nvars x num_patch x d_model] 
    '''
    def __init__(self, nvars : int, d_model : int, patch_len : int = 8, stride : int = 4, shared = True):
        # nvars , d_model , patch_len, stride , mask_ratio = 0. , shared = True
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.pad    = patch_len - stride , 0
        self.shared = shared
        self.layers = nn.ModuleList([
            nn.Conv1d(in_channels=1,out_channels=d_model,kernel_size=patch_len,stride=stride) 
            for _ in range(1 if shared else nvars)
        ])
        
    def forward(self, x : Tensor) -> Tensor:
        '''
        in : [bs x seq_len x nvars]
        out: [bs x nvars x num_patch x d_model] 
        '''
        bs , seq_len , nvars = x.shape
        x = x.permute(0,2,1)                        # [bs x nvars x seq_len]
        x = x.unsqueeze(2)                          # [bs x nvars x 1 x seq_len]
        x = x.reshape(-1,1,seq_len)
        x = F.pad(x,pad=self.pad,mode='replicate')  # [bs * nvars x 1 x (seq_len + patch_len - stride)] 
        if self.shared:
            x = self.layers[0](x)                   # [bs * nvars x d_model x num_patch] 
            x = x.reshape(bs,nvars,*x.shape[1:])    # [bs x nvars x d_model x num_patch] 
        else:
            x = x.reshape(bs,nvars,1,-1)            # [bs x nvars x 1 x (seq_len + kernel_size - stride)] 
            x_i = [layer(x[:,i,:,:]) for i , layer in enumerate(self.layers)]
            x = torch.stack(x_i, dim=1)             # [bs x nvars x d_model x num_patch]
        x = x.permute(0,1,3,2)                      # [bs x nvars x num_patch x d_model] 
        return x 

class MtcnTSMixer(nn.Module):
    '''
    in : [bs x nvars x d_model x num_patch]
    out: [bs x nvars * d_model x num_patch]
    '''
    def __init__(self, nvars, d_model, kernel_size = 3, activation = 'gelu'):
        super().__init__()
        self.nvars   = nvars
        self.d_model = d_model
        self.act     = Layer.Act.get_activation_fn(activation)
        self.dw_conv = nn.Conv1d(
            in_channels     = nvars * d_model, 
            out_channels    = nvars * d_model, 
            kernel_size     = kernel_size,
            groups          = nvars * d_model,
            padding         = 'same'
        )
        self.bn = nn.BatchNorm1d(nvars * d_model)
        
    def forward(self , x : Tensor) -> Tensor:
        '''
        in : [bs x nvars x d_model x num_patch]
        out: [bs x nvars * d_model x num_patch]
        '''
        bs = x.shape[0]
        x = x.reshape(bs,self.nvars*self.d_model,-1)    # [bs x nvars * d_model x num_patch]
        x = self.dw_conv(x)                             # [bs x nvars * d_model x num_patch]
        x = self.act(x)                                 # [bs x nvars * d_model x num_patch]
        x = self.bn(x)                                  # [bs x nvars * d_model x num_patch]
        return x

class MtcnFeatureMixer(nn.Module):
    '''
    in : [bs x nvars * d_model x num_patch]
    out: [bs x nvars x d_model x num_patch]
    '''
    def __init__(self, nvars, d_model, kernel_size = 1 , activation = 'gelu'):
        super().__init__()
        self.nvars   = nvars
        self.d_model = d_model
        self.act     = Layer.Act.get_activation_fn(activation)
        self.pw_con_up = nn.Conv1d(
            in_channels     = nvars * d_model, 
            out_channels    = kernel_size * nvars * d_model, 
            kernel_size     = 1,
            groups          = nvars
        )
        self.pw_con_down = nn.Conv1d(
            in_channels     = kernel_size * nvars * d_model, 
            out_channels    = nvars * d_model, 
            kernel_size     = 1 ,
            groups          = nvars
        )
        
    def forward(self, x : Tensor) -> Tensor:
        '''
        in : [bs x nvars * d_model x num_patch]
        out: [bs x nvars x d_model x num_patch]
        '''
        bs = x.shape[0]
        x = self.pw_con_up(x)                           # [bs x kernel_size * nvars * d_model x num_patch]
        x = self.act(x)                                 # [bs x kernel_size * nvars * d_model x num_patch]
        x = self.pw_con_down(x)                         # [bs x nvars * d_model x num_patch]
        x = x.reshape(bs,self.nvars,self.d_model,-1)    # [bs x nvars x d_model x num_patch]
        return x
    
    
class MtcnChannelMixer(nn.Module):
    '''
    in : [bs x nvars x d_model x num_patch]
    out: [bs x nvars x d_model x num_patch]
    '''
    def __init__(self, nvars, d_model, kernel_size = 1 , activation = 'gelu'):
        super().__init__()
        self.nvars   = nvars
        self.d_model = d_model
        self.act     = Layer.Act.get_activation_fn(activation)
        self.pw_con_up = nn.Conv1d(
            in_channels     = nvars * d_model, 
            out_channels    = kernel_size * nvars * d_model, 
            kernel_size     = 1,
            groups          = d_model
        )
        self.pw_con_down = nn.Conv1d(
            in_channels     = kernel_size * nvars * d_model, 
            out_channels    = nvars * d_model, 
            kernel_size     = 1,
            groups          = d_model
        )
        
    def forward(self , x : Tensor) -> Tensor:
        '''
        in : [bs x nvars x d_model x num_patch]
        out: [bs x nvars x d_model x num_patch]
        ''' 
        bs = x.shape[0]
        x = x.permute(0,2,1,3)                                      # [bs x d_model x nvars x num_patch]
        x = x.reshape(bs,self.nvars*self.d_model,-1)                # [bs x d_model * nvars x num_patch]
        x = self.pw_con_up(x)                                       # [bs x kernel_size * d_model * nvars x num_patch]
        x = self.act(x)                                             # [bs x kernel_size * d_model * nvars x num_patch]
        x = self.pw_con_down(x)                                     # [bs x d_model * nvars x num_patch]
        x = x.reshape(bs,self.d_model,self.nvars,-1)                # [bs x d_model x nvars x num_patch]
        x = x.permute(0,2,1,3)                                      # [bs x nvars x d_model x num_patch]
        return x  

class ModernTCNBlock(nn.Module):
    '''
    in : [bs x nvars x num_patch x d_model]
    out: [bs x nvars x num_patch x d_model]
    '''
    def __init__(self, nvars, d_model, kernel_size, expansion_factor = 1,channel_mixer=True,activation='gelu'):
        super().__init__()
        self.ts_mixer = MtcnTSMixer(nvars, d_model, kernel_size, activation)
        self.feature_mixer = MtcnFeatureMixer(nvars, d_model, expansion_factor,activation)
        self.channel_mixer = MtcnChannelMixer(nvars, d_model, expansion_factor,activation) if channel_mixer else nn.Sequential()
    
    def forward(self , x : Tensor) -> Tensor:
        '''
        in : [bs x nvars x num_patch x d_model]
        out: [bs x nvars x num_patch x d_model]
        '''
        z = x.permute(0,1,3,2)      # [bs x nvars x d_model x num_patch]
        z = self.ts_mixer(z)        # [bs x nvars x d_model x num_patch]
        z = self.feature_mixer(z)   # [bs x nvars x d_model x num_patch]
        z = self.channel_mixer(z)   # [bs x nvars x d_model x num_patch]
        z = z.permute(0,1,3,2)      # [bs x nvars x num_patch x d_model]
        return z + x
    
class ModernTCNEncoder(nn.Module):
    '''
    in : [bs x nvars x num_patch x d_model]   
    out: [bs x nvars x d_model x num_patch]
    '''
    def __init__(self, nvars, num_patch, n_layers=3, d_model=128, kernel_size=3, expansion_factor = 1 , 
                 channel_mixer = True, dropout=0., activation='gelu', pe='zeros', learn_pe=True,  **kwargs):

        super().__init__()
        self.nvars = nvars
        self.W_pos = Layer.PE.positional_encoding(pe, learn_pe, num_patch, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(*[
            ModernTCNBlock(nvars=nvars,d_model=d_model,kernel_size=kernel_size,expansion_factor=expansion_factor,
                           channel_mixer=channel_mixer,activation=activation) 
                           for _ in range(n_layers)])
        
    def forward(self , x : Tensor) -> Tensor:
        '''
        in : [bs x nvars x num_patch x d_model]   
        out: [bs x nvars x d_model x num_patch]
        '''
        x = self.dropout(x + self.W_pos)            # [bs x nvars x num_patch x d_model]
        x = self.encoder(x)                         # [bs x nvars x num_patch x d_model]
        x = x.permute(0,1,3,2)                      # [bs x nvars x d_model x num_patch]
        return x
    
class ModernTCNPredictionHead(nn.Module):
    '''
    in : [bs x nvars x d_model x num_patch]
    out: [bs x predict_steps]
    '''
    def __init__(self, nvars, d_model, num_patch, predict_steps = 1 , head_dropout=0, flatten=False , shared = False):
        super().__init__()

        self.shared = shared
        self.nvars = nvars
        self.flatten = flatten
        head_dim = d_model * num_patch
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(head_dim, d_model),
                nn.Dropout(head_dropout)
            ) for _ in range(1 if shared else nvars)
        ])
        self.linear = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(d_model * nvars , predict_steps),
        )
    
    def forward(self, x : Tensor) -> Tensor:
        '''
        in : [bs x nvars x d_model x num_patch]
        out: [bs x predict_steps]
        '''
        if self.shared:
            x = self.layers[0](x)          # [bs x nvars x d_model]
        else:
            x_i = [layer(x[:,i,:,:]) for i,layer in enumerate(self.layers)] 
            x = torch.stack(x_i, dim=1)    # [bs x nvars x d_model]
        x = x.transpose(2,1)               # [bs x d_model x nvars]
        x = self.linear(x)                 # [bs x predict_steps]  
        return x

class ModernTCNPretrainHead(nn.Module):
    '''
    in : [bs x nvars x d_model x num_patch]
    out: [bs x seq_len x nvars]
    '''
    def __init__(self, d_model , num_patch , seq_len , dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model * num_patch , seq_len)

    def forward(self, x : Tensor) -> Tensor:
        '''
        in : [bs x nvars x d_model x num_patch]
        out: [bs x seq_len x nvars]
        '''
        x = self.dropout(x)                     # [bs x nvars x d_model x num_patch]
        x = x.flatten(start_dim=2)              # [bs x nvars x d_model (x) num_patch]
        x = self.linear(x)                      # [bs x nvars x seq_len]
        x = x.permute(0,2,1)                    # [bs x seq_len x nvars]                     
        
        return x
    
if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from src.model.modernTCN import ModernTCNEmbed , ModernTCN

    batch_size = 2 
    seq_len = 30
    patch_len = 3
    stride = 2
    nvars = 6
    mask_ratio = 0.4
    d_model = 16
    predict_steps = 1

    num_patch = max(seq_len + patch_len - stride, 0) // stride - 1

    x = torch.rand(batch_size , seq_len , nvars)
    y = torch.rand(batch_size , predict_steps)
    print(x.shape , y.shape)
    embed = ModernTCNEmbed(nvars , d_model , patch_len=patch_len , stride=stride, shared=True)
    print(embed(x).shape)
    mtcn = ModernTCN(nvars ,seq_len , d_model , patch_len, stride , predict_steps=predict_steps , head_type='pretrain')
    print(x.shape , mtcn(x).shape)