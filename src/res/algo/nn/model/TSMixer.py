import torch

from torch import nn , Tensor

from src.proj import Logger
from .. import layer as Layer

__all__ = ['TSMixer']

class TSMixer(nn.Module):
    """
    in:  [bs x seq_len x nvars]
    out: [bs x seq_len x nvars] for pretrain
         [bs x predict_steps] for prediction
    """
    def __init__(
        self, 
        nvars : int , 
        seq_len: int , 
        d_model:int , 
        patch_len:int = 5 , 
        stride:int|None = None, 
        channel_mixer = True , expansion_factor = 2 , gated_attn = True ,
        shared_embedding=True, shared_head = False,
        revin:bool=True,
        norm_type:str='batch', dropout:float=0., act_type:str='gelu', 
        pe:str='zeros', learn_pe:bool=True, head_dropout = 0, predict_steps:int = 1,
        head_type = 'prediction', **kwargs
    ):
        super().__init__()
        assert head_type in ['pretrain', 'prediction'], \
            f'head type should be either pretrain, prediction, or regression, but got {head_type}'

        self.nvars = nvars
        self.head_type = head_type
        self.mask_fwd = head_type == 'pretrain'
        if stride is None: 
            stride = patch_len // 2
        num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1

        # RevIN
        self.revin = Layer.RevIN.RevIN(num_features = nvars) if revin else None

        # Embedding
        self.embed = TSMixerEmbed(nvars,d_model,patch_len,stride,mask_ratio=0.4,shared=shared_embedding)

        # Backbone
        self.backbone = TSMixerEncoder(            
            nvars, num_patch, d_model=d_model, channel_mixer = channel_mixer, 
            dropout=dropout, expansion_factor = expansion_factor, gated_attn = gated_attn , norm_type = norm_type, act_type=act_type, 
            pe=pe, learn_pe=learn_pe, **kwargs)
        
        # Head
        if head_type == 'pretrain':
            self.head = TSMixerPretrainHead(d_model, num_patch ,seq_len, head_dropout)
        elif head_type == 'prediction':
            self.head = TSMixerPredictionHead(nvars, d_model, num_patch, predict_steps, 
                                              head_dropout , shared = shared_head , act_type = act_type)

    def forward(self, x : Tensor) -> Tensor:                             
        """
        in:  [bs x seq_len x nvars]
        out: [bs x seq_len x nvars] for pretrain
             [bs x predict_steps] for prediction
        """   

        if self.revin is not None: 
            x = self.revin(x , 'norm')              # [bs x seq_len x nvars]
        x = self.embed(x , self.mask_fwd)           # [bs x nvars x num_patch x d_model]
        x = self.backbone(x)                        # [bs x nvars x d_model x num_patch]

        if self.revin is not None: 
            x = x.permute(0,2,3,1)                  # [bs x d_model x num_patch x nvars]
            x = self.revin(x , 'denorm')            # [bs x d_model x num_patch x nvars]
            x = x.permute(0,3,1,2)                  # [bs x nvars x d_model x num_patch]

        x = self.head(x)                            # [bs x seq_len x nvars] | [bs x predict_steps]
        return x
    
class TSMixerPredictionHead(nn.Module):
    '''
    in : [bs x nvars x d_model x num_patch]
    out: [bs x predict_steps]
    '''
    def __init__(self, nvars, d_model, num_patch, predict_steps = 1 , head_dropout=0, 
                 flatten=False , shared = False , act_type = 'gelu'):
        super().__init__()

        self.shared = shared
        self.nvars = nvars
        self.flatten = flatten
        head_dim = d_model * num_patch

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(head_dim, d_model),
                Layer.Act.get_activation_fn(act_type),
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

class TSMixerPretrainHead(nn.Module):
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
    
# %%
class TSMixerEmbed(nn.Module):
    '''
    in : [bs x seq_len x nvars]
    out: [bs x nvars x num_patch x d_model] 
    same as patchTST
    '''
    def __init__(self, nvars , d_model , patch_len, stride , mask_ratio = 0. , shared = True):
        super().__init__()
        self.nvars = nvars
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio
        self.shared = shared 
        self.layers = nn.ModuleList([
            nn.Linear(patch_len, d_model) 
            for _ in range(1 if shared else nvars)
        ])

    def forward(self, x : Tensor , mask = False) -> Tensor:
        '''
        in : [bs x seq_len x nvars]
        out: [bs x nvars x num_patch x d_model] 
        '''
        if mask:
            x = self.patch_masking(x)[0]
        else:
            x = self.create_patch(x , self.patch_len, self.stride)[0] 
        # [bs x num_patch x nvars x patch_len]
        x = x.permute(0,2,1,3)  # x_p: [bs x nvars x num_patch x patch_len]
        if self.shared:
            x = self.layers[0](x)
        else:
            x_i = [layer(x[:,i,:,:]) for i,layer in enumerate(self.layers)]
            x = torch.stack(x_i, dim=1)
        return x # [bs x nvars x num_patch x d_model]
    
    @staticmethod
    def create_patch(x , patch_len, stride):
        num_patch = (max(x.shape[1], patch_len)-patch_len) // stride + 1
        s_begin = x.shape[1] - patch_len - stride*(num_patch-1)
        x_patch = x[:, s_begin:, :].unfold(dimension=1, size=patch_len, step=stride)                 
        return x_patch, x_patch.shape[1] # x_patch: [bs x num_patch x nvars x patch_len]
    
    def patch_masking(self , x):
        x_patch, _ = self.create_patch(x , self.patch_len, self.stride)    # xb_patch: [bs x num_patch x nvars x patch_len]
        x_patch_mask , _ , mask , _ = self.random_masking(x_patch, self.mask_ratio)   # xb_mask: [bs x num_patch x nvars x patch_len]
        mask = mask.bool()    # mask: [bs x num_patch x nvars]
        return x_patch_mask , mask

    @staticmethod
    def random_masking(x_patch , mask_ratio):
        # x_patch: [bs x num_patch x nvars x patch_len]
        bs, L, nvars, D = x_patch.shape
        x = x_patch.clone()
        
        len_keep = int(L * (1 - mask_ratio))
            
        noise = torch.rand(bs, L, nvars,device=x_patch.device)  # noise in [0, 1], bs x L x nvars
            
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)                                     # ids_restore: [bs x L x nvars]

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep, :]                                             # ids_keep: [bs x len_keep x nvars]         
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1, D))    # x_kept: [bs x len_keep x nvars  x patch_len]
    
        # removed x
        x_removed = torch.zeros(bs, L-len_keep, nvars, D, device=x_patch.device)            # x_removed: [bs x (L-len_keep) x nvars x patch_len]
        x_ = torch.cat([x_kept, x_removed], dim=1)                                          # x_: [bs x L x nvars x patch_len]

        # combine the kept part and the removed one
        x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,1,D)) # x_masked: [bs x num_patch x nvars x patch_len]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([bs, L, nvars], device=x.device)                                  # mask: [bs x num_patch x nvars]
        mask[:, :len_keep, :] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)                                 # [bs x num_patch x nvars]
        return x_masked, x_kept, mask, ids_restore


class MixerNormLayer(nn.Module):
    '''
    Batch / Layer Norm
    in : [bs x nvars x num_patch x d_model] 
    out: [bs x nvars x num_patch x d_model] 
    same as patchTST
    '''
    def __init__(self, norm_type : str , d_model : int):
        super().__init__()
        self.batch_norm = 'batch' in norm_type.lower()
        if self.batch_norm:
            self.norm = nn.BatchNorm1d(d_model) # 默认格式为(N,C) 或 (N,C,L)
        else:
            self.norm = nn.LayerNorm(d_model)  # 默认对最后一个维度进行LayerNorm
            
    def forward(self, x : Tensor) -> Tensor:
        '''
        in : [bs x nvars x num_patch x d_model] 
        out: [bs x nvars x num_patch x d_model] 
        '''
        if self.batch_norm:
            bs = x.shape[0]
            x = x.permute(0,1,3,2)                  # [bs x nvars x d_model x num_patch] 
            x = x.reshape(-1,*x.shape[-2:])         # [bs * nvars x d_model x num_patch] 
            x = self.norm(x)
            x = x.reshape(bs,-1,*x.shape[-2:])      # [bs x nvars x d_model x num_patch] 
            x = x.permute(0,1,3,2)                  # [bs x nvars x num_patch x d_model] 
        else:
            x = self.norm(x)
        return x

class GatedAttention(nn.Module):
    '''
    in : [... x in_features]
    out: [... x out_features]
    '''
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, x : Tensor) -> Tensor:
        attn_weight = self.attn_softmax(self.attn_layer(x))
        return x * attn_weight
 
class MixerMLP(nn.Module):
    '''
    similar to FFN
    in : [... x in_features]
    out: [... x out_features]
    '''
    def __init__(self, in_size: int, out_size: int, expansion_factor=1,dropout=0.,act_type='gelu'):
        super().__init__()
        num_hidden = in_size * expansion_factor
        self.layers = nn.Sequential(
            nn.Linear(in_size, num_hidden) ,
            Layer.Act.get_activation_fn(act_type) ,
            nn.Dropout(dropout) ,
            nn.Linear(num_hidden , out_size) ,
            nn.Dropout(dropout)
        )

    def forward(self, x : Tensor) -> Tensor:
        return self.layers(x)

class PatchMixerBlock(nn.Module):
    '''
    inter patch information extraction with channel independence
    in : [bs x nvars x num_patch x d_model] 
    out: [bs x nvars x num_patch x d_model] 
    '''
    def __init__(self,
                 num_patch,
                 d_model,
                 dropout = 0.,
                 expansion_factor = 2,
                 gated_attn=True,
                 norm_type='batch',
                 act_type='gelu' ,
                 ):
        super().__init__()

        self.norm = MixerNormLayer(norm_type , d_model)
        self.mlp = MixerMLP(num_patch,num_patch,expansion_factor,dropout,act_type)
        self.gating = GatedAttention(num_patch, num_patch) if gated_attn else nn.Sequential()

    def forward(self, x : Tensor) -> Tensor:
        '''
        in : [bs x nvars x num_patch x d_model] 
        out: [bs x nvars x num_patch x d_model] 
        '''
        residual = x
        x = self.norm(x)          # [bs x nvars x num_patch x d_model] 
        x = x.permute(0,1,3,2)    # [bs x nvars x d_model x num_patch] 
        x = self.mlp(x)           # [bs x nvars x d_model x num_patch] 
        x = self.gating(x)        # [bs x nvars x d_model x num_patch] 
        x = x.permute(0,1,3,2)    # [bs x nvars x num_patch x d_model] 
        return residual + x       # [bs x nvars x num_patch x d_model] 
    
class FeatureMixerBlock(nn.Module):
    '''
    inter Feature information extraction
    in : [bs x nvars x num_patch x d_model] 
    out: [bs x nvars x num_patch x d_model] 
    '''
    def __init__(self , 
                 d_model ,
                 dropout = 0. ,
                 expansion_factor = 2 ,
                 gated_attn=True ,
                 norm_type = 'batch' ,
                 act_type = 'gelu' ,
                 ):
        super().__init__()

        self.norm = MixerNormLayer(norm_type,d_model)
        self.mlp = MixerMLP(d_model,d_model,expansion_factor,dropout,act_type)
        self.gating = GatedAttention(d_model, d_model) if gated_attn else nn.Sequential()

    def forward(self, x : Tensor) -> Tensor:
        '''
        in : [bs x nvars x num_patch x d_model] 
        out: [bs x nvars x num_patch x d_model] 
        '''
        residual = x
        x = self.norm(x)          # [bs x nvars x num_patch x d_model] 
        x = self.mlp(x)           # [bs x nvars x d_model x num_patch] 
        x = self.gating(x)        # [bs x nvars x d_model x num_patch] 
        return residual + x       # [bs x nvars x num_patch x d_model] 

class ChannelMixerBlock(nn.Module):
    '''
    inter Channel information extraction
    in : [bs x nvars x num_patch x d_model] 
    out: [bs x nvars x num_patch x d_model] 
    '''

    def __init__(self,
                 d_model,
                 in_channel,
                 dropout = 0.,
                 expansion_factor = 2,
                 gated_attn=True,
                 norm_type = 'batch' ,
                 act_type = 'gelu' ,
                 ):

        super().__init__()

        self.norm = MixerNormLayer(norm_type,d_model)
        self.mlp = MixerMLP(in_channel,in_channel,expansion_factor,dropout,act_type)
        self.gating = GatedAttention(in_channel, in_channel) if gated_attn else nn.Sequential()

    def forward(self, x : Tensor) -> Tensor:
        '''
        in : [bs x nvars x num_patch x d_model] 
        out: [bs x nvars x num_patch x d_model] 
        '''
        residual = x
        x = self.norm(x)          # [bs x nvars x num_patch x d_model]
        x = x.permute(0,3,2,1)    # [bs x d_model x num_patch x nvars]
        x = self.gating(x)        # [bs x d_model x num_patch x nvars]
        x = self.mlp(x)           # [bs x d_model x num_patch x nvars]
        x = x.permute(0,3,2,1)    # [bs x nvars x num_patch x d_model]
        return x + residual       # [bs x nvars x num_patch x d_model]
    
class TSMixerEncoder(nn.Module):
    '''
    in : [bs x nvars x num_patch x d_model]
    out: [bs x nvars x d_model x num_patch]
    '''
    def __init__(self, nvars, num_patch, d_model , channel_mixer = True, 
                 dropout=0., expansion_factor = 2, gated_attn = True , norm_type = 'batch', act_type='gelu', 
                 pe='zeros', learn_pe=True, **kwargs):

        super().__init__()
        self.nvars = nvars

        # Positional encoding
        self.W_pos = Layer.PE.positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        self.mixer_p = PatchMixerBlock(num_patch , d_model , dropout , expansion_factor , gated_attn , norm_type , act_type)
        self.mixer_f = FeatureMixerBlock(d_model , dropout , expansion_factor , gated_attn , norm_type , act_type)
        self.mixer_c = ChannelMixerBlock(d_model, nvars , dropout , expansion_factor , gated_attn , norm_type , act_type) if channel_mixer else nn.Sequential()
        
    def forward(self, x : Tensor) -> Tensor:          
        '''
        in : [bs x nvars x num_patch x d_model]   
        out: [bs x nvars x d_model x num_patch]
        '''
        x = self.dropout(x + self.W_pos)            # [bs x nvars x num_patch x d_model]
        x = self.mixer_p(x)                         # [bs x nvars x num_patch x d_model]
        x = self.mixer_f(x)                         # [bs x nvars x num_patch x d_model]
        x = self.mixer_c(x)                         # [bs x nvars x num_patch x d_model]
        x = x.permute(0,1,3,2)                      # [bs x nvars x d_model x num_patch]
        return x

class ts_mixer(TSMixer):
    def __init__(self , input_dim , seq_len , hidden_dim , num_output = 1 , **kwargs):
        super().__init__(nvars = input_dim , seq_len = seq_len , d_model = hidden_dim , 
                         predict_steps = num_output , head_type = 'prediction' , **kwargs)
        
if __name__ == '__main__':

    import torch
    import torch.nn as nn

    batch_size = 2 
    seq_len = 30
    patch_len = 3
    stride = 2
    nvars = 6
    mask_ratio = 0.4
    d_model = 16
    predict_steps = 1
    shared_embedding = True

    num_patch = max(seq_len - patch_len, 0) // stride + 1 # 15

    x = torch.rand(batch_size , seq_len , nvars)
    y = torch.rand(batch_size , predict_steps)

    Logger.stdout(x.shape , y.shape)
    embed = TSMixerEmbed(nvars,d_model,patch_len,stride,mask_ratio=0.4,shared=shared_embedding)
    Logger.stdout(embed(x).shape , (batch_size , nvars , num_patch , d_model))
    net = TSMixer(nvars , seq_len , d_model , patch_len , stride , shared_embedding=shared_embedding , head_type='pretrain' , predict_steps = predict_steps)
    Logger.stdout(net(x).shape)


