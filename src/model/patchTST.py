
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import layer as Layer

__all__ = ['PatchTST']

class PatchTST(nn.Module):
    """
    in:  [bs x seq_len x nvars]
    out: [bs x seq_len x nvars] for pretrain
         [bs x predict_steps] for prediction
    """
    def __init__(
        self, 
        nvars : int , 
        seq_len: int , 
        d_model: int , 
        patch_len:int = 5 , 
        stride:int|None = None, 
        shared_embedding=True, shared_head = False,
        revin:bool=True,n_layers:int=3, n_heads=8, d_ff:int=64, 
        norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act_type:str='gelu', 
        res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
        pe:str='zeros', learn_pe:bool=True, head_dropout = 0, predict_steps:int = 1,
        head_type = 'prediction', verbose:bool=False, **kwargs
    ):

        super().__init__()

        assert head_type in ['pretrain', 'prediction'], 'head type should be either pretrain, prediction, or regression'
        self.nvars = nvars
        self.head_type = head_type
        self.mask_fwd = head_type == 'pretrain'
        if stride is None: stride = patch_len // 2
        num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1

        # RevIN
        self.revin = Layer.RevIN.RevIN(num_features = nvars) if revin else None

        # Embedding
        self.embed = PatchTSTEmbed(nvars,d_model,patch_len,stride,mask_ratio=0.4,shared=shared_embedding)

        # Backbone
        self.backbone = PatchTSTEncoder(
            nvars, num_patch=num_patch, patch_len=d_model, 
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
            shared_embedding=shared_embedding, d_ff=d_ff,norm=norm,
            attn_dropout=attn_dropout, dropout=dropout, act_type=act_type, 
            res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
            pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)
        
        # Head
        if head_type == 'pretrain':
            self.head = PatchTSTPretrainHead(d_model, num_patch ,seq_len, head_dropout)
        elif head_type == 'prediction':
            self.head = PatchTSTPredictionHead(nvars, d_model, num_patch, predict_steps, 
                                               head_dropout , shared = shared_head , act_type = act_type)

    def forward(self, x):                             
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

class ModelPretrain(PatchTST):
    def __init__(self, head_type = 'pretrain', **kwargs) -> None:
        assert head_type == 'pretrain' , head_type
        super().__init__(head_type = 'pretrain', **kwargs)
    
    def pretrain_label(self , x):
        return x

class ModelPredict(PatchTST):
    def __init__(self, head_type = 'prediction', **kwargs) -> None:
        assert head_type == 'prediction' , head_type
        super().__init__(head_type = 'prediction', **kwargs)

class PatchTSTEmbed(nn.Module):
    '''
    in : [bs x seq_len x nvars]
    out: [bs x nvars x num_patch x d_model] 
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

    def forward(self, x , mask = False):
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
    
class PatchTSTPredictionHead(nn.Module):
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
    
    def forward(self, x):                     
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

class PatchTSTPretrainHead(nn.Module):
    '''
    in : [bs x nvars x d_model x num_patch]
    out: [bs x seq_len x nvars]
    '''
    def __init__(self, d_model , num_patch , seq_len , dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model * num_patch , seq_len)

    def forward(self, x):
        '''
        in : [bs x nvars x d_model x num_patch]
        out: [bs x seq_len x nvars]
        '''
        x = self.dropout(x)                     # [bs x nvars x d_model x num_patch]
        x = x.flatten(start_dim=2)              # [bs x nvars x d_model (x) num_patch]
        x = self.linear(x)                      # [bs x nvars x seq_len]
        x = x.permute(0,2,1)                    # [bs x seq_len x nvars]                     
        
        return x

class PatchTSTEncoder(nn.Module):
    '''
    in : [bs x nvars x num_patch x d_model]   
    out: [bs x nvars x d_model x num_patch]
    '''
    def __init__(self, nvars, num_patch, d_model, n_layers=3, n_heads=8, 
                 d_ff=64, norm='BatchNorm', attn_dropout=0., dropout=0., act_type='gelu', store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.nvars = nvars

        # Positional encoding
        self.W_pos = Layer.PE.positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
            pre_norm=pre_norm, act_type=act_type, res_attention=res_attention, n_layers=n_layers, 
            store_attn=store_attn)

    def forward(self, x):          
        '''
        in : [bs x nvars x num_patch x d_model]   
        out: [bs x nvars x d_model x num_patch]
        '''
        x = x.reshape(-1,*x.shape[-2:])             # [bs * nvars x num_patch x d_model]
        x = self.dropout(x + self.W_pos)            # [bs * nvars x num_patch x d_model]
        # Encoder

        x = self.encoder(x)                         # [bs * nvars x num_patch x d_model]
        x = x.reshape(-1,self.nvars,*x.shape[-2:])  # [bs x nvars x num_patch x d_model]
        x = x.permute(0,1,3,2)                      # [bs x nvars x d_model x num_patch]
        return x
    
class TSTEncoder(nn.Module):
    """
    in : [bs x num_patch x d_model]
    out: [bs x num_patch x d_model]
    """
    def __init__(self, d_model, n_heads, d_ff = 64, 
                 norm='BatchNorm', attn_dropout=0., dropout=0., act_type='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                     attn_dropout=attn_dropout, dropout=dropout,
                                                     act_type=act_type, res_attention=res_attention,
                                                     pre_norm=pre_norm, store_attn=store_attn) 
                                                     for _ in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src):
        """
        in : [bs x num_patch x d_model]
        out: [bs x num_patch x d_model]
        """
        output = src
        scores = None

        if self.res_attention:
            for i , mod in enumerate(self.layers): 
                output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output

class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff = 64, store_attn=False,
                 norm='BatchNorm', attn_dropout=0., dropout=0., bias=True, 
                 act_type='gelu', res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f'd_model ({d_model}) must be divisible by n_heads ({n_heads})'
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = Layer.Attention.MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if 'batch' in norm.lower():
            self.norm_attn = nn.Sequential(Layer.Transpose(1,2), nn.BatchNorm1d(d_model), Layer.Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                Layer.Act.get_activation_fn(act_type),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if 'batch' in norm.lower():
            self.norm_ffn = nn.Sequential(Layer.Transpose(1,2), nn.BatchNorm1d(d_model), Layer.Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src , prev = None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src

if __name__ == '__main__' :

    import torch
    from src.model.patchTST import ModelPretrain,  ModelPredict
    batch_size = 2 
    seq_len = 30
    patch_len = 3
    stride = 2
    n_inputs = 6
    mask_ratio = 0.4
    d_model = 16
    predict_steps = 5

    x = torch.rand(batch_size , seq_len , n_inputs)
    y = torch.rand(batch_size , predict_steps)

    model_pretrain = ModelPretrain(nvars = n_inputs, seq_len = seq_len , d_model = d_model, 
                                   patch_len = patch_len, stride = stride, 
                                   res_attention=True, pre_norm=False, store_attn=False, pe='zeros', learn_pe=True, 
                                   head_dropout = 0, head_type = 'pretrain', individual = False, verbose=True)
    print(model_pretrain(x).shape , model_pretrain.pretrain_label(x).shape)
    
    model_predict = ModelPredict(nvars = n_inputs, seq_len = seq_len , d_model = d_model, 
                                 patch_len = patch_len,  stride = stride, predict_steps = predict_steps ,
                                 res_attention=True, pre_norm=False, store_attn=False, pe='zeros', learn_pe=True, 
                                 head_dropout = 0, head_type = 'prediction', individual = False, verbose=True)
    print(model_predict(x).shape , y.shape)
