"""PatchLayer: Patch-based layer.
Based on PatchTSTEmbed.
"""

import torch
from torch import nn , Tensor
import torch.nn.functional as F

class LinearPatch(nn.Module):
    '''
    Linear patch layer.
    in : [bs x seq_len x nvars]
    out: [bs x nvars x num_patch x d_model] 
    '''
    def __init__(self, nvars : int, d_model : int, patch_len : int, stride : int, mask_ratio : float = 0. , shared : bool = True):
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

    def forward(self, x : Tensor, mask = False) -> Tensor:
        '''
        in : [bs x seq_len x nvars]
        out: [bs x nvars x num_patch x d_model] 
        '''
        if mask:
            x = self.patch_masking(x , self.patch_len, self.stride, self.mask_ratio)[0]
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
    
    @classmethod
    def create_patch(cls , x : Tensor, patch_len : int, stride : int) -> tuple[Tensor, int]:
        num_patch = (max(x.shape[1], patch_len)-patch_len) // stride + 1
        s_begin = x.shape[1] - patch_len - stride*(num_patch-1)
        x_patch = x[:, s_begin:, :].unfold(dimension=1, size=patch_len, step=stride)                 
        return x_patch, x_patch.shape[1] # x_patch: [bs x num_patch x nvars x patch_len]
    
    @classmethod
    def patch_masking(cls , x : Tensor , patch_len : int, stride : int, mask_ratio : float) -> tuple[Tensor, Tensor]:
        x_patch, _ = cls.create_patch(x , patch_len, stride)    # xb_patch: [bs x num_patch x nvars x patch_len]
        x_patch_mask , _ , mask , _ = cls.random_masking(x_patch, mask_ratio)   # xb_mask: [bs x num_patch x nvars x patch_len]
        mask = mask.bool()    # mask: [bs x num_patch x nvars]
        return x_patch_mask , mask

    @staticmethod
    def random_masking(x_patch : Tensor, mask_ratio : float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

class ConvolutionalPatch(nn.Module):
    '''
    Convolutional patch layer.
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