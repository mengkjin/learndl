import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
    def forward(self, x):
        '''
        in : [bs x seq_len x nvars]
        out: [bs x nvars x num_patch x d_model] 
        '''
        bs , seq_len , nvars = x.shape
        # num_patch = (max(seq_len, self.kernel_size)-self.kernel_size) // self.stride + 2
        x = x.permute(0,2,1)                        # [bs x nvars x seq_len]
        x = x.unsqueeze(2)                          # [bs x nvars x 1 x seq_len]
        x = x.reshape(-1,1,seq_len)
        x = F.pad(x,pad=self.pad,mode='replicate')  # [bs * nvars x 1 x (seq_len + kernel_size - stride)] 

        if self.shared:
            x = self.layers[0](x)                     # [bs * nvars x d_model x num_patch] 
            x = x.reshape(bs,nvars,*x.shape[1:])    # [bs x nvars x d_model x num_patch] 
        else:
            x = x.reshape(bs,nvars,1,-1)            # [bs x nvars x 1 x (seq_len + kernel_size - stride)] 
            x_i = [layer(x[:,i,:,:]) for i , layer in enumerate(self.layers)]
            x = torch.stack(x_i, dim=1)             # [bs x nvars x d_model x num_patch]
        x = x.permute(0,1,3,2)                      # [bs x nvars x num_patch x d_model] 
        return x 

