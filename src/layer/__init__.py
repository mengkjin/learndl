import torch
import torch.nn as nn

from copy import deepcopy

from . import (
    Attention , Embed , PE , RevIN , Act
)

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class EwLinear(nn.Module):
    def __init__(self, dim = -1 , keepdim = True):
        super().__init__()
        self.dim , self.keepdim = dim , keepdim
    def forward(self, inputs):
        return inputs.mean(dim = self.dim , keepdim = self.keepdim)
    
class Parallel(nn.Module):
    def __init__(self, sub_mod , num_mod , feedforward = True , concat_output = False):
        super().__init__()
        self.mod_list = nn.ModuleList([deepcopy(sub_mod) for _ in range(num_mod)])
        self.feedforward = feedforward
        self.concat_output = concat_output
    def forward(self, inputs):
        output = tuple([mod(inputs[i] if self.feedforward else inputs) for i,mod in enumerate(self.mod_list)])
        if self.concat_output:
            if isinstance(output[0] , (list,tuple)):
                output = tuple([torch.cat([out[i] for out in output] , dim = -1) for i in range(len(output[0]))])  
            else:
                output = torch.cat(output , dim = -1)
        return output