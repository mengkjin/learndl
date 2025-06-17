# to deal with different versions of torch

import torch
from pathlib import Path

def torch_load_old(path : str | Path , weights_only : bool = False , **kwargs):
    return torch.load(path , **kwargs)
    
def torch_load_new(path : str | Path , weights_only : bool = False , **kwargs):
    return torch.load(path , weights_only = weights_only , **kwargs)

if torch.__version__ < '2.6.0':
    torch_load = torch_load_old
else:
    torch_load = torch_load_new