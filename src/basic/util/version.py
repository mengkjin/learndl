# to deal with different versions of torch

import torch
from pathlib import Path

def torch_load(path : str | Path , weights_only : bool = False , **kwargs):
    if torch.__version__ < '2.6.0':
        return torch.load(path , **kwargs)
    else:
        return torch.load(path , weights_only = weights_only , **kwargs)