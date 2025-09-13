# to deal with different versions of torch

import torch
from pathlib import Path

def torch_load_old(path : str | Path , weights_only : bool = False , **kwargs):
    try:
        return torch.load(path , **kwargs)
    except ModuleNotFoundError as e:
        print(f'torch_load_old({path}) error: ModuleNotFoundError: {e}')
        raise e
    
def torch_load_new(path : str | Path , weights_only : bool = False , **kwargs):
    try:
        return torch.load(path , weights_only = weights_only , **kwargs)
    except ModuleNotFoundError as e:
        print(f'torch_load_new({path}) error: ModuleNotFoundError: {e}')
        raise e

if torch.__version__ < '2.6.0':
    torch_load = torch_load_old
else:
    torch_load = torch_load_new