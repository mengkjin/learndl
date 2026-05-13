"""``torch.load`` wrapper that picks ``weights_only`` API for torch >= 2.6."""
from __future__ import annotations
import torch
from src.proj import Logger
from src.proj.core import strPath

__all__ = ['torch_load' , 'RequireGrad']

def torch_load_old(path : strPath , weights_only : bool = False , **kwargs):
    """``torch.load`` without ``weights_only`` (torch < 2.6)."""
    try:
        return torch.load(path , **kwargs)
    except ModuleNotFoundError as e:
        Logger.error(f'torch_load_old({path}) error: ModuleNotFoundError: {e}')
        raise
    
def torch_load_new(path : strPath , weights_only : bool = False , **kwargs):
    """``torch.load`` with ``weights_only`` (torch >= 2.6)."""
    try:
        return torch.load(path , weights_only = weights_only , **kwargs)
    except ModuleNotFoundError as e:
        Logger.error(f'torch_load_new({path}) error: ModuleNotFoundError: {e}')
        raise

class RequireGrad:
    def __init__(self , require_grad : bool = True):
        self.require_grad = require_grad

    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(self.require_grad)

    def __exit__(self , exc_type , exc_value , traceback):
        torch.set_grad_enabled(self.prev)

if torch.__version__ < '2.6.0':
    torch_load = torch_load_old
else:
    torch_load = torch_load_new