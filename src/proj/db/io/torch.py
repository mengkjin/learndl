"""``torch.load`` wrapper that picks ``weights_only`` API for torch >= 2.6."""
from __future__ import annotations
from typing import Any
from src.proj.log import Logger
from src.proj.core import strPath

__all__ = ['torch_load' , 'torch_save']

def torch_save(obj : Any , path : strPath , prefix : str | None = None , indent : int = 1 , vb_level : Any = 3) -> bool:
    """
    save torch object to path with footnote
    """
    import torch
    torch.save(obj, path)
    if prefix:
        Logger.footnote(f'{prefix} saved to {path}' , indent = indent , vb_level = vb_level)
    return True

def torch_load(path : strPath , weights_only : bool = False , **kwargs):
    """``torch.load`` wrapper that picks ``weights_only`` API for torch >= 2.6."""
    import torch
    try:
        if torch.__version__ < '2.6.0':
            return torch.load(path , **kwargs)
        else:
            return torch.load(path , weights_only = weights_only , **kwargs)
    except ModuleNotFoundError as e:
        Logger.error(f'torch_load_new({path}) error: ModuleNotFoundError: {e}')
        raise