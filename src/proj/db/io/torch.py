"""``torch.load`` wrapper that picks ``weights_only`` API for torch >= 2.6."""
from __future__ import annotations
import os
import re
import warnings
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.proj.log import Logger
from src.proj.core import strPath

__all__ = ['torch_load' , 'torch_save']

_STALE_TORCH_WARN = re.compile(r'numpy\.core' , re.I)

def _is_stale_torch_warning(msg : warnings.WarningMessage) -> bool:
    return msg.category is DeprecationWarning and bool(_STALE_TORCH_WARN.search(str(msg.message)))

def _atomic_torch_save(obj : Any , path : strPath) -> None:
    """Write a torch file atomically via a same-directory temp file."""
    import torch
    tmp_path = Path(path).with_name(f'{Path(path).name}.tmp.{os.getpid()}.{uuid4().hex[:8]}')
    try:
        torch.save(obj , tmp_path)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

def torch_save(obj : Any , path : strPath , prefix : str | None = None , indent : int = 1 , vb_level : Any = 3) -> bool:
    """
    save torch object to path with footnote
    """
    _atomic_torch_save(obj, path)
    if prefix:
        Logger.footnote(f'{prefix} saved to {path}' , indent = indent , vb_level = vb_level)
    return True

def torch_load(path : strPath , weights_only : bool = False , migrate_stale : bool = True , **kwargs):
    """``torch.load`` wrapper that picks ``weights_only`` API for torch >= 2.6.

    When legacy pickles emit ``numpy.core`` deprecation warnings, the loaded object
    is re-saved in place so subsequent loads match the current torch/numpy stack.
    """
    import torch
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter('always' , DeprecationWarning)
        try:
            if torch.__version__ < '2.6.0':
                obj = torch.load(path , **kwargs)
            else:
                obj = torch.load(path , weights_only = weights_only , **kwargs)
        except ModuleNotFoundError as e:
            Logger.error(f'torch_load({path}) error: ModuleNotFoundError: {e}')
            raise

    if migrate_stale and any(_is_stale_torch_warning(w) for w in caught):
        target_path = Path(path)
        Logger.footnote(f'Migrating stale torch file: {target_path}' , vb_level = 2)
        _atomic_torch_save(obj , target_path)

    return obj
