"""Export DataFrames to Excel and matplotlib figures to a single PDF."""
from __future__ import annotations
import os
import pandas as pd

from typing import Any , Literal , Mapping , TYPE_CHECKING

from src.proj.log import Logger
from src.proj.core import strPath

if TYPE_CHECKING:
    import polars as pl

__all__ = ['dfs_to_excel']

def dfs_to_excel(dfs : Mapping[str , pd.DataFrame | pl.DataFrame] , path : strPath , mode : Literal['a','w'] = 'w' , 
                 sheet_prefix = '' , prefix : str | None = None , indent : int = 1 , vb_level : Any = 3):
    """Write each DataFrame to a sheet; optionally log via ``Logger.footnote``.

    Returns:
        Output ``path``.
    """
    os.makedirs(os.path.dirname(path) , exist_ok=True)
    if mode == 'a': 
        mode = 'a' if os.path.exists(path) else 'w'
    with pd.ExcelWriter(path , 'openpyxl' , mode = mode) as writer:
        for key, value in dfs.items():
            if not isinstance(value , pd.DataFrame):
                value = value.to_pandas()
            value.to_excel(writer, sheet_name = f'{sheet_prefix}{key}')
    if prefix: 
        Logger.footnote(f'{prefix} saved to {path}' , indent = indent , vb_level = vb_level)
    return path
