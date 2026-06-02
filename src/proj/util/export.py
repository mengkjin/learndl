"""Export DataFrames to Excel and matplotlib figures to a single PDF."""

import os
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from typing import Any , Literal , Mapping

from src.proj.log import Logger
from src.proj.core import strPath

__all__ = ['dfs_to_excel' , 'figs_to_pdf']

_NONCHAR_STRIP = str.maketrans({"\ufeff": "", "\ufffe": "", "\uffff": ""})

def _sanitize_figure_text(fig) -> None:
    """Strip non-characters from all text objects in figure.

    Matplotlib may warn when rendering non-characters like U+FFFE.
    """
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    assert isinstance(fig , Figure) , f'fig must be a matplotlib figure , but got {type(fig)}'
    for obj in fig.findobj(match=Text):
        try:
            s = obj.get_text()
        except Exception:
            continue
        if isinstance(s, str) and any(ch in s for ch in ("\ufeff", "\ufffe", "\uffff")):
            obj.set_text(s.translate(_NONCHAR_STRIP))

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
            if isinstance(value , pl.DataFrame):
                value = value.to_pandas()
            value.to_excel(writer, sheet_name = f'{sheet_prefix}{key}')
    if prefix: 
        Logger.footnote(f'{prefix} saved to {path}' , indent = indent , vb_level = vb_level)
    return path

def figs_to_pdf(figs , path : strPath , prefix : str | None = None , indent : int = 1 , vb_level : Any = 3):
    """Save figures to one PDF and close each figure.

    Returns:
        Output ``path``.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(os.path.dirname(path) , exist_ok=True)
    with PdfPages(path) as pdf:
        for key, fig in figs.items():
            _sanitize_figure_text(fig)
            pdf.savefig(fig)
            plt.close(fig)
    if prefix: 
        Logger.footnote(f'{prefix} saved to {path}' , indent = indent , vb_level = vb_level)
    return path