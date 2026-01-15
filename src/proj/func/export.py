import os
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from pathlib import Path
from typing import Literal

from src.proj.log import Logger

__all__ = ['dfs_to_excel' , 'figs_to_pdf']

def dfs_to_excel(dfs : dict[str , pd.DataFrame] , path : str | Path , mode : Literal['a','w'] = 'w' , 
                 name_prefix = '' , print_prefix = None , indent : int = 1 , vb_level : int = 3):
    os.makedirs(os.path.dirname(path) , exist_ok=True)
    if mode == 'a': 
        mode = 'a' if os.path.exists(path) else 'w'
    with pd.ExcelWriter(path , 'openpyxl' , mode = mode) as writer:
        for key, value in dfs.items():
            value.to_excel(writer, sheet_name = f'{name_prefix}{key}')
    if print_prefix: 
        Logger.footnote(f'{print_prefix} saved to {path}' , indent = indent , vb_level = vb_level)
    return path

def figs_to_pdf(figs : dict[str , Figure] , path : str | Path , print_prefix = None , indent : int = 1 , vb_level : int = 3):
    os.makedirs(os.path.dirname(path) , exist_ok=True)
    with PdfPages(path) as pdf:
        for key, fig in figs.items():
            pdf.savefig(fig)
            plt.close(fig)
    if print_prefix: 
        Logger.footnote(f'{print_prefix} saved to {path}' , indent = indent , vb_level = vb_level)
    return path