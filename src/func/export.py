import os
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from typing import Literal

def dfs_to_excel(dfs : dict[str , pd.DataFrame] , path : str , mode : Literal['a','w'] = 'w' , 
                 name_prefix = ''):
    os.makedirs(os.path.dirname(path) , exist_ok=True)
    if mode == 'a': mode = 'a' if os.path.exists(path) else 'w'
    with pd.ExcelWriter(path , 'openpyxl' , mode = mode) as writer:
        for key, value in dfs.items():
            value.to_excel(writer, sheet_name = f'{name_prefix}{key}')
    return path

def figs_to_pdf(figs : dict[str , Figure] , path : str):
    os.makedirs(os.path.dirname(path) , exist_ok=True)
    with PdfPages(path) as pdf:
        for key, fig in figs.items():
            pdf.savefig(fig)
            plt.close(fig)