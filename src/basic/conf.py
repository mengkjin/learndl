from typing import Literal

SILENT        : bool = False
SAVE_OPT_DB   : Literal['feather' , 'parquet'] = 'feather'
SAVE_OPT_BLK  : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
SAVE_OPT_NORM : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
SAVE_OPT_MODEL: Literal['pt'] = 'pt'

class Silence:
    def __enter__(self) -> None: 
        global SILENT
        SILENT = True
    def __exit__(self , *args) -> None: 
        global SILENT
        SILENT = False