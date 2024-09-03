from typing import Any , Literal
from . import path as PATH

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

def _read_conf(conf_type : str , name : str , **kwargs) -> dict[str , Any]:
    p = PATH.conf.joinpath(conf_type , f'{name}.yaml')
    assert p.exists() , p
    return PATH.read_yaml(p , **kwargs)

def glob(name : str):
    if name == 'tushare_indus': kwargs = {'encoding' : 'gbk'}
    else: kwargs = {}
    return _read_conf('glob' , name , **kwargs)

def train(name : str):
    return _read_conf('glob' , name)

