from dataclasses import dataclass
from typing import Literal , Optional

@dataclass
class CustomConf:
    SILENT        : bool = False
    SAVE_OPT_DB   : Literal['feather' , 'parquet'] = 'feather'
    SAVE_OPT_BLK  : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
    SAVE_OPT_NORM : Literal['pt' , 'pth' , 'npz' , 'npy' , 'np'] = 'pt'
    SAVE_OPT_MODEL: Literal['pt'] = 'pt'

CONF = CustomConf()