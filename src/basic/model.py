
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Literal , Optional
from . import path as PATH

@dataclass
class ModelPath:
    name : str

    @property
    def base(self): return PATH.model.joinpath(self.name)

    @property
    def archive(self): return self.base.joinpath('archive')

    @property
    def conf(self): return self.base.joinpath('configs')

    @property
    def rslt(self): return self.base.joinpath('detailed_analysis')

    @property
    def snapshot(self): return self.base.joinpath('snapshot')

    def full_path(self , model_num , model_date , model_type):
        return self.archive.joinpath(self.name , str(model_num) , str(model_date) , model_type)

@dataclass
class RegModel:
    name : str
    type : Literal['best' , 'swalast' , 'swabest'] = 'best'
    num  : int | list[int] | range | Literal['all'] = 0
    alias : Optional[str] = None

    def __post_init__(self):
        self.model_path = ModelPath(self.name)

    @property
    def num0(self):
        if self.num == 'all':
            return 0
        elif isinstance(self.num , int):
            return self.num
        else:
            return list(self.num)[0]

    @property
    def model_nums(self):
        path = PATH.model.joinpath(self.name)
        return np.sort(np.array([sub.name for sub in path.iterdir() if sub.is_dir() and sub.name.isdigit()]).astype(int))
    
    @property
    def model_dates(self):
        path = PATH.model.joinpath(self.name , str(self.num0))
        return np.sort(np.array([sub.name for sub in path.iterdir() if sub.is_dir() and sub.name.isdigit()]).astype(int))
    
    @property
    def model_types(self):
        model_date = self.model_dates[-1]
        path = PATH.model.joinpath(self.name , str(self.num0) , str(model_date))
        return np.sort(np.array([sub.name for sub in path.iterdir() if sub.is_dir()]))
    
    def full_path(self , model_num , model_date , model_type):
        return self.model_path.full_path(model_num , model_date , model_type)

    
FACTOR_DESTINATION_LAPTOP = Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha')
FACTOR_DESTINATION_SERVER = PATH.result.joinpath('Alpha')
REG_MODELS = [
    RegModel('gru_day'    , 'swalast' , 0 , 'gru_day_V0') ,
    RegModel('gruRTN_day' , 'swalast' , 0 , 'gruRTN_day_V0') , 
    RegModel('gru_avg'    , 'swabest' , 'all' , 'gru_day_V1')
    #RegModel('gruRTN_day' , 'swalast' , 'all' , 'gruRTN_day_V1') , 
]
