import os
import numpy as np

from dataclasses import dataclass
from typing import Literal , Optional
from . import path as PATH

@dataclass(slots=True)
class RegModel:
    name : str
    type : Literal['best' , 'swalast' , 'swabest'] = 'best'
    num  : int | list[int] | range | Literal['all'] = 0
    alias : Optional[str] = None

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
        path = os.path.join(PATH.main , 'model' , self.name)
        return np.sort(np.array([int(sub) for sub in os.listdir(path) if os.path.isdir(os.path.join(path , sub)) and sub.isdigit()]))
    
    @property
    def model_dates(self):
        path = os.path.join(PATH.main , 'model' , self.name , str(self.num0))
        return np.sort(np.array([int(sub) for sub in os.listdir(path) if os.path.isdir(os.path.join(path , sub)) and sub.isdigit()]))
    
    @property
    def model_types(self):
        model_date = self.model_dates[-1]
        path = os.path.join(PATH.main , 'model' , self.name , str(self.num0) , str(model_date))
        return np.sort(np.array([sub for sub in os.listdir(path) if os.path.isdir(os.path.join(path , sub))]))
    
    def full_path(self , model_num , model_date , model_type):
        return os.path.join(PATH.main , 'model' , self.name , str(model_num) , str(model_date) , model_type)
    
FACTOR_DESTINATION = '//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha'
REG_MODELS = [
    RegModel('gru_day'    , 'swalast' , 0 , 'gru_day_V0') ,
    RegModel('gruRTN_day' , 'swalast' , 0 , 'gruRTN_day_V0') , 
    RegModel('gru_avg'    , 'swabest' , 'all' , 'gru_day_V1')
    #RegModel('gruRTN_day' , 'swalast' , 'all' , 'gruRTN_day_V1') , 
]
