
import numpy as np

from pathlib import Path
from typing import Any , Literal , Optional

from . import path as PATH

class ModelPath:
    def __init__(self , model_name : Path | str | None | Any) -> None:
        if isinstance(model_name , ModelPath):
            name = model_name.name
        elif isinstance(model_name , Path):
            assert model_name.parent == PATH.model , model_name
            name = model_name.name
        elif model_name is None:
            name = ''
        else:
            name = model_name
        self.name = name

    @staticmethod
    def sub_dirs(path : Path , as_int = False):
        arr = [sub.name for sub in path.iterdir() if sub.is_dir()]
        if as_int: arr = np.array([v for v in arr if v.isdigit()]).astype(int)
        return np.sort(arr)
    def __bool__(self):         return bool(self.name)
    def __repr__(self) -> str:  return f'{self.__class__.__name__}(name={self.name})'
    def __call__(self , *args): return self.base.joinpath(*[str(arg) for arg in args])
    def archive(self , *args):  return self('archive' , *args)
    def conf(self , *args):     return self('configs' , *args)
    def rslt(self , *args):     return self('detailed_analysis' , *args)
    def snapshot(self , *args): return self('snapshot' , *args)
    def full_path(self , model_num , model_date , model_type):
        return self('archive', model_num , model_date , model_type)
    def exists(self , model_num , model_date , model_type):
        path = self.full_path(model_num , model_date , model_type)
        return path.exists() and len([s for s in path.iterdir()]) > 0
    def mkdir(self , model_nums = None , exist_ok=False):
        self.archive().mkdir(parents=True,exist_ok=exist_ok)
        if model_nums is not None: [self.archive(mm).mkdir(exist_ok=exist_ok) for mm in model_nums]
        self.conf().mkdir(exist_ok=exist_ok)
        self.rslt().mkdir(exist_ok=exist_ok)
        self.snapshot().mkdir(exist_ok=exist_ok)
    @property
    def base(self): 
        assert self.name
        return PATH.model.joinpath(self.name)
    @property
    def model_nums(self):
        return self.sub_dirs(self.archive() , as_int = True)
    @property
    def model_dates(self):
        return self.sub_dirs(self.archive(self.model_nums[-1]) , as_int = True)
    @property
    def model_types(self):
        return self.sub_dirs(self.archive(self.model_nums[-1] , self.model_dates[-1]) , as_int = False)

class RegModel(ModelPath):
    def __init__(self, name: str , type : Literal['best' , 'swalast' , 'swabest'] = 'best' ,
                 num  : int | list[int] | range | Literal['all'] = 0 , 
                 alias : Optional[str] = None) -> None:
        super().__init__(name)
        self.type = type
        self.num = num
        self.alias = alias
        self.model_path = ModelPath(self.name)

    def __repr__(self) -> str:  return f'{self.__class__.__name__}(name={self.name},type={self.type},num={str(self.num)},alias={str(self.alias)})'
    
FACTOR_DESTINATION_LAPTOP = Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha')
FACTOR_DESTINATION_SERVER = PATH.result.joinpath('Alpha')
REG_MODELS = [
    RegModel('gru_day'    , 'swalast' , 0 , 'gru_day_V0') ,
    RegModel('gruRTN_day' , 'swalast' , 0 , 'gruRTN_day_V0') , 
    RegModel('gru_avg'    , 'swabest' , 'all' , 'gru_day_V1')
    #RegModel('gruRTN_day' , 'swalast' , 'all' , 'gruRTN_day_V1') , 
]
