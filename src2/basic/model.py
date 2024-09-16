import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch import Tensor
from typing import Any , Literal , Optional

from . import path as PATH

FACTOR_DESTINATION_LAPTOP = Path('//hfm-pubshare/HFM各部门共享/量化投资部/龙昌伦/Alpha')
FACTOR_DESTINATION_SERVER = PATH.result.joinpath('Alpha')
_registered_models = [
    {'name': 'gru_day' , 'submodel' : 'swalast' , 'num'  : 0 , 'alias' : 'gru_day_V0'} ,
    {'name': 'gruRTN_day' , 'submodel' : 'swalast' , 'num'  : 0 , 'alias' : 'gruRTN_day_V0'} ,
    {'name': 'gru_avg' , 'submodel' : 'swalast' , 'num'  : 'all' , 'alias' : 'gru_day_V1'} ,
]

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
    def full_path(self , model_num , model_date , submodel):
        return self('archive', model_num , model_date , submodel)
    def model_file(self , model_num , model_date , submodel):
        return ModelFile(self('archive', model_num , model_date , submodel))
    def exists(self , model_num , model_date , submodel):
        path = self.full_path(model_num , model_date , submodel)
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
    def model_submodels(self):
        return self.sub_dirs(self.archive(self.model_nums[-1] , self.model_dates[-1]) , as_int = False)
    
class HiddenPath:
    def __init__(self , hidden_key : str) -> None:
        self.key = hidden_key

    @staticmethod
    def hidden_key(model_num : int , model_date : int , submodel : str) : 
        return f'hidden.{model_num}.{submodel}.{model_date}.feather'

    def model_dates(self):
        name , num , submodel = self.key.split('.')
        prefix = f'hidden.{num}.{submodel}.'
        suffix = f'.feather'
        dates = []
        for p in PATH.hidden.joinpath(name).iterdir():
            if p.name.startswith(prefix) and p.name.endswith(suffix):
                dates.append(int(p.name.removeprefix(prefix).removesuffix(suffix)))
        return np.sort(dates)

    def get_hidden_df(self , model_date : int):
        possible_model_dates = self.model_dates()
        hmd = possible_model_dates[possible_model_dates <= model_date].max()
        name , num , submodel = self.key.split('.')
        hidden_path = PATH.hidden.joinpath(name , f'hidden.{num}.{submodel}.{hmd}.feather')
        hidden_df = pd.read_feather(hidden_path)
        return hidden_df

class ModelDict:
    __slots__ = ['state_dict' , 'booster_head' , 'booster_dict']
    def __init__(self ,
                 state_dict  : Optional[dict[str,Tensor]] = None , 
                 booster_head : Optional[dict[str,Any]] = None ,
                 booster_dict : Optional[dict[str,Any]] = None) -> None:
        self.state_dict = state_dict
        self.booster_head = booster_head
        self.booster_dict = booster_dict

    def reset(self):
        self.state_dict = None
        self.booster_head = None
        self.booster_dict = None

    def save(self , path : str | Path , stack = False):
        if isinstance(path , str): path = Path(path)
        path.mkdir(parents=True,exist_ok=True)
        for key in self.__slots__:
            if (value := getattr(self , key)) is not None:
                torch.save(value , path.joinpath(f'{key}.stack.pt' if stack else f'{key}.pt'))

    @property
    def legal(self):
        if self.state_dict is not None:
            assert self.booster_dict is None 
        else:
            assert self.booster_head is None
        return True

class ModelFile:
    def __init__(self , model_path : Path) -> None:
        self.model_path = model_path
    def __getitem__(self , key): return self.load(key)
    def load(self , key : str) -> Any:
        assert key in ModelDict.__slots__ , (key , ModelDict.__slots__)
        path = self.model_path.joinpath(f'{key}.pt')
        return torch.load(path , map_location='cpu') if path.exists() else None
    def exists(self) -> bool: 
        return any([self.model_path.joinpath(f'{key}.pt').exists() for key in ModelDict.__slots__])
    def model_dict(self):
        return ModelDict(**{key:self.load(key) for key in ModelDict.__slots__})
    
class RegisteredModel(ModelPath):
    def __init__(self, name: str , submodel : Literal['best' , 'swalast' , 'swabest'] = 'best' ,
                 num  : int | list[int] | range | Literal['all'] = 0 , 
                 alias : Optional[str] = None) -> None:
        super().__init__(name)
        self.submodel = submodel
        self.num = num
        self.alias = alias
        self.model_path = ModelPath(self.name)

    def __repr__(self) -> str:  return f'{self.__class__.__name__}(name={self.name},type={self.submodel},num={str(self.num)},alias={str(self.alias)})'


REG_MODELS = [RegisteredModel(**reg_model) for reg_model in _registered_models]
