import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch import Tensor
from typing import Any , Literal , Optional

from .. import path as PATH
from .. import conf as CONF




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
    def __init__(self , model_name : str , model_num : int , submodel : str) -> None:
        self.model_name , self.model_num , self.submodel = model_name , model_num , submodel

    @classmethod
    def from_key(cls , hidden_key : str):
        model_name , model_num , submodel = cls.parse_hidden_key(hidden_key)
        return cls(model_name , model_num , submodel)

    @staticmethod
    def create_hidden_key(model_name : str , model_num : int , submodel : str) : 
        return f'{model_name}.{model_num}.{submodel}'
    
    @property
    def hidden_key(self):
        return self.create_hidden_key(self.model_name , self.model_num , self.submodel)
    
    @staticmethod
    def parse_hidden_key(hidden_key : str): 
        model_name , model_num , submodel = hidden_key.split('.')
        assert submodel in ['best' , 'swabest' , 'swalast'] , hidden_key
        return model_name , int(model_num) , submodel
    
    @staticmethod
    def target_hidden_path(model_name : str , model_num : int , model_date , submodel : str):
        return PATH.hidden.joinpath(model_name , str(model_num) , f'{model_date}.{submodel}.feather')
    
    def target_path(self , model_date: int):
        return self.target_hidden_path(self.model_name , self.model_num , model_date , self.submodel)

    def model_dates(self):
        suffix = f'.{self.submodel}.feather'
        parent = self.target_path(0).parent
        dates = [int(p.name.removesuffix(suffix)) for p in parent.iterdir() if p.name.endswith(suffix)]
        return np.sort(dates)

    def save_hidden_df(self , hidden_df : pd.DataFrame , model_date : int):
        hidden_path = self.target_path(model_date)
        hidden_path.parent.mkdir(parents=True , exist_ok=True)
        hidden_df.to_feather(hidden_path)

    def get_hidden_df(self , model_date : int , exact = False):
        if not exact: model_date = self.latest_hidden_model_date(model_date)
        if not self.target_path(model_date).exists(): 
            hidden_df = pd.DataFrame()
        else:
            hidden_df = pd.read_feather(self.target_path(model_date))
        return model_date , hidden_df
    
    def latest_hidden_model_date(self , model_date):
        possible_model_dates = self.model_dates()
        return possible_model_dates[possible_model_dates <= model_date].max()

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
                 alias : Optional[str] = None , start_dt = -1) -> None:
        super().__init__(name)
        self.submodel = submodel
        self.num = num
        self.alias = alias
        self.model_path = ModelPath(self.name)
        self.start_dt = start_dt

    def __repr__(self) -> str:  return f'{self.__class__.__name__}(name={self.name},type={self.submodel},num={str(self.num)},alias={str(self.alias)})'

class HiddenExtractingModel(ModelPath):
    '''for a model to predict recent/history data'''
    def __init__(self , name : str , 
                 submodels : Optional[list | np.ndarray | Literal['best' , 'swalast' , 'swabest']] = None ,
                 nums : Optional[list | np.ndarray | int] = None ,
                 alias : Optional[str] = None):
        super().__init__(name)
        self.submodels = submodels
        self.nums = nums
        self.alias = alias
        self.model_path = ModelPath(self.name)
        
update_models = CONF.load('schedule' , 'update_models')
REG_MODELS = [RegisteredModel(**reg_model)       for reg_model in update_models['REG_MODELS']]
HID_MODELS = [HiddenExtractingModel(**hid_model) for hid_model in update_models['HID_MODELS']]