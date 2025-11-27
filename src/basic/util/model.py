import torch
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any , Literal

from src.proj import MACHINE , PATH
from src.basic import DB
from .version import torch_load
from .calendar import CALENDAR

__all__ = ['ModelPath' , 'HiddenPath' , 'ModelDict' , 'ModelFile' , 'ModelDBMapping' , 'RegisteredModel' , 'HiddenExtractingModel']

class ModelPath:
    """
    model path
    access stored model in learndl/models
    example:
        model_path = ModelPath('gru_day_V0')
    """
    def __init__(self , model_name : Path | str | None | Any) -> None:
        if isinstance(model_name , ModelPath):
            name = model_name.name
        elif isinstance(model_name , Path):
            assert model_name.absolute().parent in [PATH.model , PATH.null_model] , model_name
            name = model_name.name
        elif model_name is None:
            name = ''
        else:
            name = model_name
        self.name = name

    @staticmethod
    def sub_dirs(path : Path , as_int = False) -> np.ndarray:
        """sub directories"""
        arr = [sub.name for sub in path.iterdir() if sub.is_dir()]
        if as_int: 
            arr = np.array([v for v in arr if v.isdigit()]).astype(int)
        return np.sort(arr)
    def __bool__(self):         
        return bool(self.name)
    def __repr__(self) -> str:  
        return f'{self.__class__.__name__}(model_name={self.name})'
    def __call__(self , *args): 
        return self.base.joinpath(*[str(arg) for arg in args])
    def archive(self , *args) -> Path:
        """model archive path"""
        return self('archive' , *args)
    def conf(self , *args) -> Path:  
        """model configs path"""
        if self:
            return self('configs' , *args)
        else:
            return PATH.conf.joinpath(*args)
    def rslt(self , *args) -> Path:     
        """model results path"""
        return self('detailed_analysis' , *args)
    def snapshot(self , *args) -> Path:
        """model snapshot path"""
        return self('snapshot' , *args)
    def full_path(self , model_num , model_date , submodel) -> Path:
        """full model path of a given model date / num / submodel"""
        return self('archive', model_num , model_date , submodel)
    def model_file(self , model_num , model_date , submodel) -> 'ModelFile':
        """model file of a given model date / num / submodel"""
        return ModelFile(self('archive', model_num , model_date , submodel))
    def exists(self , model_num , model_date , submodel) -> bool:
        """check if a given model date / num / submodel exists"""
        path = self.full_path(model_num , model_date , submodel)
        return path.exists() and len([s for s in path.iterdir()]) > 0
    def mkdir(self , model_nums = None , exist_ok=False):
        """create model directory"""
        self.archive().mkdir(parents=True,exist_ok=exist_ok)
        if model_nums is not None: 
            [self.archive(mm).mkdir(exist_ok=exist_ok) for mm in model_nums]
        self.conf().mkdir(exist_ok=exist_ok)
        self.rslt().mkdir(exist_ok=exist_ok)
        self.snapshot().mkdir(exist_ok=exist_ok)
    @property
    def root_path(self) -> Path:
        """root of model path based on name"""
        return PATH.null_model if self.name.startswith(('db@' , 'factor@')) else PATH.model
    @property
    def base(self) -> Path:
        """model base path"""
        assert self.name , f'{self.__class__.__name__} model name is not set'
        return self.root_path.joinpath(self.name)
    @property
    def model_nums(self) -> np.ndarray:
        """model numbers"""
        return self.sub_dirs(self.archive() , as_int = True)
    @property
    def model_dates(self):
        """model dates"""
        return self.sub_dirs(self.archive(self.model_nums[-1]) , as_int = True)
    @property
    def model_submodels(self):
        """model submodels"""
        return self.sub_dirs(self.archive(self.model_nums[-1] , self.model_dates[-1]) , as_int = False)
    def load_config(self):
        """load model config"""
        from src.res.model.util.config import TrainConfig
        return TrainConfig(self.base , stage = 0)
    def next_model_date(self):
        """next model date to train"""
        from src.basic.util.calendar import CALENDAR
        config = self.load_config()
        return CALENDAR.td(self.model_dates[-1] , config.model_interval)
    
class HiddenPath:
    """hidden factor path for nn models , used for extracting hidden states"""
    def __init__(self , model_name : str , model_num : int , submodel : str) -> None:
        self.model_name , self.model_num , self.submodel = model_name , model_num , submodel
        assert self.model_name in [p.name for p in PATH.hidden.iterdir()] , \
            f'Hidden path does not contains hidden data of {self.model_name}'

    def __repr__(self): return f'{self.__class__.__name__}(model_name={self.model_name},model_num={self.model_num},submodel={self.submodel})'

    @classmethod
    def from_key(cls , hidden_key : str) -> 'HiddenPath':
        """
        create hidden path from hidden key
        example:
            HiddenPath.from_key('gru_day_V0.1.best')
        """
        model_name , model_num , submodel = cls.parse_hidden_key(hidden_key)
        return cls(model_name , model_num , submodel)

    @staticmethod
    def create_hidden_key(model_name : str , model_num : int , submodel : str) -> str:
        """
        create hidden key
        example:
            HiddenPath.create_hidden_key('gru_day_V0' , 1 , 'best') # gru_day_V0.1.best
        """
        return f'{model_name}.{model_num}.{submodel}'
    
    @property
    def hidden_key(self) -> str:
        """current hidden path's hidden key"""
        return self.create_hidden_key(self.model_name , self.model_num , self.submodel)
    
    @staticmethod
    def parse_hidden_key(hidden_key : str) -> tuple[str, int, str]:
        """parse hidden key"""
        model_name , model_num , submodel = hidden_key.split('.')
        assert submodel in ['best' , 'swabest' , 'swalast'] , f'{hidden_key} has invalid submodel: {submodel}'
        return model_name , int(model_num) , submodel
    
    @staticmethod
    def target_hidden_path(model_name : str , model_num : int , model_date , submodel : str) -> Path:
        """target hidden path"""
        return PATH.hidden.joinpath(model_name , str(model_num) , f'{model_date}.{submodel}.feather')
    
    def target_path(self , model_date: int) -> Path:
        """target hidden path of a given model date"""
        return self.target_hidden_path(self.model_name , self.model_num , model_date , self.submodel)
    
    def last_modified_date(self , model_date : int | None = None) -> int:
        """last modified date of the hidden path"""
        if model_date is None: 
            model_dates = self.model_dates()
            model_date = int(model_dates.max()) if len(model_dates) else -1
        return PATH.file_modified_date(self.target_path(model_date))
    
    def last_modified_time(self , model_date : int | None = None) -> int:
        """last modified time of the hidden path"""
        if model_date is None: 
            model_dates = self.model_dates()
            model_date = int(model_dates.max()) if len(model_dates) else -1
        return PATH.file_modified_time(self.target_path(model_date))

    def model_dates(self) -> np.ndarray:
        """model dates of source model"""
        suffix = f'.{self.submodel}.feather'
        parent = self.target_path(0).parent
        dates = [int(p.name.removesuffix(suffix)) for p in parent.iterdir() if p.name.endswith(suffix)]
        return np.sort(dates)

    def save_hidden_df(self , hidden_df : pd.DataFrame , model_date : int) -> None:
        """save hidden dataframe"""
        hidden_path = self.target_path(model_date)
        DB.save_df(hidden_df , hidden_path , overwrite = True)

    def get_hidden_df(self , model_date : int , exact = False) -> tuple[int, pd.DataFrame]:
        """get hidden dataframe"""
        if not exact: 
            model_date = self.latest_hidden_model_date(model_date)
        hidden_df = DB.load_df(self.target_path(model_date))
        return model_date , hidden_df
    
    def latest_hidden_model_date(self , model_date) -> int:
        """latest hidden model date"""
        possible_model_dates = self.model_dates()
        return possible_model_dates[possible_model_dates <= model_date].max()

class ModelDict:
    """model dictionary for nn/booster models"""
    __slots__ = ['state_dict' , 'booster_head' , 'booster_dict']
    def __init__(self ,
                 state_dict  : dict[str,torch.Tensor] | None = None , 
                 booster_head : dict[str,Any] | None = None ,
                 booster_dict : dict[str,Any] | None = None) -> None:
        self.state_dict = state_dict
        self.booster_head = booster_head
        self.booster_dict = booster_dict

    def __repr__(self): return f'{self.__class__.__name__}(state_dict={self.state_dict},booster_head={self.booster_head},booster_dict={self.booster_dict})'

    def reset(self) -> None:
        """reset model dictionary"""
        self.state_dict = None
        self.booster_head = None
        self.booster_dict = None

    def save(self , path : str | Path , stack = False) -> None:
        """uniformly save model dictionary"""
        if isinstance(path , str): 
            path = Path(path)
        path.mkdir(parents=True,exist_ok=True)
        for key in self.__slots__:
            if (value := getattr(self , key)) is not None:
                torch.save(value , path.joinpath(f'{key}.stack.pt' if stack else f'{key}.pt'))

    @property
    def is_valid(self) -> bool:
        """check if model dictionary is valid"""
        if self.state_dict is not None:
            assert self.booster_dict is None 
        else:
            assert self.booster_head is None
        return True

class ModelFile:
    """model file for nn/booster models"""
    def __init__(self , model_path : Path) -> None:
        self.model_path = model_path
    def __getitem__(self , key): return self.load(key)
    def __repr__(self): return f'{self.__class__.__name__}(path={self.model_path})'
    def load(self , key : str) -> Any:
        """load model dictionary"""
        assert key in ModelDict.__slots__ , (key , ModelDict.__slots__)
        path = self.model_path.joinpath(f'{key}.pt')
        return torch_load(path , map_location='cpu') if path.exists() else None
    def exists(self) -> bool: 
        """check if model file exists"""
        return any([self.model_path.joinpath(f'{key}.pt').exists() for key in ModelDict.__slots__])
    def model_dict(self) -> ModelDict:
        """load model dictionary"""
        return ModelDict(**{key:self.load(key) for key in ModelDict.__slots__})

class ModelDBMapping:
    """model db mapping definition"""
    def __init__(self ,  name : str , src : str , key : str , col : str | list[str] | None = None) -> None:
        self.name = name
        self.src = src
        self.key = key
        self.col = [col] if isinstance(col , str) else col
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name},src={self.src},key={self.key},col={self.col})'
    
    @classmethod
    def from_yaml(cls , name : str , base_path : ModelPath | Path | str | None = None) -> 'ModelDBMapping':
        base_path = ModelPath(base_path)
        mapping : dict[str,Any] = PATH.read_yaml(base_path.conf('registry' , 'db_models_mapping'))
        return cls(name , **mapping[name.removeprefix('db@')])

    @classmethod
    def from_dict(cls , name : str , mapping : dict[str,Any]) -> 'ModelDBMapping':
        return cls(name , **mapping[name.removeprefix('db@')])

    def load_block(self , start_dt : int , end_dt : int , silent = True):
        from src.data.loader import BlockLoader
        return BlockLoader(db_src = self.src , db_key = self.key , feature = self.col).load(start_dt , end_dt , silent = silent)
    
class RegisteredModel(ModelPath):
    '''
    for a registeredmodel to predict recent/history data
    model dict stored in configs/registry/update_models.yaml
    '''
    START_DT = 20170101 if MACHINE.server else 20170101
    FMP_STEP = 5
    MODEL_DICT : dict[str,dict[str,Any]] = MACHINE.configs('registry' , 'registered_models')

    def __init__(self, pred_name : str , name: str | Any = None , 
                 submodel : Literal['best' , 'swalast' , 'swabest'] | Any = None ,
                 num  : int | list[int] | range | Literal['all'] | Any = None , 
                 start_dt = START_DT , assertion = True) -> None:
        if assertion:
            assert pred_name in self.MODEL_DICT , f'{pred_name} is not a registered model'
        if pred_name in self.MODEL_DICT:
            reg_dict = self.MODEL_DICT[pred_name]
            name     = reg_dict['name']
            submodel = reg_dict['submodel']
            num      = reg_dict['num']

        self.pred_name = pred_name
        super().__init__(name)
        self.submodel = submodel
        self.num = num
        self.model_path = ModelPath(self.name)
        self.start_dt = start_dt
        assert start_dt > 20070101 , f'start_dt must be a valid date , got {start_dt}'

    def __repr__(self) -> str:  
        return f'{self.__class__.__name__}(pred_name={self.pred_name},name={self.name},submodel={self.submodel},num={str(self.num)})'
    
    @classmethod
    def SelectModels(cls , pred_names : list[str] | str | None = None) -> list['RegisteredModel']:
        """select registered models"""
        if pred_names is None: 
            pred_names = list(cls.MODEL_DICT.keys())
        if isinstance(pred_names , str): 
            pred_names = [pred_names]
        return [cls(key) for key in pred_names]
    
    @property
    def pred_dates(self) -> np.ndarray:
        """model pred dates"""
        return DB.dates('pred' , self.pred_name)
    
    @property
    def pred_target_dates(self) -> np.ndarray:
        """model pred target dates"""
        start_dt = max(self.start_dt , CALENDAR.td(min(self.model_dates) , 1))
        end_dt = None
        return CALENDAR.td_within(start_dt , end_dt)
    
    @property
    def fmp_dates(self) -> np.ndarray:
        """model factor portfolio dates"""
        return DB.dir_dates(PATH.fmp.joinpath(self.pred_name))
    
    @property
    def fmp_target_dates(self) -> np.ndarray:
        """model factor portfolio target dates"""
        return self.pred_target_dates[::self.FMP_STEP]
    
    def save_pred(self , df : pd.DataFrame , date : int | Any , overwrite = False) -> None:
        """save model pred"""
        DB.save(df , 'pred' , self.pred_name , date , overwrite)

    def load_pred(self , date : int , verbose = True , **kwargs) -> pd.DataFrame:
        """load model pred"""
        df = DB.load('pred' , self.pred_name , date , verbose = verbose , **kwargs)
        if df.empty: 
            return df
        if self.pred_name not in df.columns:
            assert self.name in df.columns , f'{self.pred_name} or {self.name} not in df.columns : {df.columns}'
            df = df.rename(columns={self.name:self.pred_name})
            self.save_pred(df , date , overwrite = True)
        return df

    def save_fmp(self , df : pd.DataFrame , date : int | Any , overwrite = False) -> None:
        """save model factor portfolio"""
        if df.empty: 
            return
        path = PATH.fmp.joinpath(self.pred_name , f'{self.pred_name}.{date}.feather')
        DB.save_df(df , path , overwrite = overwrite)

    def load_fmp(self , date : int , verbose = True , **kwargs) -> pd.DataFrame:
        """load model factor portfolio"""
        path = PATH.fmp.joinpath(self.pred_name , f'{self.pred_name}.{date}.feather')
        if not path.exists(): 
            if verbose: 
                print(f'{path} does not exist')
            return pd.DataFrame()
        return pd.read_feather(path , **kwargs)
    
    @property
    def account_dir(self) -> Path:
        """model factor portfolio account directory"""
        return PATH.fmp_account.joinpath(self.pred_name)

class HiddenExtractingModel(ModelPath):
    '''
    for a registeredmodel to extract hidden states
    model dict stored in configs/registry/update_models.yaml
    '''
    MODEL_DICT : dict[str,dict[str,Any]] = MACHINE.configs('registry' , 'hidden_models')
    def __init__(self , hidden_name : str , name: str | Any = None , 
                 submodels : list | np.ndarray | Literal['best' , 'swalast' , 'swabest'] | None = None ,
                 nums : list | np.ndarray | int | None = None , assertion = True):
        if assertion:
            assert hidden_name in self.MODEL_DICT , f'{hidden_name} is not a registered model'
        if hidden_name in self.MODEL_DICT:
            reg_dict  = self.MODEL_DICT[hidden_name]
            name      = reg_dict['name']
            submodels = reg_dict['submodels']
            nums      = reg_dict['nums']

        self.hidden_name = hidden_name
        super().__init__(name)
        self.submodels = submodels
        self.nums = nums
        self.model_path = ModelPath(self.name)

    def __repr__(self) -> str:  
        return f'{self.__class__.__name__}(hidden_name={self.hidden_name},name={self.name},submodels={self.submodels},nums={str(self.nums)})'

    @classmethod
    def SelectModels(cls , hidden_names : list[str] | str | None = None) -> list['HiddenExtractingModel']:   
        """select hidden models"""
        if hidden_names is None: 
            hidden_names = list(cls.MODEL_DICT.keys())
        if isinstance(hidden_names , str): 
            hidden_names = [hidden_names]
        return [cls(key) for key in hidden_names]