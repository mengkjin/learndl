import torch
import numpy as np
import pandas as pd

import shutil
from datetime import datetime , timedelta
from pathlib import Path
from typing import Any , Literal

from src.proj import PATH , Logger , LogFile , Proj , DB 
from src.proj.calendar import CALENDAR
from src.proj.func import torch_load

from .abc import MODEL_SETTINGS , parse_model_input , combine_full_name , TYPE_MODULE_TYPES , is_null_module_type

__all__ = ['ModelPath' , 'HiddenPath' , 'ModelDict' , 'ModelFile' , 'PredictionModel' , 'HiddenExtractionModel']

class ModelPath:
    f"""
    model path
    access stored model in learndl/models
    example:
        model_path = ModelPath('gru_day_V0')

    should supply a subdirectory under models/ , or a string that contains at least 'module_name@model_name' , or a ModelPath object
    full_name: [st@]module_type@module_name@model_name[@index]
        st: optional, short test mode
        module_type: nn, boost, db, factor
        module_name: module name under the module type
        model_name: model name , custom, can contain data types and hidden types
        index: optional, model index
    
    """

    def __new__(cls , model_input : 'ModelPath |Path | str | None' , *args , **kwargs) -> 'ModelPath':
        if isinstance(model_input , ModelPath):
            return model_input
        else:
            return super().__new__(cls)

    def __init__(self , model_input : Path | str | None | Any) -> None:
        if not isinstance(model_input , self.__class__):
            self.parse_input(model_input)

    def __bool__(self):         
        return bool(self.full_name)
    def __repr__(self) -> str:  
        return f'{self.__class__.__name__}(full_name={self.full_name})'
    def __eq__(self , other : 'ModelPath'):
        return self.full_name == other.full_name

    def parse_input(self , model_input : Path | str | None):
        parsed_model_input = parse_model_input(model_input)
        self.full_name : str = parsed_model_input.pop('full_name')
        self.full_name_kwargs : dict[str,Any] = parsed_model_input
        assert self.full_name_kwargs['st'] in ['st' , ''] , f'st {self.full_name_kwargs['st']} is not valid'
        if isinstance(model_input , Path):
            assert model_input == self.base , f'model_input {model_input} is not the same as base {self.base} , should adjust mannually'

    @property
    def is_short_test(self) -> bool:
        return self.full_name_kwargs['st'] == 'st'
    @property
    def is_null_model(self) -> bool:
        return bool(self) and is_null_module_type(self.module_type)
    @property
    def module_type(self) -> TYPE_MODULE_TYPES:
        return self.full_name_kwargs['module_type']
    @property
    def model_module(self) -> str:
        return self.full_name_kwargs['module_name']
    @property
    def full_module_name(self) -> str:
        return f'{self.module_type}@{self.model_module}' if self else ''
    @property
    def model_clean_name(self) -> str:
        return self.full_name_kwargs['model_clean_name']
    @property
    def model_name(self) -> str:
        return f'{self.model_clean_name}@{self.model_name_index}' if self.model_name_index > 1 else self.model_clean_name
    @property
    def model_name_index(self) -> int:
        index_str = self.full_name_kwargs['model_name_index']
        if index_str:
            index = int(index_str)
            assert index >= 2 , f'model_name_index {index} must be greater than or equal to 2'
        else:
            index = 1
        return index
    @property
    def base(self) -> Path:
        if self.full_name == '':
            return Path('')
        if self.is_short_test:
            return PATH.model_st / self.full_name.removeprefix('st@')
        else:
            match self.module_type:
                case 'nn':
                    return PATH.model_nn / f'{self.model_module}@{self.model_name}'
                case 'boost':
                    return PATH.model_boost / f'{self.model_module}@{self.model_name}'
                case 'factor':
                    return PATH.model_factor / self.model_module
                case _:
                    raise ValueError(f'Invalid module type [{self.module_type}]')

    @property
    def root_path(self) -> Path:
        return self.base.parent

    @property
    def log_file(self) -> LogFile:
        return LogFile.initiate('model' , 'operation' , f'{self.full_name}')

    @property
    def is_resumable(self) -> bool:
        return any(p.is_file() for p in self.archive().rglob('*'))

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
    

    def with_new_index(self , index : int):
        assert index > 0 , f'index {index} must be greater than 0'
        new_full_name_kwargs = self.full_name_kwargs | {'model_name_index' : index}
        self.parse_input(combine_full_name(**new_full_name_kwargs))
        return self

    def with_full_name(self , full_name : str):
        self.parse_input(full_name)
        return self

    def replace_by(self , other : 'ModelPath'):
        self.parse_input(other.full_name)
        return self

    @staticmethod
    def sub_dirs(path : Path , as_int = False) -> np.ndarray:
        """sub directories"""
        arr = [sub.name for sub in path.iterdir() if sub.is_dir()]
        if as_int: 
            arr = np.array([v for v in arr if v.isdigit()]).astype(int)
        return np.sort(arr)

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
    def conf_file(self , *args) -> Path:
        """model config file"""
        return self.conf(*args).with_suffix('.yaml')
    def rslt(self , *args) -> Path:     
        """model results path"""
        return self('results' , *args)
    def snapshot(self , *args) -> Path:
        """model snapshot path"""
        return self('snapshot' , *args)
    def log(self , *args) -> Path:
        """model log path"""
        return self('log' , *args)
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
        self.rslt().mkdir(exist_ok=True)
        self.snapshot().mkdir(exist_ok=True)
    def clear_model_path(self):
        """clear model directory , but archive directory will protected , log directory will be remained"""
        if self.is_resumable and not self.is_short_test:
            Logger.error(f'{self} is resumable and not a short test model, cannot clear , you have to delete it manually')
            return
        else:
            [shutil.rmtree(folder) for folder in [self.archive() , self.conf() , self.rslt() , self.snapshot()]]
    
    def remove_model_path(self):
        """delete model directory and log file , will ask for confirmation if not a short test model"""
        if not self.is_short_test:
            confirm = input(f'{self} is not a short test model, are you sure you want to delete it? (y/n)')
            if confirm != 'y':
                return
        if self.base.exists():
            shutil.rmtree(self.base , ignore_errors=True)
        self.log_file.unlink(confirm=False)

    def confirm(self , model_module : str | None = None):
        """confirm model path"""
        if model_module:
            if self.model_module != model_module:
                raise ValueError(f'model_module of {self} is {self.model_module}, does not match given model_module {model_module}')
        if self.is_null_model:
            assert self.model_module == self.model_name , f'{self} is a null model, {self.model_module} and {self.model_name} do not match'

    def find_resumable_candidates(self , sort = True) -> list['ModelPath']:
        if self.is_null_model and self.model_module == self.model_name:
            return []
        else:
            candidates = []
            for path in self.root_path.iterdir():
                model_path = self if path == self.base else ModelPath(path)
                if model_path.model_clean_name == self.model_clean_name:
                    candidates.append(model_path)
            if sort:
                candidates.sort(key = lambda x: x.model_name_index)
            return candidates

    def iter_model_archives(self , start_model_date : int = -1 , end_model_date : int = 99991231):
        if not self.archive().exists():
            return
        for model_num_path in self.archive().iterdir():
            if model_num_path.is_file():
                continue
            model_num = int(model_num_path.name)
            for model_date_path in model_num_path.iterdir():
                if model_date_path.is_file():
                    continue
                model_date = int(model_date_path.name)
                if model_date < start_model_date or model_date > end_model_date:
                    continue
                for submodel_path in model_date_path.iterdir():
                    if submodel_path.is_file():
                        continue
                    submodel = submodel_path.name
                    for file in submodel_path.rglob('*.*'):
                        if file.is_dir():
                            continue
                        yield model_num , model_date , submodel , file
                        
    def load_config(self):
        """load model config"""
        from src.res.model.util.config import ModelConfig
        return ModelConfig(self.base , stage = 0)
    def next_model_date(self):
        """next model date to train"""
        from src.proj.calendar import CALENDAR
        config = self.load_config()
        return CALENDAR.td(self.model_dates[-1] , config.interval).as_int()
    def collect_model_archives(self , start_model_date : int = -1 , end_model_date : int = 99991231) -> list[Path]:
        """collect model archive paths"""
        return [p for _,_,_,p in self.iter_model_archives(start_model_date , end_model_date)]

    def log_operation(self , operation : str | None = None):
        if operation is None or not self:
            return
        self.log_file.write(operation)

    def check_last_operation(self , operation : str | None = None , interval_hours : int = 24) -> tuple[datetime | None , timedelta , bool]:
        """check if the last operation is within the interval
        Args:
            category (str | None): the category of the operation
            interval_hours (int): the interval in hours
        Returns:
            tuple[last_time , time_elapsed , skip]: the last operation time, the time elapsed, and whether to skip the operation
            - last_time: the time of the last operation , if no operation logs are found, return 2000-01-01
            - time_elapsed: the time elapsed since the last operation
            - skip: if the last operation is inside the interval and should be skipped
        """
        last_time = datetime(2000,1,1)
        entries = self.log_file.read() if operation else []
        for entry in entries:
            if entry.title == operation:
                last_time = entry.timestamp
                break

        time_elapsed = datetime.now() - last_time
        skip = time_elapsed.total_seconds() / 3600  < interval_hours
        return last_time , time_elapsed , skip

    def guess_module(self) -> str:
        return str(PATH.read_yaml(self.conf_file('train', 'model'))['module']).lower().replace(' ' , '').replace('/', '@')

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

    def save_hidden_df(self , hidden_df : pd.DataFrame , model_date : int , indent : int = 1 , vb_level : int = 2) -> None:
        """save hidden dataframe"""
        hidden_path = self.target_path(model_date)
        DB.save_df(hidden_df , hidden_path , overwrite = True , prefix = f'Hidden States' , indent = indent , vb_level = vb_level)

    def get_hidden_df(self , model_date : int , exact = False , indent : int = 1 , vb_level : int = 2) -> tuple[int, pd.DataFrame]:
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
    """model dictionary for nn/boost models"""
    __slots__ = ['state_dict' , 'boost_head' , 'boost_dict']
    def __init__(self ,
                 state_dict  : dict[str,torch.Tensor] | None = None , 
                 boost_head : dict[str,Any] | None = None ,
                 boost_dict : dict[str,Any] | None = None) -> None:
        self.state_dict = state_dict
        self.boost_head = boost_head
        self.boost_dict = boost_dict

    def __repr__(self): return f'{self.__class__.__name__}(state_dict={self.state_dict},boost_head={self.boost_head},boost_dict={self.boost_dict})'

    def reset(self) -> None:
        """reset model dictionary"""
        self.state_dict = None
        self.boost_head = None
        self.boost_dict = None

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
            assert self.boost_dict is None 
        else:
            assert self.boost_head is None
        return True

class ModelFile:
    """model file for nn/boost models"""
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

class PredictionModel(ModelPath):
    '''
    for a prediction model to predict recent/history data
    model dict stored in configs/proj/model_settings.yaml file under prediction section
    '''
    START_DT = 20170101
    FMP_STEP = 5
    MODEL_DICT : dict[str,dict[str,Any]] = MODEL_SETTINGS['prediction']

    def __new__(cls , *args , **kwargs) -> 'PredictionModel | Any':
        return super().__new__(cls , *args , **kwargs)

    def __init__(self, pred_name : str , name: str | Any = None , 
                 submodel : Literal['best' , 'swalast' , 'swabest'] | Any = None ,
                 num  : int | list[int] | range | Literal['all'] | Any = None , 
                 start_dt = START_DT , assertion = True) -> None:
        if assertion:
            assert pred_name in self.MODEL_DICT , f'{pred_name} is not a prediction model'
        if pred_name in self.MODEL_DICT:
            reg_dict = self.MODEL_DICT[pred_name]
            name     = reg_dict['name']
            submodel = reg_dict['submodel']
            num      = reg_dict['num']

        self.pred_name = pred_name
        super().__init__(name)
        self.submodel = submodel
        self.num = num
        self.model_path = ModelPath(self.full_name)
        self.start_dt = start_dt
        assert start_dt > 20070101 , f'start_dt must be a valid date , got {start_dt}'

    def __repr__(self) -> str:  
        return f'{self.__class__.__name__}(pred_name={self.pred_name},full_name={self.full_name},submodel={self.submodel},num={str(self.num)})'
    
    @classmethod
    def SelectModels(cls , pred_names : list[str] | str | None = None) -> list['PredictionModel']:
        """select prediction models"""
        if pred_names is None: 
            pred_names = list(cls.MODEL_DICT.keys())
        if isinstance(pred_names , str): 
            pred_names = [pred_names]
        return [cls(key) for key in pred_names]

    @classmethod
    def CollectModelArchives(cls , pred_names : list[str] | str | None = None , start_model_date : int = -1 , end_model_date : int = 99991231) -> list[Path | Any]:
        paths : list[Path] = []
        for model in cls.SelectModels(pred_names):
            paths.extend(model.model_path.collect_model_archives(start_model_date , end_model_date))
        return paths

    @classmethod
    def PackModelArchives(cls , start_model_date : int = -1 , end_model_date : int = 99991231) -> Path:
        files = cls.CollectModelArchives(start_model_date = start_model_date , end_model_date = end_model_date)
        path = PATH.updater.joinpath('model_archives').joinpath(f'model_archives_{start_model_date}_{end_model_date}.tar')
        DB.pack_files_to_tar(files , path , overwrite = True , indent = 0 , vb_level = 1)
        return path

    @classmethod
    def UnpackModelArchives(cls , path : Path | str | None = None , delete_tar = True , overwrite = False) -> None:
        if path is None:
            paths = [p for p in PATH.main.glob('*.tar') if p.name.startswith('model_archives_')]
            paths += [p for p in PATH.updater.joinpath('model_archives').glob('*.tar')]
        else:
            paths = [Path(path)]
        
        for path in paths:
            DB.unpack_files_from_tar(path , PATH.main , overwrite = overwrite , indent = 0 , vb_level = 1)
            if delete_tar:
                path.unlink()
            
    @property
    def pred_dates(self) -> np.ndarray:
        """model pred dates"""
        return DB.dates('pred' , self.pred_name)
    
    @property
    def pred_target_dates(self) -> np.ndarray:
        """model pred target dates"""
        start_dt = max(self.start_dt , int(CALENDAR.td(min(self.model_dates) , 1)))
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
    
    def save_pred(self , df : pd.DataFrame , date : int | Any , overwrite = False , indent : int = 1 , vb_level : int = 2 , reason : str = '') -> None:
        """save model pred"""
        DB.save(df , 'pred' , self.pred_name , date , overwrite = overwrite , indent = indent , vb_level = vb_level , reason = reason)

    def load_pred(self , date : int , indent = 1 , vb_level : int = 2 , **kwargs) -> pd.DataFrame:
        """load model pred"""
        df = DB.load('pred' , self.pred_name , date , indent = indent , vb_level = vb_level , **kwargs)
        if not df.empty and self.pred_name not in df.columns:
            assert self.model_clean_name in df.columns or self.model_name in df.columns , \
                f'{self.pred_name} / {self.model_clean_name} / {self.model_name} not in df.columns : {df.columns}'
            df = df.rename(columns={self.model_clean_name:self.pred_name , self.model_name:self.pred_name})
            self.save_pred(df , date , overwrite = True , indent = indent , vb_level = Proj.vb.max , reason = f'column rename from {self.model_clean_name} to {self.pred_name}')
        return df

    def save_fmp(self , df : pd.DataFrame , date : int | Any , overwrite = False , indent = 1 , vb_level : int = 2) -> None:
        """save model factor portfolios for a given date (multiple portfolios in one dataframe)"""
        path = PATH.fmp.joinpath(self.pred_name , f'{self.pred_name}.{date}.feather')
        DB.save_df(df , path , overwrite = overwrite , prefix = f'Model FMP' , indent = indent , vb_level = vb_level)

    def load_fmp(self , date : int , vb_level : int = 2 , **kwargs) -> pd.DataFrame:
        """load model factor portfolios for a given date (multiple portfolios in one dataframe)"""
        path = PATH.fmp.joinpath(self.pred_name , f'{self.pred_name}.{date}.feather')
        if not path.exists(): 
            Logger.alert1(f'{path} does not exist' , vb_level = vb_level)
        return DB.load_df(path)
    
    @property
    def account_dir(self) -> Path:
        """model factor portfolio account directory"""
        return PATH.fmp_account.joinpath(self.pred_name)

class HiddenExtractionModel(ModelPath):
    '''
    for a hidden extraction model to extract hidden states
    model dict stored in configs/proj/model_settings.yaml file under hidden_extraction section
    '''
    MODEL_DICT : dict[str,dict[str,Any]] = MODEL_SETTINGS['hidden_extraction']
    def __new__(cls , *args , **kwargs) -> 'HiddenExtractionModel | Any':
        return super().__new__(cls , *args , **kwargs)

    def __init__(self , hidden_name : str , name: str | Any = None ,    
                 submodels : list | np.ndarray | Literal['best' , 'swalast' , 'swabest'] | None = None ,
                 nums : list | np.ndarray | int | None = None , assertion = True):
        if assertion:
            assert hidden_name in self.MODEL_DICT , f'{hidden_name} is not a hidden extraction model'
        if hidden_name in self.MODEL_DICT:
            reg_dict  = self.MODEL_DICT[hidden_name]
            name      = reg_dict['name']
            submodels = reg_dict['submodels']
            nums      = reg_dict['nums']

        self.hidden_name = hidden_name
        super().__init__(name)
        self.submodels = submodels
        self.nums = nums
        self.model_path = ModelPath(self.full_name)

    def __repr__(self) -> str:  
        return f'{self.__class__.__name__}(hidden_name={self.hidden_name},full_name={self.full_name},submodels={self.submodels},nums={str(self.nums)})'

    @classmethod
    def SelectModels(cls , hidden_names : list[str] | str | None = None) -> list['HiddenExtractionModel']:   
        """select hidden models"""
        if hidden_names is None: 
            hidden_names = list(cls.MODEL_DICT.keys())
        if isinstance(hidden_names , str): 
            hidden_names = [hidden_names]
        return [cls(key) for key in hidden_names]