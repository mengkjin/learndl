"""
ModelPath: The all-in-one model path class for model training and inference.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import shutil
from datetime import datetime , timedelta
from functools import cached_property
from pathlib import Path
from typing import Any , Literal

from src.proj import PATH , DB , CALENDAR , Const , Logger , Base , Save , Load , Dates
from src.proj.core import as_int_array
from src.proj.bases import ModuleType

from .basic import parse_model_input , combine_full_name , is_null_module_type
from .model_file import ModelFile

__all__ = ['ModelPath' , 'PredictorPath']

class ModelPath:
    """
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
    def __init__(self , model_input : str | Base.strPath | ModelPath | None , **kwargs) -> None:
        super().__init__(**kwargs)

        self.parse_input(model_input.base if isinstance(model_input , ModelPath) else model_input)

    def __bool__(self):         
        return bool(self.full_name)
    def __repr__(self) -> str:  
        return f'{self.__class__.__name__}(full_name={self.full_name})'
    def __eq__(self , other : ModelPath):
        return self.full_name == other.full_name

    def parse_input(self , model_input : Base.strPath | None):
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
    def module_type(self) -> ModuleType:
        return ModuleType(self.full_name_kwargs['module_type'])
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
    def relative_base(self) -> str:
        return str(self.base.relative_to(PATH.model))

    @property
    def root_path(self) -> Path:
        return self.base.parent

    @cached_property
    def log_file(self):
        from src.proj.log.logfile import LogFile
        return LogFile.initialize('model' , 'operation' , f'{self.full_name}')

    @property
    def is_resumable(self) -> bool:
        return self.archive().exists() and any(p.is_file() for p in self.archive().rglob('*'))

    @property
    def model_nums(self) -> np.ndarray:
        """model numbers"""
        return self.sub_dirs(self.archive() , as_int = True)
    @property
    def model_dates(self) -> np.ndarray:
        """model dates"""
        if len(self.model_nums) == 0:
            return np.array([])
        return self.sub_dirs(self.archive(self.model_nums[-1]) , as_int = True)
    @property
    def model_submodels(self) -> np.ndarray:
        """model submodels"""
        if len(self.model_nums) == 0 or len(self.model_dates) == 0:
            return np.array([])
        return self.sub_dirs(self.archive(self.model_nums[-1] , self.model_dates[-1]) , as_int = False)

    def with_new_index(self , index : int):
        assert index > 0 , f'index {index} must be greater than 0'
        new_full_name_kwargs : dict[str,Any] = self.full_name_kwargs | {'model_name_index' : index}
        self.parse_input(combine_full_name(**new_full_name_kwargs))
        return self

    def with_full_name(self , full_name : str):
        self.parse_input(full_name)
        return self

    def replace_by(self , other : ModelPath):
        self.parse_input(other.full_name)
        return self

    @staticmethod
    def sub_dirs(path : Path , as_int = False) -> np.ndarray:
        """sub directories"""
        if not path.exists():
            return np.array([])
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
        return self('configs' , *args)
    def conf_file(self , key : str) -> Path:
        """model config file"""
        assert self , 'empty model path cannot have config files'
        return self('configs').joinpath(f"{key}.yaml")
    def rslt(self , *args) -> Path:     
        """model results path"""
        return self('results' , *args)
    def snapshot(self , *args) -> Path:
        """model snapshot path"""
        return self('snapshot' , *args)
    def full_path(self , model_num , model_date , submodel) -> Path:
        """full model path of a given model date / num / submodel"""
        return self('archive', model_num , model_date , submodel)
    def model_file(self , model_num , model_date , submodel) -> ModelFile:
        """model file of a given model date / num / submodel"""
        return ModelFile(self('archive', model_num , model_date , submodel))
    def exists(self , model_num , model_date , submodel) -> bool:
        """check if a given model date / num / submodel exists"""
        path = self.full_path(model_num , model_date , submodel)
        return path.exists() and len([s for s in path.iterdir()]) > 0
    def mkdir(self , model_nums = None , exist_ok=False):
        """create model directory"""
        if not self.base.exists():
            self.base.mkdir(parents=True,exist_ok=exist_ok)
            self.log_operation('create_model_path')
        self.archive().mkdir(exist_ok=exist_ok)
        if model_nums is not None: 
            [self.archive(mm).mkdir(exist_ok=exist_ok) for mm in model_nums]
        self.conf().mkdir(exist_ok=exist_ok)
        self.rslt().mkdir(exist_ok=True)
        self.snapshot().mkdir(exist_ok=True)

    def rename(self , new_clean_name : str):
        """rename model directory"""
        # get new full name
        if new_clean_name == self.model_clean_name:
            return self
        old_full_name = self.full_name
        new_full_name = combine_full_name(**self.full_name_kwargs | {'model_clean_name' : new_clean_name})
        assert not ModelPath(new_full_name).base.exists() , f'{new_full_name} already exists , cannot rename to it!'
        
        if (model_config_file := self.conf_file('model')).exists():
            configs = PATH.read_yaml(model_config_file) | {'model.name' : new_clean_name}
            PATH.dump_yaml(configs , model_config_file)

        if (schedule_config_file := self.conf_file('schedule')).exists():
            configs = PATH.read_yaml(schedule_config_file) | {'model.name' : new_clean_name}
            PATH.dump_yaml(configs , schedule_config_file)

        if self.log_file.host_file.exists():
            self.log_file.rename(new_full_name)
        
        if self.base.exists():
            config_file = PATH.read_yaml(self.conf_file('model'))
            config_file['model.name'] = new_clean_name
            PATH.dump_yaml(config_file , self.conf_file('model'))

            old_base_path = self.base
            self.full_name_kwargs['model_clean_name'] = new_clean_name
            self.full_name = new_full_name
            old_base_path.rename(self.base)
            self.log_operation(f'rename_model_path from {old_full_name} to {new_full_name}')
        return self

    def clear_model_path(self):
        """clear model directory , but archive directory will protected , log directory will be remained"""
        if self.is_resumable and not self.is_short_test:
            Logger.error(f'{self} is resumable and not a short test model, cannot clear , you have to delete it manually')
            return
        else:
            [shutil.rmtree(folder , ignore_errors=True) for folder in [self.archive() , self.conf() , self.rslt() , self.snapshot()]]
            self.log_operation('clear_model_path')
    
    def remove_model_path(self):
        """delete model directory and log file , will ask for confirmation if not a short test model"""
        raise NotImplementedError('remove_model_path is not implemented')

    def confirm(self , model_module : str | None = None):
        """confirm model path"""
        if model_module:
            if self.model_module != model_module:
                raise ValueError(f'model_module of {self} is {self.model_module}, does not match given model_module {model_module}')
        if self.is_null_model:
            assert self.model_module == self.model_name , f'{self} is a null model, {self.model_module} and {self.model_name} do not match'

    def find_resumable_candidates(self , sort = True , folder_not_exist = False) -> list[ModelPath]:
        if self.is_null_model and self.model_module == self.model_name:
            return []
        else:
            candidates = []
            for path in self.root_path.glob('*'):
                if path.is_file() or path.name.startswith('./$@'):
                    continue
                model_path = self if path == self.base else ModelPath(path)
                if model_path.model_clean_name == self.model_clean_name and (model_path.is_resumable or folder_not_exist):
                    candidates.append(model_path)
            if sort:
                candidates.sort(key = lambda x: x.model_name_index)
            return candidates

    def find_resumable_candidates_indices(self , folder_not_exist = False) -> list[int]:
        candidates = self.find_resumable_candidates(sort = False, folder_not_exist = folder_not_exist)
        return [mp.model_name_index for mp in candidates]

    def find_new_index(self , folder_not_exist = False) -> int:
        """find unoccupied new index for a new model"""
        candidates_indices = self.find_resumable_candidates_indices(folder_not_exist = folder_not_exist)
        if not candidates_indices:
            return 1
        return int(np.setdiff1d(np.arange(1, max(candidates_indices) + 2),candidates_indices).min())

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
        return ModelConfig(self.base , stage = 0 , resume = 1)
    def next_model_date(self):
        """next model date to train"""
        from src.proj import CALENDAR
        config = self.load_config()
        return CALENDAR.td(self.model_dates[-1] , config.interval).as_int()
    def collect_model_archives(self , start_model_date : int = -1 , end_model_date : int = 99991231) -> list[Path]:
        """collect model archive paths"""
        return [p for _,_,_,p in self.iter_model_archives(start_model_date , end_model_date)]

    def log_operation(self , operation : str | None = None):
        if operation is None or not self:
            return
        self.log_file.write(operation)

    def check_last_operation(self , operation : str | None = None , interval_hours : int = 24) -> tuple[datetime | None, timedelta, bool]:
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

    def move_to_archive(self):
        """move model to archive"""
        assert self.base.exists() , f'{self.base} does not exist'
        self.log_operation('move_to_archive')
        source = self.base
        target = PATH.model_archive.joinpath(f'{self.full_name}.{datetime.now().strftime("%Y%m%d%H%M%S")}')
        assert not target.exists() , f'{target} already exists'
        shutil.move(source , target)
        Logger.success(f'{PATH.relative(source)} has been moved to archive {PATH.relative(target)}')

    @classmethod
    def resume_from_archive(cls , start_with : str):
        """
        revert model from archive
        full_name can be the full name of the model, or the full name of the model with the datetime suffix
        """
        candidates = [path for path in PATH.model_archive.iterdir() if path.is_dir() and path.name.startswith(start_with)]
        if not candidates:
            Logger.alert1(f'No candidates found for {start_with}')
            return None
        if len(candidates) == 1:
            source = candidates[0]
        else:
            Logger.alert1(f'Multiple ({len(candidates)}) candidates found for {start_with}, please choose one by yourself:')
            for i , candidate in enumerate(candidates):
                Logger.note(f'{i+1:02d}. {PATH.relative(candidate)}')
            source = candidates[int(input(f'Which one to choose? (1-{len(candidates)}): ').strip()) - 1]
        model_path = cls(source.name.split('.')[0])
        model_path = model_path.with_new_index(model_path.find_new_index(folder_not_exist = True))
        target = model_path.base
        shutil.move(source , target)
        Logger.success(f'{PATH.relative(source)} has been reverted from archive {PATH.relative(target)}')
        model_path.log_operation('resume_from_archive')
        return model_path

class PredictorPath(ModelPath , Base.BoundLogger):
    """
    for a prediction model to predict recent/history data
    model dict stored in configs/proj/model_settings.yaml file under prediction section
    """
    START_DATE = 20170101
    FMP_STEP = 5
    MODEL_DICT : dict[str,dict[str,Any]] = Const.Model.strategies['prediction']

    def __init__(
        self, model_input : str | Base.strPath | ModelPath | None , 
        model_num : Base.intNums | Literal['all'] | None = None ,
        submodel : str = 'best' , pred_name : str | None = None , * , 
        indent : int = 0 , vb_level : Any = 1 , **kwargs
    ) -> None:
        super().__init__(model_input = model_input , indent=indent, vb_level=vb_level, **kwargs)
        self._model_num = model_num
        self._submodel = submodel
        self._pred_name = pred_name

    def to_model_path(self) -> ModelPath:
        return ModelPath(self.base)

    def __repr__(self) -> str:  
        return (
            f'{self.__class__.__name__}(pred_name={self.pred_name},full_name={self.full_name},'
            f'model_num={str(self._model_num)},submodel={self._submodel})'
        )

    def __eq__(self , other : PredictorPath):
        return (
            self.full_name == other.full_name and 
            np.array_equal(self.use_model_nums , other.use_model_nums) and 
            self._submodel == other._submodel
        )

    def to_model(self):
        from src.res.model.model_module.application import ArchivedPredictorModel
        return ArchivedPredictorModel(self)
    
    @property
    def is_stored_strategy(self) -> bool:
        return self.pred_name in self.MODEL_DICT

    @property
    def start(self) -> int:
        return self.START_DATE

    @property
    def pred_name(self) -> str:
        if self._pred_name is None:
            self._pred_name = self.model_name
        return self._pred_name

    @property
    def use_model_nums(self) -> np.ndarray:
        if self._model_num == 'all':
            return self.model_nums
        else:
            return as_int_array(self._model_num)

    @property
    def use_submodel(self) -> str:
        return self._submodel
            
    @property
    def pred_dates(self) -> Dates:
        """model pred dates"""
        return DB.dates('pred' , self.pred_name)
    
    @property
    def pred_target_dates(self) -> Dates:
        """model pred target dates"""
        if len(self.model_dates) == 0:
            return Dates()
        start = max(self.start , int(CALENDAR.td(min(self.model_dates) , 1)))
        end = None
        return Dates(start , end , updated = True)
    
    @property
    def fmp_dates(self) -> Dates:
        """model factor portfolio dates"""
        return DB.dir_dates(PATH.fmp_port.joinpath(self.pred_name))
    
    @property
    def fmp_target_dates(self) -> Dates:
        """model factor portfolio target dates"""
        return self.pred_target_dates[::self.FMP_STEP]
    
    def save_pred(self , df : pd.DataFrame , date : int , overwrite = False , reason : str = '') -> None:
        """save model pred"""
        df = df.rename(columns={self.model_clean_name:self.pred_name , self.model_name:self.pred_name})
        DB.save(df , 'pred' , self.pred_name , date , overwrite = overwrite , indent = self.indent + 1 , vb_level = self.vb_level + 1 , reason = reason)

    def load_pred(self , date : int , closest = False , **kwargs) -> pd.DataFrame:
        """load model pred"""
        df = DB.load('pred' , self.pred_name , date , closest = closest , vb_level = 'max' , **kwargs)
        if not df.empty and self.pred_name not in df.columns:
            assert self.model_clean_name in df.columns or self.model_name in df.columns , \
                f'{self.pred_name} / {self.model_clean_name} / {self.model_name} not in df.columns : {df.columns}'
            df = df.rename(columns={self.model_clean_name:self.pred_name , self.model_name:self.pred_name})
            self.save_pred(df , date , overwrite = True , reason = f'column rename from {self.model_clean_name} to {self.pred_name}')
        return df

    def save_fmp(self , df : pd.DataFrame , date : int , overwrite = False) -> None:
        """save model factor portfolios for a given date (multiple portfolios in one dataframe)"""
        path = PATH.fmp_port.joinpath(self.pred_name , f'{self.pred_name}.{date}.feather')
        Save.df(df , path , overwrite = overwrite , prefix = f'Model FMP' , indent = self.indent + 1 , vb_level = self.vb_level + 1)

    def load_fmp(self , date : int , **kwargs) -> pd.DataFrame:
        """load model factor portfolios for a given date (multiple portfolios in one dataframe)"""
        path = PATH.fmp_port.joinpath(self.pred_name , f'{self.pred_name}.{date}.feather')
        if not path.exists(): 
            Logger.alert1(f'{path} does not exist')
        return Load.df(path)
    
    @property
    def account_dir(self) -> Path:
        """model factor portfolio account directory"""
        return PATH.fmp_acc.joinpath(self.pred_name)

    @classmethod
    def SelectModels(cls , pred_names : Base.alias.NamesType = None) -> list[PredictorPath]:
        """select prediction models"""
        pred_names = Base.ensure_name_list(pred_names,list(cls.MODEL_DICT.keys()))
        paths = []
        for key in pred_names:
            if key in cls.MODEL_DICT:
                reg_dict = cls.MODEL_DICT[key]
                name     = reg_dict['name']
                num      = reg_dict['num']
                submodel = reg_dict['submodel']
                paths.append(cls(name , num , submodel , pred_name = key))
        return paths

    @classmethod
    def CollectModelArchives(
        cls , pred_names : Base.alias.NamesType = None ,
        start_model_date : int = -1 , end_model_date : int = 99991231) -> list[Path]:
        paths : list[Path] = []
        for model in cls.SelectModels(pred_names):
            paths.extend(model.collect_model_archives(start_model_date , end_model_date))
        return paths

    @classmethod
    def PackModelArchives(
        cls , start_model_date : int = -1 , end_model_date : int = 99991231
    ) -> Path:
        files = cls.CollectModelArchives(start_model_date = start_model_date , end_model_date = end_model_date)
        path = PATH.updater.joinpath('model_archives').joinpath(f'model_archives_{start_model_date}_{end_model_date}.tar')
        Save.pack(files , path , overwrite = True , indent = 0 , vb_level = 1)
        return path

    @classmethod
    def UnpackModelArchives(
        cls , path : Base.strPath | None = None , delete_tar = True , overwrite = False
    ) -> None:
        if path is None:
            paths = [p for p in PATH.main.glob('*.tar') if p.name.startswith('model_archives_')]
            paths += [p for p in PATH.updater.joinpath('model_archives').glob('*.tar')]
        else:
            paths = [Path(path)]
        
        for path in paths:
            Load.unpack(path , PATH.main , overwrite = overwrite , indent = 0 , vb_level = 1)
            if delete_tar:
                path.unlink()