

import numpy as np
import pandas as pd
import importlib.util
import inspect

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any , Literal , Type

from ...basic import PATH
from ...data.tushare.basic import CALENDAR
from ...func.singleton import singleton_threadsafe

_FACTOR_UPDATE_JOBS : list[tuple[int , str , 'StockFactorCalculator']] = []

_FACTOR_INIT_DATE = 20070101
_FACTOR_CATEGORY0_SET = ['fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative']
_FACTOR_CATEGORY1_SET = {
    'fundamental' : ['quality' , 'growth' , 'value' , 'earning'] ,
    'analyst' : ['surprise' , 'coverage' , 'forecast' , 'adjustment'] ,
    'high_frequency' : ['hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity'] ,
    'behavior' : ['momentum' , 'volatility' , 'correlation' , 'liquidity'] ,
    'money_flow' : ['holding' , 'trading'] ,
    'alternative' : None
}

def factor_folder(factor_name : str): 
    return Path(f'{PATH.factor}/{factor_name}')

def factor_path(factor_name : str , date : int): 
    return factor_folder(factor_name).joinpath(f'{int(date) // 10000}/{factor_name}.{int(date)}.feather')

def insert_update_job(date : int , level : str , obj : 'StockFactorCalculator'):
    _FACTOR_UPDATE_JOBS.append((date , level , obj))

def perform_update_jobs(overwrite = False , show_progress = True , ignore_error = False , selective_cls : Type['StockFactorCalculator'] | None = None):
    _FACTOR_UPDATE_JOBS.sort(key=lambda x: (x[0], x[1], x[2]))
    
    for item in _FACTOR_UPDATE_JOBS[:]:
        date , level , obj = item
        if selective_cls is not None and obj.__class__ != selective_cls: continue
        
        try:
            obj.calculate(date).deploy(overwrite = overwrite , show_progress = show_progress)
        except Exception as e:
            if ignore_error:
                print(f'Factor : {obj.factor_name} update at date {date} failed: {e}')
            else:
                raise e
        _FACTOR_UPDATE_JOBS.remove(item)

class StockFactorCalculator(ABC):
    init_date : int = -1
    category0 : Literal['fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative'] | Any
    category1 : Literal['quality' , 'growth' , 'value' , 'earning' , 'surprise' , 'coverage' , 'forecast' , 
                        'adjustment' , 'hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity' , 
                        'momentum' , 'volatility' , 'correlation' , 'liquidity' , 'holding' , 'trading'] | Any = None
    description : str = ''

    def __new__(cls , *args , **kwargs):
        cls.validate_attr()
        return super().__new__(cls)

    def __init__(self):
        self.factors : dict[int , pd.DataFrame] = {}

    def __repr__(self):
        return f'{self.factor_name}(from {self.init_date} , {self.category0} , {self.category1})'

    @abstractmethod
    def calc_factor(self , date : int) -> pd.DataFrame:
        '''calculate factor value , must have secid and factor_value / factor_name columns'''
        return pd.DataFrame()

    @property
    def factor_name(self): return self.__class__.__name__
    
    @classmethod
    def factor_folder(cls , factor_name : str | None = None):
        return factor_folder(cls.__name__ if factor_name is None else factor_name)
    
    @classmethod
    def factor_path(cls , date : int | Any , mkdir = True , factor_name : str | None = None):
        path = factor_path(cls.__name__ if factor_name is None else factor_name , date)
        if mkdir and factor_name is None: path.parent.mkdir(parents=True , exist_ok=True)
        return path
    
    def factor_values(self):
        return pd.concat([df.assign(date = d) for d , df in self.factors.items()]).reset_index().set_index(['date','secid'])

    @classmethod
    def validate_attr(cls):
        '''validate attribute of factor'''
        assert cls.init_date >= _FACTOR_INIT_DATE , f'init_date should be later than {_FACTOR_INIT_DATE} , but got {cls.init_date}'

        assert cls.category0 in _FACTOR_CATEGORY0_SET , \
            f'category0 is should be in {_FACTOR_CATEGORY0_SET}, but got {cls.category0}'
        
        category1_list = _FACTOR_CATEGORY1_SET[cls.category0]
        
        if category1_list is not None:
            assert cls.category1 in category1_list , \
                f'category1 is should be in {category1_list}, but got {cls.category1}'
        else:
            assert cls.category1 , 'category1 is not set'
        
        assert cls.description , 'description is not set'

        return cls
    
    def validate_value(self , date : int , df : pd.DataFrame , strict = False):
        '''validate factor value'''

        assert 20991231 >= date >= self.init_date , \
            f'calc_date is should be in [{self.init_date} , 20991231], but got {date}'

        mininum_finite_count = 100 if strict else 0
        actual_finite_count = np.isfinite(df[self.factor_name].to_numpy()).sum()
        if actual_finite_count < mininum_finite_count:
            raise ValueError(f'factor_value must have at least {mininum_finite_count} finite values , but got {actual_finite_count}')
        
        return self

    def calculate(self , date : int | Iterable| Any):
        '''calculate factor value of a given date and store to factor_data'''
        if isinstance(date , Iterable):
            for d in date: self.calculate(d)
        else:
            date = int(date)
            assert date >= self.init_date , f'date is should be greater than or equal to {self.init_date}, but got {date}'
            df = self.calc_factor(date).reset_index().set_index('secid').\
                rename(columns={'factor_value':self.factor_name})[[self.factor_name]]
            self.factors[date] = df
        return self

    def deploy(self , strict = True , overwrite = False , show_progress = False):
        '''store factor data after calculate'''
        dates = list(self.factors.keys())
        for date in dates:
            df = self.factors.pop(date)
            path = self.factor_path(date , True)
            if path.exists() and not overwrite: 
                if show_progress: print(f'Factor : {self.factor_name} at date {date} already there')
                continue
            try:
                self.validate_value(date , df , strict = strict)
                df.to_feather(self.factor_path(date , True))
                if show_progress: print(f'Factor : {self.factor_name} at date {date} deploy successful')
            except ValueError as e:
                print(f'Factor : {self.factor_name} at date {date} is invalid: {e}')

        return self

    def load(self, date : int | Iterable | Any , factor_name : str | None = None):
        '''load factor data from storage'''
        if isinstance(date , Iterable):
            dfs = {int(d):self.load(d , factor_name) for d in date}
            dfs = [df.assign(date = d) for d , df in dfs.items() if isinstance(df , pd.DataFrame)]
            if dfs:
                return pd.concat(dfs).reset_index().set_index(['date','secid'])
            else:
                return None
        else:
            if int(date) in self.factors and factor_name is None: return self.factors[int(date)]
            factor_path = self.factor_path(date = date , factor_name = factor_name)
            if factor_path.exists():
                return pd.read_feather(factor_path)
            else:
                return None

    @classmethod
    def stored_dates(cls):
        paths = PATH.list_files(cls.factor_folder() , recur=True)
        dates = np.array(sorted(PATH.R_path_date(paths)) , dtype=int)
        return dates
    
    @classmethod
    def update_jobs(cls , start : int = -1 , end : int = 99991231 , overwrite = False , level : str | None = None):
        obj = cls()
        dates = CALENDAR.td_within(max(start , cls.init_date) , end)
        if not overwrite:
            dates = np.setdiff1d(CALENDAR.td_within(max(start , cls.init_date) , end) , cls.stored_dates())
        [insert_update_job(d , 'levelunknown' if level is None else level , obj) for d in dates]
        return obj

    @classmethod
    def Update(cls , overwrite = False , show_progress = True , ignore_error = False):
        '''update factor data from self.init_date to today'''
        perform_update_jobs(overwrite , show_progress , ignore_error , cls)
    
    @classmethod
    def factor_hierarchy(cls):
        return StockFactorHierarchy()
    
@singleton_threadsafe
class StockFactorHierarchy(ABC):
    def __init__(self):
        self.definition_path =  PATH.main.joinpath('src' , 'factor' , 'factor_definition')
        assert self.definition_path.exists() , f'{self.definition_path} does not exist'
        self.hier : dict[str , list[Type[StockFactorCalculator]]] = {}
        
        self.load()

    def load(self):        
        for level_path in self.definition_path.iterdir():
            if not level_path.is_dir(): continue

            for file_path in level_path.iterdir():
                if file_path.suffix != '.py': continue
                spec_name = f'{level_path.stem}.{file_path.stem}'
                
                spec = importlib.util.spec_from_file_location(spec_name, file_path)
                assert spec is not None and spec.loader is not None , f'{file_path} is not a valid module'
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                for _ , obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == spec_name:
                        assert issubclass(obj , StockFactorCalculator) , f'{obj} is not a subclass of StockFactorCalculator'
                        if level_path.stem not in self.hier: self.hier[level_path.stem] = []
                        self.hier[level_path.stem].append(obj)

        return self

    def factor_df(self , level : str | None = None , category0 : str | None = None , category1 : str | None = None):
        attr_list = ['__name__' , 'init_date' , 'category0' , 'category1' , 'description']
        
        df_dict = [
            [level , *[getattr(factor_cls , attr) for attr in attr_list]] 
            for level , factor_cls in self
        ]

        df = pd.DataFrame(df_dict, columns=['level' , *attr_list]).rename(columns={'__name__' : 'factor_name'})
        if level is not None: df = df[df['level'] == level]
        if category0 is not None: df = df[df['category0'] == category0]
        if category1 is not None: df = df[df['category1'] == category1]
        return df
    
    def jobs(self , as_df = True):
        if as_df:
            return pd.DataFrame(_FACTOR_UPDATE_JOBS , columns=['date' , 'level' , 'factor'])
        else:
            return _FACTOR_UPDATE_JOBS

    def __repr__(self):
        str_level_factors = [','.join(f'{level}({len(factors)})' for level , factors in self.hier.items())]
        return f'StockFactorHierarchy({str_level_factors})'

    def factor_names(self):
        return [f'{level} : {factor_cls.__name__}' for level in self.iter_levels() for factor_cls in self.iter_factors(level)]

    def __iter__(self):
        return ((level , factor_cls) for level in self.iter_levels() for factor_cls in self.iter_factors(level))

    def iter_levels(self):
        return iter(self.hier)
    
    def iter_factors(self , level : str):
        return iter(self.hier[level])
    
    def __getitem__(self , key : str):
        return self.hier[key]
    
    @classmethod
    def update_jobs(cls , start : int = -1 , end : int = 99991231 , overwrite = False):
        obj = cls()
        [factor_cls().update_jobs(start , end , overwrite , level) for level , factor_cls in obj]
        return obj
    
    @classmethod
    def Update(cls , overwrite = False , show_progress = True , ignore_error = True):
        perform_update_jobs(overwrite , show_progress , ignore_error)
