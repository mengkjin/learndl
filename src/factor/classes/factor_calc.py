

import numpy as np
import pandas as pd
import importlib.util
import inspect , threading

from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path
from typing import Any , Literal , Type , final

from ...basic import PATH
from ...data import TSData
from ...func.singleton import SingletonABCMeta

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

def validate_factor_calculator_attr(cls : Type['StockFactorCalculator']):
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


def factor_folder(factor_name : str): 
    return Path(f'{PATH.factor}/{factor_name}')

def factor_path(factor_name : str , date : int , mkdir = True): 
    path = factor_folder(factor_name).joinpath(f'{int(date) // 10000}/{factor_name}.{int(date)}.feather')
    if mkdir: path.parent.mkdir(parents=True , exist_ok=True)
    return path

def factor_stored_dates(factor_name):
    paths = PATH.list_files(factor_folder(factor_name) , recur=True)
    dates = np.array(sorted(PATH.R_path_date(paths)) , dtype=int)
    return dates

def insert_update_job(date : int , level : str , obj : 'StockFactorCalculator'):
    _FACTOR_UPDATE_JOBS.append((date , level , obj))

def perform_update_jobs(overwrite = False , show_progress = True , ignore_error = False , factor_name : str | None = None):
    _FACTOR_UPDATE_JOBS.sort(key=lambda x: (x[0], x[1], x[2]))
    
    for item in _FACTOR_UPDATE_JOBS[:]:
        date , level , obj = item
        if factor_name is not None and obj.factor_name != factor_name: continue
        
        try:
            obj.calculate(date).deploy(overwrite = overwrite , show_progress = show_progress)
        except Exception as e:
            if ignore_error:
                print(f'Factor : {obj.factor_name} update at date {date} failed: {e}')
            else:
                raise e
        _FACTOR_UPDATE_JOBS.remove(item)

class StockFactorCalculator(metaclass=SingletonABCMeta):
    init_date : int = -1
    category0 : Literal['fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative'] | Any
    category1 : Literal['quality' , 'growth' , 'value' , 'earning' , 'surprise' , 'coverage' , 'forecast' , 
                        'adjustment' , 'hf_momentum' , 'hf_volatility' , 'hf_correlation' , 'hf_liquidity' , 
                        'momentum' , 'volatility' , 'correlation' , 'liquidity' , 'holding' , 'trading'] | Any = None
    description : str = ''
    other_info : dict[str , Any] = {}

    @classmethod
    def update_info(cls , **kwargs): 
        cls.other_info.update(kwargs)
        return cls

    @abstractmethod
    def calc_factor(self , date : int) -> pd.DataFrame | pd.Series:
        '''calculate factor value , must have secid and factor_value / factor_name columns'''
        return pd.DataFrame()

    def __new__(cls , *args , **kwargs):
        validate_factor_calculator_attr(cls)
        return super().__new__(cls)

    @final
    def __init__(self):
        if not hasattr(self , 'factors'):
            self.factors : dict[int , pd.DataFrame] = {}

    def __repr__(self):
        return f'{self.factor_name}(from{self.init_date},{self.category0},{self.category1})[{len(self.factors)}dates]'
    
    def __getitem__(self , date : int):
        return self.factors[date]
    
    def __call__(self , date : int):
        if date not in self.factors: self.calculate(date)
        return self.factors[date]

    @property
    def factor_name(self): return self.__class__.__name__

    @property
    def factor_folder(self): return factor_folder(self.factor_name)

    @property
    def stored_dates(self): return factor_stored_dates(self.factor_name)
    
    def factor_path(self , date : int | Any):
        return factor_path(self.factor_name , date , mkdir = True)
    
    def factor_values(self):
        return pd.concat([df.assign(date = d) for d , df in self.factors.items()]).reset_index().set_index(['date','secid'])
    
    def validate_value(self , date : int , df : pd.DataFrame , strict = False):
        '''validate factor value'''

        assert 20991231 >= date >= self.init_date , \
            f'calc_date is should be in [{self.init_date} , 20991231], but got {date}'

        mininum_finite_count = 100 if strict else 0
        actual_finite_count = np.isfinite(df[self.factor_name].to_numpy()).sum()
        if actual_finite_count < mininum_finite_count:
            raise ValueError(f'factor_value must have at least {mininum_finite_count} finite values , but got {actual_finite_count}')
        
        return self

    def calculate(self , date : int | Iterable | Any):
        '''calculate factor value of a given date and store to factor_data'''

        if isinstance(date , Iterable):
            for d in date: self.calculate(d)
        else:
            date = int(date)
            assert date >= self.init_date , f'date is should be greater than or equal to {self.init_date}, but got {date}'

            df = self.calc_factor(date)
            if isinstance(df , pd.Series):
                df = df.rename(self.factor_name).to_frame()
            elif isinstance(df , pd.DataFrame):
                df = df.reset_index().set_index('secid').rename(columns={'factor_value':self.factor_name})[[self.factor_name]]
            else:
                raise ValueError(f'calc_factor must return a DataFrame or Series , but got {type(df)}')
            self.factors[date] = df
        return self

    def deploy(self , strict = True , overwrite = False , show_progress = False):
        '''store factor data after calculate'''
        dates = list(self.factors.keys())
        for date in dates:
            df = self.factors.pop(date)
            path = self.factor_path(date)
            if path.exists() and not overwrite: 
                if show_progress: print(f'Factor : {self.factor_name} at date {date} already there')
                continue
            try:
                self.validate_value(date , df , strict = strict)
                df.to_feather(self.factor_path(date))
                if show_progress: print(f'Factor : {self.factor_name} at date {date} deploy successful')
            except ValueError as e:
                print(f'Factor : {self.factor_name} at date {date} is invalid: {e}')

        return self

    def load(self, date : int | Iterable | Any , factor_name : str | None = None):
        '''load factor data from storage'''
        if factor_name is None: factor_name = self.factor_name
        if isinstance(date , Iterable):
            dfs = {int(d):self.load(d , factor_name) for d in date}
            dfs = [df.assign(date = d) for d , df in dfs.items() if isinstance(df , pd.DataFrame)]
            if dfs:
                return pd.concat(dfs).reset_index().set_index(['date','secid'])
            else:
                return None
        else:
            if int(date) in self.factors and factor_name is None: return self.factors[int(date)]
            path = factor_path(factor_name , date , False)
            if path.exists():
                return pd.read_feather(path)
            else:
                return None
    
    def update_jobs(self , start : int = -1 , end : int = 99991231 , overwrite = False , level : str | None = None):
        dates = TSData.CALENDAR.td_within(max(start , self.init_date) , end)
        if not overwrite:
            dates = np.setdiff1d(TSData.CALENDAR.td_within(max(start , self.init_date) , end) , self.stored_dates)
        [insert_update_job(d , 'levelunknown' if level is None else level , self) for d in dates]
        return self

    @classmethod
    def Update(cls , overwrite = False , show_progress = True , ignore_error = False):
        '''update factor data from self.init_date to today'''
        perform_update_jobs(overwrite , show_progress , ignore_error , cls.__name__)
    
class StockFactorHierarchy:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.definition_path =  PATH.main.joinpath('src_factor_definition')
        assert self.definition_path.exists() , f'{self.definition_path} does not exist'
        self.load()

    def load(self):     
        self.pool : dict[str , Type[StockFactorCalculator]] = {}   
        self.hier : dict[str , list[Type[StockFactorCalculator]]] = {}
        for level_path in self.definition_path.iterdir():
            if not level_path.is_dir() or level_path.stem == 'ignore': continue

            for file_path in level_path.iterdir():
                if file_path.suffix != '.py': continue
                spec_name = f'{level_path.stem}.{file_path.stem}'
                
                spec = importlib.util.spec_from_file_location(spec_name, file_path)
                assert spec is not None and spec.loader is not None , f'{file_path} is not a valid module'
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                for _ , obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ == spec_name and issubclass(obj , StockFactorCalculator) and (obj is not StockFactorCalculator):
                        assert obj.__name__ not in self.pool , f'{obj.__name__} in module {spec_name} is duplicated'
                        obj.update_info(level = level_path.stem , file_name = file_path.stem , factor_name = obj.__name__)
                        
                        self.pool[obj.__name__] = obj
                        if level_path.stem not in self.hier: self.hier[level_path.stem] = []
                        self.hier[level_path.stem].append(obj)

        return self

    def factor_df(self , level : str | None = None , category0 : str | None = None , category1 : str | None = None , file_name : str | None = None):
        info_list = ['level' , 'file_name' , 'factor_name']
        attr_list = ['init_date' , 'category0' , 'category1' , 'description']
        
        df_dict = [[cls.other_info[i] for i in info_list] + [getattr(cls , a) for a in attr_list] for cls in self]
        df = pd.DataFrame(df_dict, columns=[*info_list , *attr_list])
        if level is not None: df = df[df['level'] == level]
        if category0 is not None: df = df[df['category0'] == category0]
        if category1 is not None: df = df[df['category1'] == category1]
        if file_name is not None: df = df[df['file_name'] == file_name]
        return df
    
    def jobs(self , as_df = True):
        return pd.DataFrame(_FACTOR_UPDATE_JOBS , columns=['date' , 'level' , 'factor']) if as_df else _FACTOR_UPDATE_JOBS

    def __repr__(self):
        str_level_factors = [','.join(f'{level}({len(factors)})' for level , factors in self.hier.items())]
        return f'StockFactorHierarchy({str_level_factors})'

    def factor_names(self):
        return [f'{cls.__name__}({str(cls.other_info)})' for cls in self]

    def __iter__(self):
        return (cls for level in self.iter_levels() for cls in self.iter_level_factors(level))

    def iter_levels(self):
        return iter(self.hier)
    
    def iter_level_factors(self , level : str):
        return (cls for cls in self.hier[level])

    def iter_instance(self):
        return (cls() for cls in self)
    
    def __getitem__(self , key : str):
        return self.hier[key]
    
    def get_factor(self , factor_name : str):
        return self.pool[factor_name]
    
    def test_calc_all_factors(self , date : int , check_duplicates = True):
        factor_values : dict[str , pd.Series] = {}
        for obj in self.iter_instance():
            print(f'{obj.factor_name} ' , end='')
            df = obj.calc_factor(date)
            factor_values[obj.factor_name] = df[obj.factor_name] if isinstance(df , pd.DataFrame) else df
            print('calculated')
        if check_duplicates:
            abnormal_diffs = {}
            for fn1 , fn2 in combinations(factor_values.keys() , 2):
                f1 = (factor_values[fn1] - factor_values[fn1].mean()) / factor_values[fn1].std()
                f2 = (factor_values[fn2] - factor_values[fn2].mean()) / factor_values[fn2].std()

                diff = (f1 - f2).fillna(0).abs().std()
                corr = f1.corr(f2)
                if diff <= 0.01 or abs(corr) >= 0.999: 
                    abnormal_diffs[f'{fn1}.{fn2}'] = {'diff_std':diff , 'corr' : corr}
            if len(abnormal_diffs) == 0: 
                print('no abnormal factor diffs')
            else:
                print(f'abnormal factor diffs: {abnormal_diffs}')
        return factor_values
    
    @classmethod
    def update_jobs(cls , start : int = -1 , end : int = 99991231 , overwrite = False):
        obj = cls()
        [factor_cls().update_jobs(start , end , overwrite , factor_cls.other_info['level']) for factor_cls in obj]
        return obj
    
    @staticmethod
    def Update(overwrite = False , show_progress = True , ignore_error = True):
        perform_update_jobs(overwrite , show_progress , ignore_error)
