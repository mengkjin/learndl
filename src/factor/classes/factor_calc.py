

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ...basic import PATH
from ...data.tushare.basic import CALENDAR

_FACTOR_UPDATE_JOBS : dict[int , dict[str , 'StockFactorCalculator']] = {}

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

class StockFactorCalculator(ABC):
    init_date : int = -1
    category0 : str = '' # 'fundamental' , 'analyst' , 'high_frequency' , 'behavior' , 'money_flow' , 'alternative'
    category1 : str = ''
    description : str = ''

    def __new__(cls , *args , **kwargs):
        cls.validate_attr()
        return super().__new__(cls)

    def __init__(self):
        self.factors : dict[int , pd.DataFrame] = {}

    def __repr__(self):
        return f'StockFactor(name={self.factor_name},category0={self.category0},category1={self.category1},from={self.init_date})'

    @abstractmethod
    def calc_factor(self , date : int) -> pd.DataFrame:
        '''calculate factor value , must have secid and factor_value / factor_name columns'''
        return pd.DataFrame()

    @property
    def factor_name(self): return self.__class__.__name__
    
    def factor_folder(self , factor_name : str | None = None):
        return factor_folder(self.factor_name if factor_name is None else factor_name)
    
    def factor_values(self):
        return pd.concat([df.assign(date = d) for d , df in self.factors.items()]).reset_index().set_index(['date','secid'])

    def factor_path(self , date : int | Any , mkdir = True , factor_name : str | None = None):
        path = factor_path(self.factor_name if factor_name is None else factor_name , date)
        if mkdir and factor_name is None: path.parent.mkdir(parents=True , exist_ok=True)
        return path

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

    def calculate(self , date : int | Iterable| Any , overwrite = False):
        '''calculate factor value of a given date and store to factor_data'''
        if isinstance(date , Iterable):
            for d in date: self.calculate(d , overwrite)
        else:
            date = int(date)
            assert date >= self.init_date , f'date is should be greater than or equal to {self.init_date}, but got {date}'
            if overwrite or date not in self.factors: 
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
                if show_progress: print(f'Factor:{self.factor_name} at date {date} already there')
                continue
            try:
                self.validate_value(date , df , strict = strict)
                df.to_feather(self.factor_path(date , True))
                if show_progress: print(f'Factor:{self.factor_name} at date {date} deploy successful')
            except ValueError as e:
                print(f'Factor:{self.factor_name} at date {date} is invalid: {e}')

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

    def stored_dates(self):
        paths = PATH.list_files(self.factor_folder() , recur=True)
        dates = np.array(sorted(PATH.R_path_date(paths)) , dtype=int)
        return dates
    
    def update_jobs(self , start : int = -1 , end : int = 99991231):
        store_dates = self.stored_dates()
        dates = np.setdiff1d(CALENDAR.td_within(max(start , self.init_date) , end) , store_dates)
        for d in dates:
            if d not in _FACTOR_UPDATE_JOBS: _FACTOR_UPDATE_JOBS[d] = {}
            _FACTOR_UPDATE_JOBS[d][self.factor_name] = self
        return self

    def Update(self , start : int = -1 , end : int = 99991231 , show_progress = True , ignore_error = False):
        '''update factor data from self.init_date to today'''
        self.update_jobs(start , end)

        for date in _FACTOR_UPDATE_JOBS: 
            obj = _FACTOR_UPDATE_JOBS[date].pop(self.factor_name , None)
            if obj is None: continue
            assert obj is self , f'obj is should be {self} , but got {obj}'

            try:
                obj.calculate(date).deploy()
                if show_progress: print(f'Factor:{self.factor_name} update at date {date} finish')
            except Exception as e:
                if ignore_error:
                    print(f'Factor:{self.factor_name} update at date {date} failed: {e}')
                else:
                    raise e
                
        return self