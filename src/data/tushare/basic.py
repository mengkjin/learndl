# 导入tushare
import os , time
import tushare as ts
import numpy as np
import pandas as pd

from typing import Any , Literal
from abc import abstractmethod , ABC

from ..basic import get_target_path , get_source_dates , get_target_dates , save_df
from ...env import DB_BY_DATE , DB_BY_NAME
from ...func import date_diff , today

pro = ts.pro_api('2026c96ef5fa7fc3241c96baafd638c585284c7fefaa00b93ef0a62c')

def code_to_secid(df : pd.DataFrame , code_col = 'ts_code' , retain = False):
    '''switch old symbol into secid'''
    if code_col not in df.columns.values: return df
    replace_dict = {'T00018' : '600018'}
    df['secid'] = df[code_col].astype(str).str.slice(0, 6).replace(replace_dict)
    df['secid'] = df['secid'].where(df['secid'].str.isdigit() , '-1').astype(int)
    if not retain: del df[code_col]
    return df

def updatable(date , last_date , freq : Literal['d' , 'w' , 'm']):
    if freq == 'd':
        return date > last_date
    elif freq == 'w':
        return date_diff(date , last_date) > 6
    elif freq == 'm':
        return ((date // 100) % 100) != ((last_date // 100) % 100)
    
def dates_to_update(date , last_date , freq : Literal['d' , 'w' , 'm']):
    if last_date >= date: return np.array([])
    if freq == 'd':
        date_list = pd.date_range(str(last_date) , str(date)).strftime('%Y%m%d').astype(int).to_numpy()[1:]
    elif freq == 'w':
        date_list = pd.date_range(str(last_date) , str(date)).strftime('%Y%m%d').astype(int).to_numpy()[::7][1:]
    elif freq == 'm':
        date_list = pd.date_range(str(last_date) , str(date) , freq='ME').strftime('%Y%m%d').astype(int).to_numpy()
        if last_date in date_list: date_list = date_list[1:]
    return date_list

class TushareFetecher(ABC):
    def __init__(self , db_type : Literal['info' , 'flow'] , update_freq : Literal['d' , 'w' , 'm']) -> None:
        self.db_type : Literal['info' , 'flow'] = db_type
        self.update_freq : Literal['d' , 'w' , 'm']= update_freq
        if db_type == 'info':
            assert self.db_src() in DB_BY_NAME , (db_type , self.db_src() , self.db_key())
        elif db_type == 'flow':
            assert self.db_src() in DB_BY_DATE , (db_type , self.db_src() , self.db_key())
        else:
            raise KeyError(db_type)
    
    @abstractmethod
    def get_data(self , date : int | Any = None) -> pd.DataFrame: 
        '''get required dataframe at date''' 
    @abstractmethod
    def db_src(self) -> str: ...
    @abstractmethod
    def db_key(self) -> str: ...
    
    def target_path(self , date : int | Any = None , makedir = False):
        if self.db_type == 'flow': assert date is not None
        date = date if self.db_type == 'flow' else None
        return get_target_path(self.db_src() , self.db_key() , date = date , makedir=makedir)

    def fetch_and_save(self , date : int | Any = None):
        if self.db_type == 'flow': assert date is not None
        df = self.get_data(date)
        if len(df): save_df(df , self.target_path(date , True))

    def last_date(self):
        if self.db_type == 'flow':
            dates = get_target_dates(self.db_src() , self.db_key())
            return max(dates) if len(dates) else 19900101
        else:
            path = self.target_path()
            if os.path.exists(path):
                return int(time.strftime('%Y%m%d',time.localtime(os.path.getmtime(path))))
            else:
                return 19900101
            
    def update(self):
        if self.db_type == 'info': 
            date = today()
            if updatable(date , self.last_date() , self.update_freq):
                print(f'{self.__class__.__name__} Updating {self.db_src()}/{self.db_key()} at {date}')
                self.fetch_and_save(date)
            else:
                print(f'{self.__class__.__name__} Already Updated at {self.last_date()}')
        else:
            dates = dates_to_update(today() , self.last_date() , self.update_freq) 
            for d in dates:
                print(f'{self.__class__.__name__} Updating {self.db_src()}/{self.db_key()} at {d}')
                self.fetch_and_save(d)
            if len(dates) == 0:
                print(f'{self.__class__.__name__} Already Updated at {self.last_date()}')

class InfoFetecher(TushareFetecher):
    def __init__(self , update_freq : Literal['d' , 'w' , 'm'] = 'd') -> None:
        super().__init__('info' , update_freq)

class DateFetecher(TushareFetecher):
    def __init__(self) -> None:
        super().__init__('flow' , 'd')

class WeekFetecher(TushareFetecher):
    def __init__(self) -> None:
        super().__init__('flow' , 'w')

class MonthFetecher(TushareFetecher):
    def __init__(self) -> None:
        super().__init__('flow' , 'm')