import time
import numpy as np
import pandas as pd

from typing import Any , Literal
from abc import abstractmethod , ABC

from src.basic import PATH , CALENDAR , Timer , Logger
from .func import updatable , dates_to_update
from .connect import TS_PARAMS

def set_tushare_server_down(value : bool):
    TS_PARAMS['server_down'] = value

def is_tushare_server_down():
    if TS_PARAMS['server_down']:
        return True
    return False 

class TushareFetcher(ABC):
    START_DATE  : int = 19970101
    DB_TYPE     : Literal['info' , 'date' , 'fina' , 'rolling' , 'fundport'] = 'info'
    UPDATE_FREQ : Literal['d' , 'w' , 'm'] = 'd'
    DB_SRC      : str = ''
    DB_KEY      : str = ''

    def __init__(self) -> None:
        if self.DB_TYPE == 'info':
            assert self.DB_SRC in PATH.DB_BY_NAME , (self.DB_TYPE , self.DB_SRC , self.DB_KEY)
            self.use_date_type = False
        elif self.DB_TYPE in ['date' , 'fina' , 'rolling' , 'fundport']:
            assert self.DB_SRC in PATH.DB_BY_DATE , (self.DB_TYPE , self.DB_SRC , self.DB_KEY)
            self.use_date_type = True
        else:
            raise KeyError(self.DB_TYPE)
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}Fetcher(type={self.DB_TYPE},db={self.DB_SRC}/{self.DB_KEY},start={self.START_DATE},freq={self.UPDATE_FREQ})'
    
    def __str__(self) -> str:
        if self.__class__.__name__.lower().endswith('fetcher'):
            name = self.__class__.__name__
        else:
            name = self.__class__.__name__ + 'Fetcher'
        return f'{name}()'

    @abstractmethod
    def get_data(self , date : int | Any = None , date2 : int | Any = None) -> pd.DataFrame: 
        '''get required dataframe at date''' 
    @abstractmethod
    def get_update_dates(self) -> list[int] | np.ndarray: ...

    def _info_fetcher_update_date(self):
        update_to = CALENDAR.update_to()    
        return [update_to] if updatable(self.last_date() , self.UPDATE_FREQ , update_to) else []


    def _date_fetcher_update_dates(self):
        return dates_to_update(self.last_date() , self.UPDATE_FREQ) 
    
    def _fina_fetcher_update_dates(self , data_freq : Literal['y' , 'h' , 'q'] = 'q' , consider_future : bool = False):
        update_to = CALENDAR.update_to()
        update = updatable(self.last_update_date() , self.UPDATE_FREQ , update_to)
        if not update: return []

        dates = CALENDAR.qe_trailing(update_to , n_past = 3 , n_future = 4 if consider_future else 0 , another_date = self.last_date())
        if data_freq == 'y': dates = [date for date in dates if date % 10000 == 1231]
        elif data_freq == 'h': dates = [date for date in dates if date % 10000 in [630,1231]]

        return dates
    
    def target_path(self , date : int | Any = None):
        if self.use_date_type:  assert date is not None
        else: date = None
        return PATH.db_path(self.DB_SRC , self.DB_KEY , date)

    def fetch_and_save(self , date : int | Any = None):
        if self.use_date_type:  assert date is not None
        PATH.db_save(self.get_data(date) , self.DB_SRC , self.DB_KEY , date = date , verbose = True)

    def set_rollback_date(self , rollback_date : int | None = None):
        CALENDAR.check_rollback_date(rollback_date)
        assert not hasattr(self , '_rollback_date') , 'rollback_date has been set'
        self._rollback_date = rollback_date

    @property
    def rollback_date(self) -> int | None:
        return getattr(self , '_rollback_date' , None)

    def last_date(self):
        '''last date that has data of the database'''
        if self.use_date_type:
            dates = PATH.db_dates(self.DB_SRC , self.DB_KEY)
            ldate =  max(dates) if len(dates) else self.START_DATE
        else:
            ldate =  PATH.file_modified_date(self.target_path() , self.START_DATE)
        if self.rollback_date: ldate = min(ldate , self.rollback_date)
        return ldate
    
    def last_update_date(self):
        '''last modified / updated date of the database'''
        if self.use_date_type:
            ldate = PATH.file_modified_date(self.target_path(self.last_date()) , self.START_DATE)
        else:
            ldate = PATH.file_modified_date(self.target_path() , self.START_DATE)
        if self.rollback_date: ldate = min(ldate , self.rollback_date)
        return ldate

    def update(self):
        try:
            self.set_rollback_date(None)
            self.update_with_retries()
        except Exception as e:
            Logger.error(f'{self.__class__.__name__} update failed: {e}')
    
    def update_rollback(self , rollback_date : int):
        try:
            self.set_rollback_date(rollback_date)
            self.update_with_retries()
        except Exception as e:
            Logger.error(f'{self.__class__.__name__} update rollback failed: {e}')

    def check_server_down(self):
        if is_tushare_server_down():
            if not getattr(self , '_print_server_down_message' , False):
                Logger.warning(f'{self.__class__.__name__} will not update because Tushare server is down')
                setattr(self , '_print_server_down_message' , True)
            return True
        return False

    def update_dates(self , dates):
        if self.check_server_down(): return
        for date in dates: self.fetch_and_save(date)

    def update_with_retries(self , timeout_wait_seconds = 20 , timeout_max_retries = 10):
        dates = self.get_update_dates()

        if len(dates) == 0: 
            print(f'{str(self)} has no dates to update')
            return
        
        print(f'{str(self)} update dates {dates[0]} ~ {dates[-1]}')
        while timeout_max_retries >= 0:
            try:
                self.update_dates(dates)
            except Exception as e:
                if '最多访问' in str(e):
                    if timeout_max_retries <= 0: raise e
                    Logger.warning(f'{e} , wait {timeout_wait_seconds} seconds')
                    time.sleep(timeout_wait_seconds)
                elif 'Connection to api.waditu.com timed out' in str(e):
                    Logger.warning(e)
                    set_tushare_server_down(True)
                    self.check_server_down()
                    raise Exception('Tushare server is down, skip today\'s update')
                else: 
                    raise e
            else:
                break
            timeout_max_retries -= 1
            dates = self.get_update_dates()

    def iterate_fetch(self , fetch_func , limit = 2000 , max_fetch_times = 200 , **kwargs):
        dfs : list[pd.DataFrame] = []
        offset = 0
        while True:
            df : pd.DataFrame | Any = fetch_func(**kwargs , offset = offset , limit = limit)
            if not isinstance(df , pd.DataFrame): raise TypeError(f'{fetch_func.__name__} must return a pd.DataFrame')
            elif df.empty: break
            elif len(dfs) >= max_fetch_times: raise Exception(f'{self.__class__.__name__} got more than {max_fetch_times} dfs')
            df = df.dropna(axis=1, how='all')
            if not df.empty: dfs.append(df)
            offset += limit
        if dfs:
            all_df = pd.concat([df for df in dfs if not df.empty])
            all_df = all_df.reset_index([idx for idx in all_df.index.names if idx is not None] , drop = False).reset_index(drop = True)
            return all_df
        else:
            return pd.DataFrame()
        
class InfoFetcher(TushareFetcher):
    DB_TYPE = 'info'
    UPDATE_FREQ = 'd'
    DB_SRC = 'information_ts'

    def get_update_dates(self):
        return self._info_fetcher_update_date()

class DateFetcher(TushareFetcher):
    DB_TYPE = 'date'
    UPDATE_FREQ = 'd'
    DB_SRC = 'trade_ts'

    def get_update_dates(self): return self._date_fetcher_update_dates()

class WeekFetcher(DateFetcher):
    UPDATE_FREQ = 'w'

class MonthFetcher(DateFetcher):
    UPDATE_FREQ = 'm'

class FinaFetcher(TushareFetcher):
    DB_TYPE = 'fina'
    UPDATE_FREQ = 'w'
    DB_SRC = 'financial_ts'
    DATA_FREQ : Literal['y' , 'h' , 'q'] = 'q'
    CONSIDER_FUTURE = False

    def get_update_dates(self):
        return self._fina_fetcher_update_dates(self.DATA_FREQ , self.CONSIDER_FUTURE)

class RollingFetcher(TushareFetcher):
    DB_TYPE = 'rolling'
    UPDATE_FREQ = 'd'

    ROLLING_BACK_DAYS = 30
    ROLLING_SEP_DAYS = 50
    ROLLING_DATE_COL = 'date'
    SAVEING_DATE_COL = True

    def __init__(self):
        super().__init__()
        assert self.ROLLING_BACK_DAYS > 0 , 'ROLLING_BACK_DAYS must be positive'
        assert self.ROLLING_SEP_DAYS > 0 , 'ROLLING_BACK_DAYS must be positive'

    def get_update_dates(self):
        update_to = CALENDAR.update_to()
        update = updatable(self.last_update_date() , self.UPDATE_FREQ , update_to)
        if not update: return []

        rolling_last_date = max(self.START_DATE , CALENDAR.cd(self.last_date() , -self.ROLLING_BACK_DAYS))
        all_dates = dates_to_update(rolling_last_date , self.UPDATE_FREQ , update_to)
        d , dates = all_dates[0] , [all_dates[0]]
        while True:
            d = CALENDAR.cd(d , self.ROLLING_SEP_DAYS)
            dates.append(min(d , update_to))
            if d >= update_to: break
        return dates

    def update_dates(self , dates):
        '''override TushareFetcher.update_with_dates because rolling fetcher needs get data by ROLLING_SEP_DAYS intervals'''
        assert self.DB_TYPE == 'rolling' , f'{self.__class__.__name__} is not a rolling fetcher'
        for i in range(len(dates) - 1):
            start_dt , end_dt = CALENDAR.cd(dates[i] , 1) , dates[i+1]
            df = self.get_data(start_dt , end_dt)
            if df.empty: continue
            assert self.ROLLING_DATE_COL in df.columns , f'{self.ROLLING_DATE_COL} not in {df.columns}'
            for date in df[self.ROLLING_DATE_COL].unique():
                subdf = df[df[self.ROLLING_DATE_COL] == date].copy()
                if not self.SAVEING_DATE_COL: subdf = subdf.drop(columns = [self.ROLLING_DATE_COL])
                PATH.db_save(subdf , self.DB_SRC , self.DB_KEY , date = date , verbose = False)