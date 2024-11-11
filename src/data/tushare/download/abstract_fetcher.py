import time
import pandas as pd

from typing import Any , Literal
from abc import abstractmethod , ABC

from ..basic.func import file_update_date , updatable , dates_to_update , quarter_ends
from ....basic import PATH
from ....func import today , date_offset

class TushareFetcher(ABC):
    START_DATE = 19970101
    DB_TYPE : Literal['info' , 'date' , 'fina' , 'rolling'] = 'info'
    UPDATE_FREQ : Literal['d' , 'w' , 'm'] = 'd'

    def __init__(self) -> None:
        if self.DB_TYPE == 'info':
            assert self.db_src() in PATH.DB_BY_NAME , (self.DB_TYPE , self.db_src() , self.db_key())
        elif self.DB_TYPE in ['date' , 'fina' , 'rolling']:
            assert self.db_src() in PATH.DB_BY_DATE , (self.DB_TYPE , self.db_src() , self.db_key())
        else:
            raise KeyError(self.DB_TYPE)
    
    @abstractmethod
    def get_data(self , date : int | Any = None , date2 : int | Any = None) -> pd.DataFrame: 
        '''get required dataframe at date''' 
    @abstractmethod
    def db_src(self) -> str: ...
    @abstractmethod
    def db_key(self) -> str: ...
    @abstractmethod
    def update_dates(self) -> list[int]: ...

    @property
    def use_date_type(self): return self.DB_TYPE in ['date' , 'fina' , 'rolling']
    
    def target_path(self , date : int | Any = None , makedir = False):
        if self.use_date_type:  assert date is not None
        else: date = None
        return PATH.get_target_path(self.db_src() , self.db_key() , date = date , makedir=makedir)

    def fetch_and_save(self , date : int | Any = None):
        if self.use_date_type:  assert date is not None
        print(f'{self.__class__.__name__} Updating {self.db_src()}/{self.db_key()} at {date}')
        df = self.get_data(date)
        if len(df): PATH.save_df(df , self.target_path(date , True))

    def last_date(self):
        if self.use_date_type:
            dates = PATH.get_target_dates(self.db_src() , self.db_key())
            return max(dates) if len(dates) else self.START_DATE
        else:
            return file_update_date(self.target_path() , self.START_DATE)
            
    def last_update_date(self):
        if self.use_date_type:
            return file_update_date(self.target_path(self.last_date()) , self.START_DATE)
        else:
            return file_update_date(self.target_path() , self.START_DATE)

    def update(self , timeout_wait_seconds = 20 , timeout_max_retries = 20):
        update_func = self.update_with_try_except(self.update_with_dates , timeout_wait_seconds , timeout_max_retries)
        return update_func()

    def update_with_dates(self , dates):
        for date in dates: self.fetch_and_save(date)

    def update_with_try_except(self , func , timeout_wait_seconds = 20 , timeout_max_retries = 10):
        def wrapper(*args , **kwargs):
            retries = 0
            dates = self.update_dates()
            while retries < timeout_max_retries:
                try:
                    func(dates)
                except Exception as e:
                    if '最多访问' in str(e):
                        if retries > timeout_max_retries: raise e
                        print(f'{e} , wait {timeout_wait_seconds} seconds')
                        time.sleep(timeout_wait_seconds)
                    else: 
                        raise e
                else:
                    break
                retries += 1
                dates = self.update_dates()
                if len(dates) == 0: break    
        return wrapper

    def iterate_fetch(self , fetch_func , limit = 2000 , max_fetch_times = 200 , **kwargs):
        dfs = []
        offset = 0
        while True:
            df = fetch_func(**kwargs , offset = offset , limit = limit)
            if len(df) == 0: break
            if len(dfs) >= max_fetch_times: raise Exception(f'{self.__class__.__name__} got more than {max_fetch_times} dfs')
            dfs.append(df)
            offset += limit
        if len(dfs):
            return pd.concat(dfs).reset_index(drop = True)
        else:
            return pd.DataFrame()
        
class InfoFetcher(TushareFetcher):
    DB_TYPE = 'info'
    UPDATE_FREQ = 'd'

    def update_dates(self):
        this_date = today()
        last_update_date = self.last_date()
        if update := updatable(this_date , last_update_date , self.UPDATE_FREQ):
            return [this_date]
        else:
            return []

class DateFetcher(TushareFetcher):
    DB_TYPE = 'date'
    UPDATE_FREQ = 'd'

    def update_dates(self):
        this_date = today()
        last_update_date = self.last_date()
        dates = dates_to_update(this_date , last_update_date , self.UPDATE_FREQ) 
        return dates

class WeekFetcher(TushareFetcher):
    DB_TYPE = 'date'
    UPDATE_FREQ = 'w'

    def update_dates(self):
        this_date = today()
        last_update_date = self.last_date()
        dates = dates_to_update(this_date , last_update_date , self.UPDATE_FREQ) 
        return dates

class MonthFetcher(TushareFetcher):
    DB_TYPE = 'date'
    UPDATE_FREQ = 'm'

    def update_dates(self):
        this_date = today()
        last_update_date = self.last_date()
        dates = dates_to_update(this_date , last_update_date , self.UPDATE_FREQ) 
        return dates

class FinaFetcher(TushareFetcher):
    DB_TYPE = 'fina'
    UPDATE_FREQ = 'w'
    DATA_FREQ : Literal['y' , 'h' , 'q'] = 'q'
    CONSIDER_FUTURE = False

    def update_dates(self):
        this_date , last_date , last_update_date = today() , self.last_date() , self.last_update_date()

        update = updatable(this_date , last_update_date , self.UPDATE_FREQ)
        dates = quarter_ends(this_date , last_date , consider_future = self.CONSIDER_FUTURE) 
        if self.DATA_FREQ == 'y': dates = [date for date in dates if date % 10000 == 1231]
        elif self.DATA_FREQ == 'h': dates = [date for date in dates if date % 10000 in [630,1231]]
        
        if not update: dates = []
        return dates

class RollingFetcher(TushareFetcher):
    DB_TYPE = 'rolling'
    UPDATE_FREQ = 'd'

    ROLLING_BACK_DAYS = 30
    ROLLING_SEP_DAYS = 50
    ROLLING_DATE_COL = 'date'

    def update_dates(self):
        this_date , last_date , last_update_date = today() , self.last_date() , self.last_update_date()

        assert self.ROLLING_BACK_DAYS > 0 , 'ROLLING_BACK_DAYS must be positive'
        assert self.ROLLING_SEP_DAYS > 0 , 'ROLLING_BACK_DAYS must be positive'
        update = updatable(this_date , last_update_date , self.UPDATE_FREQ)
        if not update: return []

        all_dates = dates_to_update(this_date , max(self.START_DATE , int(date_offset(last_date , -self.ROLLING_BACK_DAYS))) , self.UPDATE_FREQ)
        d = all_dates[0]
        dates = [d]
        while True:
            d = date_offset(d , self.ROLLING_SEP_DAYS)
            dates.append(min(d , this_date))
            if d >= this_date: break
        return dates

    def update_with_dates(self , dates):
        for i in range(len(dates) - 1):
            start_date = date_offset(dates[i] , 1)
            end_date   = dates[i+1]
            assert start_date is not None and end_date is not None , 'start_date and end_date must be provided'
            assert self.DB_TYPE == 'rolling' , f'{self.__class__.__name__} is not a rolling fetcher'
            print(f'{self.__class__.__name__} Updating {self.db_src()}/{self.db_key()} from {start_date} to {end_date}')
            df = self.get_data(start_date , end_date)
            assert self.ROLLING_DATE_COL in df.columns , f'{self.ROLLING_DATE_COL} not in {df.columns}'
            for date in df[self.ROLLING_DATE_COL].unique():
                if len(df[df[self.ROLLING_DATE_COL] == date]): 
                    PATH.save_df(df[df[self.ROLLING_DATE_COL] == date] , self.target_path(date , True))