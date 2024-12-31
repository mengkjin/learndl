import time
import pandas as pd

from typing import Any , Literal
from abc import abstractmethod , ABC

from src.basic import PATH , CALENDAR , Timer
from .func import updatable , dates_to_update

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
    
    @abstractmethod
    def get_data(self , date : int | Any = None , date2 : int | Any = None) -> pd.DataFrame: 
        '''get required dataframe at date''' 
    @abstractmethod
    def update_dates(self) -> list[int]: ...
    
    def target_path(self , date : int | Any = None):
        if self.use_date_type:  assert date is not None
        else: date = None
        return PATH.db_path(self.DB_SRC , self.DB_KEY , date)

    def fetch_and_save(self , date : int | Any = None):
        if self.use_date_type:  assert date is not None
        PATH.db_save(self.get_data(date) , self.DB_SRC , self.DB_KEY , date = date , verbose = True)

    def last_date(self):
        if self.use_date_type:
            dates = PATH.db_dates(self.DB_SRC , self.DB_KEY)
            return max(dates) if len(dates) else self.START_DATE
        else:
            return PATH.file_modified_date(self.target_path() , self.START_DATE)
            
    def last_update_date(self):
        if self.use_date_type:
            return PATH.file_modified_date(self.target_path(self.last_date()) , self.START_DATE)
        else:
            return PATH.file_modified_date(self.target_path() , self.START_DATE)

    def update(self , timeout_wait_seconds = 20 , timeout_max_retries = 20):
        update_func = self.update_with_try_except(self.update_with_dates , timeout_wait_seconds , timeout_max_retries)
        return update_func()

    def update_with_dates(self , dates):
        for date in dates: self.fetch_and_save(date)

    def update_with_try_except(self , func , timeout_wait_seconds = 20 , timeout_max_retries = 10):
        def wrapper(*args , **kwargs):
            retries = 0
            dates = self.update_dates()
            if len(dates) == 0: 
                print(f'{self.__class__.__name__} has no dates to update')
                return
            else:
                print(f'{self.__class__.__name__} update dates {dates[0]} ~ {dates[-1]}')
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

    def update_dates(self):
        this_date = CALENDAR.today()
        last_update_date = self.last_date()
        if update := updatable(this_date , last_update_date , self.UPDATE_FREQ):
            return [this_date]
        else:
            return []

class DateFetcher(TushareFetcher):
    DB_TYPE = 'date'
    UPDATE_FREQ = 'd'
    DB_SRC = 'trade_ts'

    def update_dates(self):
        this_date = CALENDAR.today()
        last_update_date = self.last_date()
        dates = dates_to_update(this_date , last_update_date , self.UPDATE_FREQ) 
        return dates

class WeekFetcher(TushareFetcher):
    DB_TYPE = 'date'
    UPDATE_FREQ = 'w'
    DB_SRC = 'trade_ts'

    def update_dates(self):
        this_date = CALENDAR.today()
        last_update_date = self.last_date()
        dates = dates_to_update(this_date , last_update_date , self.UPDATE_FREQ) 
        return dates

class MonthFetcher(TushareFetcher):
    DB_TYPE = 'date'
    UPDATE_FREQ = 'm'
    DB_SRC = 'trade_ts'

    def update_dates(self):
        this_date = CALENDAR.today()
        last_update_date = self.last_date()
        dates = dates_to_update(this_date , last_update_date , self.UPDATE_FREQ) 
        return dates

class FinaFetcher(TushareFetcher):
    DB_TYPE = 'fina'
    UPDATE_FREQ = 'w'
    DB_SRC = 'financial_ts'
    DATA_FREQ : Literal['y' , 'h' , 'q'] = 'q'
    CONSIDER_FUTURE = False

    def update_dates(self):
        this_date , last_date , last_update_date = CALENDAR.today() , self.last_date() , self.last_update_date()

        update = updatable(this_date , last_update_date , self.UPDATE_FREQ)
        dates = CALENDAR.qe_trailing(this_date , n_past = 3 , n_future = 4 if self.CONSIDER_FUTURE else 0 , another_date = last_date)

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
    SAVEING_DATE_COL = True

    def update_dates(self):
        this_date , last_date , last_update_date = CALENDAR.today() , self.last_date() , self.last_update_date()

        assert self.ROLLING_BACK_DAYS > 0 , 'ROLLING_BACK_DAYS must be positive'
        assert self.ROLLING_SEP_DAYS > 0 , 'ROLLING_BACK_DAYS must be positive'
        update = updatable(this_date , last_update_date , self.UPDATE_FREQ)
        if not update: return []
        all_dates = dates_to_update(this_date , max(self.START_DATE , CALENDAR.cd(last_date , -self.ROLLING_BACK_DAYS)) , self.UPDATE_FREQ)
        d = all_dates[0]
        dates = [d]
        while True:
            d = CALENDAR.cd(d , self.ROLLING_SEP_DAYS)
            dates.append(min(d , this_date))
            if d >= this_date: break
        return dates

    def update_with_dates(self , dates):
        for i in range(len(dates) - 1):
            start_date = CALENDAR.cd(dates[i] , 1)
            end_date   = dates[i+1]
            assert start_date is not None and end_date is not None , 'start_date and end_date must be provided'
            assert self.DB_TYPE == 'rolling' , f'{self.__class__.__name__} is not a rolling fetcher'
            df = self.get_data(start_date , end_date)
            if df.empty: continue
            assert self.ROLLING_DATE_COL in df.columns , f'{self.ROLLING_DATE_COL} not in {df.columns}'
            with Timer(f'DataBase object [{self.DB_SRC}],[{self.DB_KEY}],[{start_date} to {end_date}]'):
                for date in df[self.ROLLING_DATE_COL].unique():
                    subdf = df[df[self.ROLLING_DATE_COL] == date].copy()
                    if not self.SAVEING_DATE_COL: subdf = subdf.drop(columns = [self.ROLLING_DATE_COL])
                    PATH.db_save(subdf , self.DB_SRC , self.DB_KEY , date = date , verbose = False)