import pandas as pd

from typing import Any , Literal
from abc import abstractmethod , ABC

from .func import file_update_date , updatable , dates_to_update , quarter_ends
from ....basic import PATH
from ....func import today

class TushareFetecher(ABC):
    START_DATE = 19970101
    DB_TYPE : Literal['info' , 'date' , 'fina'] = 'info'
    UPDATE_FREQ : Literal['d' , 'w' , 'm'] = 'd'

    def __init__(self) -> None:
        if self.DB_TYPE == 'info':
            assert self.db_src() in PATH.DB_BY_NAME , (self.DB_TYPE , self.db_src() , self.db_key())
        elif self.DB_TYPE in ['date' , 'fina']:
            assert self.db_src() in PATH.DB_BY_DATE , (self.DB_TYPE , self.db_src() , self.db_key())
        else:
            raise KeyError(self.DB_TYPE)
    
    @abstractmethod
    def get_data(self , date : int | Any = None) -> pd.DataFrame: 
        '''get required dataframe at date''' 
    @abstractmethod
    def db_src(self) -> str: ...
    @abstractmethod
    def db_key(self) -> str: ...

    @property
    def use_date_type(self): return self.DB_TYPE in ['date' , 'fina']
    
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
            
    def update_dates(self):
        this_date , last_date , last_update = today() , self.last_date() , self.last_update_date()

        # if self.db_type == 'date' or updatable(this_date , last_date , last_update_date , self.update_freq):
        if self.DB_TYPE == 'info': 
            dates = [this_date] if updatable(this_date , last_date , self.UPDATE_FREQ) else []
        elif self.DB_TYPE == 'date': 
            dates = dates_to_update(this_date , last_date , self.UPDATE_FREQ) 
        elif self.DB_TYPE == 'fina': 
            dates = quarter_ends(this_date , last_date) if updatable(this_date , last_update , self.UPDATE_FREQ) else []
        else: raise KeyError(self.DB_TYPE)    

        return dates

    def update(self):
        dates = self.update_dates()
        if len(dates) == 0: 
            print(f'{self.__class__.__name__} Already Updated at {self.last_date()}')
        else:
            for date in dates: self.fetch_and_save(date)
        
class InfoFetecher(TushareFetecher):
    DB_TYPE = 'info'
    UPDATE_FREQ = 'd'

class DateFetecher(TushareFetecher):
    DB_TYPE = 'date'
    UPDATE_FREQ = 'd'

class WeekFetecher(TushareFetecher):
    DB_TYPE = 'date'
    UPDATE_FREQ = 'w'

class MonthFetecher(TushareFetecher):
    DB_TYPE = 'date'
    UPDATE_FREQ = 'm'

class FinaFetecher(TushareFetecher):
    DB_TYPE = 'fina'
    UPDATE_FREQ = 'w'
