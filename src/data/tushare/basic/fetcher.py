import pandas as pd

from typing import Any , Literal
from abc import abstractmethod , ABC

from .func import file_update_date , updatable , dates_to_update , quarter_ends
from ....basic import DB_BY_DATE , DB_BY_NAME , get_target_path , get_target_dates , save_df
from ....func import today

class TushareFetecher(ABC):
    def __init__(self , db_type : Literal['info' , 'date' , 'fina'] , update_freq : Literal['d' , 'w' , 'm']) -> None:
        self.db_type : Literal['info' , 'date' , 'fina'] = db_type
        self.update_freq : Literal['d' , 'w' , 'm']= update_freq
        if db_type == 'info':
            assert self.db_src() in DB_BY_NAME , (db_type , self.db_src() , self.db_key())
        elif db_type in ['date' , 'fina']:
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

    @property
    def use_date_type(self): return self.db_type in ['date' , 'fina']
    
    def target_path(self , date : int | Any = None , makedir = False):
        if self.use_date_type:  assert date is not None
        else: date = None
        return get_target_path(self.db_src() , self.db_key() , date = date , makedir=makedir)

    def fetch_and_save(self , date : int | Any = None):
        if self.use_date_type:  assert date is not None
        print(f'{self.__class__.__name__} Updating {self.db_src()}/{self.db_key()} at {date}')
        df = self.get_data(date)
        if len(df): save_df(df , self.target_path(date , True))

    def last_date(self , default = 19970101):
        if self.use_date_type:
            dates = get_target_dates(self.db_src() , self.db_key())
            return max(dates) if len(dates) else default
        else:
            return file_update_date(self.target_path() , default)
            
    def last_update_date(self , default = 19970101):
        if self.use_date_type:
            return file_update_date(self.target_path(self.last_date()) , default)
        else:
            return file_update_date(self.target_path() , default)
            
    def update(self):
        this_date = today()
        last_date = self.last_date()
        last_update = self.last_update_date()

        # if self.db_type == 'date' or updatable(this_date , last_date , last_update_date , self.update_freq):
        if self.db_type == 'info': 
            dates = [this_date] if updatable(this_date , last_date , self.update_freq) else []
        elif self.db_type == 'date': 
            dates = dates_to_update(this_date , last_date , self.update_freq) 
        elif self.db_type == 'fina': 
            dates = quarter_ends(this_date , last_date) if updatable(this_date , last_update , self.update_freq) else []
        else: raise KeyError(self.db_type)    
        
        for date in dates: self.fetch_and_save(date)

        if len(dates) == 0: print(f'{self.__class__.__name__} Already Updated at {last_date}')

class InfoFetecher(TushareFetecher):
    def __init__(self , update_freq : Literal['d' , 'w' , 'm'] = 'd') -> None:
        super().__init__('info' , update_freq)

class DateFetecher(TushareFetecher):
    def __init__(self) -> None:
        super().__init__('date' , 'd')

class WeekFetecher(TushareFetecher):
    def __init__(self) -> None:
        super().__init__('date' , 'w')

class MonthFetecher(TushareFetecher):
    def __init__(self) -> None:
        super().__init__('date' , 'm')

class FinaFetecher(TushareFetecher):
    def __init__(self) -> None:
        super().__init__('fina' , 'w')