import time
import numpy as np
import pandas as pd

from abc import abstractmethod , ABCMeta
from importlib import import_module
from pathlib import Path
from typing import Any , Literal , Type , Callable , TypeVar

from src.proj import PATH , Logger , CALENDAR , DB , Dates
from src.proj.util.error_handler import retry_call
from .abc import TS

T = TypeVar('T')

class TushareFetcherMeta(ABCMeta):
    """meta class of TushareFetcher , check if the subclass is valid and register all subclasses without abstract methods"""
    registry : dict[str , Type['TushareFetcher'] | Any] = {}
    def __new__(cls , name , bases , dct):
        new_cls = super().__new__(cls , name , bases , dct)
        abstract_methods = getattr(new_cls , '__abstractmethods__' , None)
        if not abstract_methods and not name.startswith('_'):
            assert name not in cls.registry or cls.registry[name].__module__ == new_cls.__module__ , \
                f'{name} in module {new_cls.__module__} is duplicated within {cls.registry[name].__module__}'
            
            db_type = getattr(new_cls , 'DB_TYPE' , '')
            db_src = getattr(new_cls , 'DB_SRC' , '')
            db_key = getattr(new_cls , 'DB_KEY' , '')
            update_freq = getattr(new_cls , 'UPDATE_FREQ' , '')
            assert db_type , f'{name} DB_TYPE must be set'
            assert db_src , f'{name} DB_SRC must be set'
            assert db_key , f'{name} DB_KEY must be set'
            assert update_freq , f'{name} UPDATE_FREQ must be set'
            if db_type in ['info' , 'time_series']:
                assert DB.DBPath.ByName(db_src) , (db_type , db_src , db_key)
            elif db_type in ['date' , 'fina' , 'rolling' , 'fundport']:
                assert DB.DBPath.ByDate(db_src) , (db_type , db_src , db_key)
            else:
                raise KeyError(db_type)
            cls.registry[name] = new_cls

        return new_cls

class TushareIterateFetcher:
    base_path : Path = PATH.temp.joinpath('tushare_fetcher_breakpoint')
    base_path.mkdir(parents=True, exist_ok=True)
    survival_time : int = 4 # in hours

    def __init__(self , fetcher_name : str , tushare_api : Callable[..., T] , limit : int = 2000 , * , 
                 max_fetch_times : int = -1 , breakpoint : bool = True , **kwargs):
        self.fetcher_name = fetcher_name
        self.tushare_api = tushare_api

        self.limit = limit
        self.max_fetch_times = max_fetch_times
        self.kwargs = kwargs

        self.breakpoint = breakpoint

        self.api_name = TS.get_func_name(self.tushare_api)
        kwargs_str = '_'.join([f'{k}={v}' for k, v in sorted(self.kwargs.items() , key = lambda x: x[0])])
        self.breakpoint_path = self.base_path.joinpath(self.fetcher_name , self.api_name , kwargs_str)

    def __repr__(self):
        return f'Breakpoint of {self.fetcher_name} / {self.api_name}(limit={self.limit})'

    @property
    def metadata_path(self):
        return self.breakpoint_path.joinpath('metadata.json')

    def write_metadata(self , metadata : dict , overwrite : bool = False):
        if not self.breakpoint:
            return
        PATH.dump_json(metadata , self.metadata_path , overwrite = overwrite)

    def append_metadata(self , metadata : dict):
        old_metadata = self.load_metadata()
        old_metadata.update(metadata)
        self.write_metadata(old_metadata , overwrite = True)

    def load_metadata(self):
        if not self.breakpoint:
            return {}
        return PATH.read_json(self.metadata_path)

    def init_path(self):
        if not self.breakpoint:
            return
        self.breakpoint_path.mkdir(parents=True, exist_ok=True)

    def remove_path(self):
        self.clear_breakpoint()
        if self.breakpoint_path.exists():
            self.breakpoint_path.rmdir()

    def clear_breakpoint(self):
        if self.breakpoint_path.exists():
            [file.unlink() for file in self.breakpoint_path.glob('*')]

    def check_expiration(self):
        metadata = self.load_metadata()
        if 'expiration_date' in metadata and metadata['expiration_date'] < CALENDAR.now().timestamp():
            self.clear_breakpoint()
            self.init_metadata()

    def init_metadata(self):
        create_time = CALENDAR.now().timestamp()
        metadata = {
            'create_time' : create_time,
            'expiration_date' : create_time + self.survival_time * 3600,
        }
        self.write_metadata(metadata , overwrite = False)

    def save_breakpoint(self , datas : dict[Any , pd.DataFrame] , next_offset : int):
        if not self.breakpoint:
            return
        for offset , df in datas.items():
            DB.save_df(df , self.breakpoint_path.joinpath(f'bkpt.{offset}.feather'))
        metadata = {
            'save_time' : CALENDAR.now().timestamp(),
            'next_offset' : next_offset,
            'breakpoints' : list(datas.keys()),
        }
        self.append_metadata(metadata)
        Logger.success(f'Saved {self} to {self.breakpoint_path}' , indent = 1)

    def load_breakpoint(self) -> tuple[int , dict[int , pd.DataFrame]]:
        """return the offset and data of the breakpoint"""
        self.init_path()
        self.check_expiration()
        metadata = self.load_metadata()
        if 'next_offset' in metadata and 'breakpoints' in metadata:
            try:
                dfs = {}
                for bkpt in metadata['breakpoints']:
                    p = self.breakpoint_path.joinpath(f'bkpt.{bkpt}.feather')
                    dfs[int(bkpt)] = DB.load_df(p)
                Logger.success(f'Loaded {self} from {self.breakpoint_path} , next_offset={metadata['next_offset']}' , indent = 1)
                return metadata['next_offset'] , dfs
            except Exception as e:
                Logger.error(f'Error loading breakpoint: {e}')
                return 0 , {}
        else:
            return 0 , {}

    def fetch(self) -> pd.DataFrame:
        """iterate fetch from tushare"""
        offset , dfs = self.load_breakpoint()
        while True:
            if self.max_fetch_times <= 0 or len(dfs) < self.max_fetch_times:
                ret = retry_call(TS.locked(self.tushare_api) , () , self.kwargs | {'offset' : offset , 'limit' : self.limit})
                if not isinstance(ret , pd.DataFrame):
                    raise TypeError(f'{self} must return a pd.DataFrame, but got {ret}')
            else:
                ret = Exception(f'{self} got more than {self.max_fetch_times} dfs')
            if isinstance(ret , pd.DataFrame):
                if ret.empty:
                    break
                ret = ret.dropna(axis=1, how='all')
                if not ret.empty: 
                    dfs[offset] = ret
            elif isinstance(ret , Exception):
                self.save_breakpoint(dfs , offset + self.limit)
                raise ret
            else:
                raise Exception(f'{self} must return a pd.DataFrame or Exception, but got {ret}')
            offset += self.limit
        if dfs:
            all_df = pd.concat([df for df in dfs.values() if not df.empty])
            all_df = all_df.reset_index([idx for idx in all_df.index.names if idx is not None] , drop = False).reset_index(drop = True)
            self.remove_path()
            return all_df
        else:
            return pd.DataFrame()

class TushareFetcher(metaclass=TushareFetcherMeta):
    """base class of TushareFetcher"""
    START_DATE  : int = 19970101
    DB_TYPE     : Literal['info' , 'time_series' , 'date' , 'fina' , 'rolling' , 'fundport' , ''] = ''
    UPDATE_FREQ : Literal['d' , 'w' , 'm' , ''] = ''
    DB_SRC      : str = ''
    DB_KEY      : str = ''

    _stdout_indent : int = 1
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}Fetcher(type={self.DB_TYPE},db={self.DB_SRC}/{self.DB_KEY},start={self.START_DATE},freq={self.UPDATE_FREQ})'
    
    def __str__(self) -> str:
        if self.__class__.__name__.lower().endswith('fetcher'):
            name = self.__class__.__name__
        else:
            name = self.__class__.__name__ + 'Fetcher'
        return f'{name}()'

    @classmethod
    def set_stdout_indent(cls , indent : int):
        cls._stdout_indent = indent

    @classmethod
    def load_tasks(cls):
        task_path = Path(__file__).parent.with_name('task')
        for path in sorted(task_path.rglob('*.py')):
            module_name = '.'.join(path.relative_to(PATH.main).with_suffix('').parts)
            import_module(module_name)

    @abstractmethod
    def get_data(self , date : int | Any = None , date2 : int | Any = None) -> pd.DataFrame: 
        """get required dataframe on given date""" 

    @abstractmethod
    def target_dates(self) -> list[int] | np.ndarray: 
        """get update dates"""

    def _info_fetcher_update_date(self) -> list[int] | np.ndarray:
        """update date for info fetcher"""
        assert self.UPDATE_FREQ , f'{self.__class__.__name__} UPDATE_FREQ must be set'
        update_to = CALENDAR.update_to()
        return [update_to] if self.updatable(self.last_date() , self.UPDATE_FREQ , update_to) else []

    def _date_fetcher_update_dates(self) -> list[int] | np.ndarray:
        """update dates for date fetcher"""
        assert self.UPDATE_FREQ , f'{self.__class__.__name__} UPDATE_FREQ must be set'
        return TS.dates_to_update(self.last_date() , self.UPDATE_FREQ) 
    
    def _fina_fetcher_update_dates(self , data_freq : Literal['y' , 'h' , 'q'] = 'q' , consider_future : bool = False) -> list[int] | np.ndarray:
        """update dates for fina fetcher"""
        assert self.UPDATE_FREQ , f'{self.__class__.__name__} UPDATE_FREQ must be set'
        update_to = CALENDAR.update_to()
        update = self.updatable(self.last_update_date() , self.UPDATE_FREQ , update_to)
        if not update: 
            return []

        dates = CALENDAR.qe_trailing(update_to , n_past = 3 , n_future = 4 if consider_future else 0 , another_date = self.last_date())
        if data_freq == 'y': 
            dates = [date for date in dates if date % 10000 == 1231]
        elif data_freq == 'h': 
            dates = [date for date in dates if date % 10000 in [630,1231]]

        return dates

    @property
    def api(self):
        """get tushare pro api"""
        return TS.api

    @property
    def db_by_name(self) -> bool:
        """whether to use date type for the database"""
        return self.DB_TYPE in ['info' , 'time_series']
    
    def target_path(self , date : int | Any = None) -> Path:
        """get target path in database for date"""
        if self.db_by_name:  
            date = None
        else: 
            assert date is not None , f'{self.__class__.__name__} use date type but date is None'
        return DB.path(self.DB_SRC , self.DB_KEY , date)

    def set_rollback_date(self , rollback_date : int | None = None):
        """set rollback date to the fetcher for update rollback"""
        CALENDAR.check_rollback_date(rollback_date)
        assert not hasattr(self , '_rollback_date') , f'{self.__class__.__name__} rollback_date has been set'
        self._rollback_date = rollback_date

    @property
    def rollback_date(self) -> int | None:
        """get rollback date of the fetcher"""
        return getattr(self , '_rollback_date' , None)

    def last_date(self) -> int:
        """last date that has data of the database"""
        if self.db_by_name:
            ldate = PATH.file_modified_date(self.target_path() , self.START_DATE)
        else:
            dates = DB.dates(self.DB_SRC , self.DB_KEY)
            ldate = max(dates) if len(dates) else self.START_DATE
        if self.rollback_date: 
            ldate = min(ldate , self.rollback_date)
        return ldate

    @staticmethod
    def updatable(last_date : int , freq : Literal['d' , 'w' , 'm'] , update_to : int | None = None):
        """check if the date is updatable given last date and frequency"""
        return TS.updatable(last_date , freq , update_to)
    
    def last_update_date(self) -> int:
        """last modified / updated date of the database"""
        if self.db_by_name:
            ldate = PATH.file_modified_date(self.target_path() , self.START_DATE)
        else:
            ldate = PATH.file_modified_date(self.target_path(self.last_date()) , self.START_DATE)
        if self.rollback_date: 
            ldate = min(ldate , self.rollback_date)
        return ldate

    @classmethod
    def update(cls) -> None:
        """update the fetcher"""
        try:
            fetcher = cls()
            fetcher.set_rollback_date(None)
            fetcher.update_with_retries()
            fetcher.update_missing()
        except Exception as e:
            Logger.error(f'{cls.__name__} update failed: {e}')
            Logger.print_exc(e)
    
    @classmethod
    def rollback(cls , rollback_date : int) -> None:
        """update the fetcher with rollback date"""
        try:
            fetcher = cls()
            fetcher.set_rollback_date(rollback_date)
            fetcher.update_with_retries()
            fetcher.update_missing()
        except Exception as e:
            Logger.error(f'{cls.__name__} update rollback failed: {e}')
            Logger.print_exc(e)

    def check_server_down(self) -> bool:
        """check if the tushare server is down"""
        if TS.server_down:
            Logger.only_once(f'{self.__class__.__name__} will not update because Tushare server is down' , object = self , mark = 'tushare_server_down' , printer = Logger.error)
            return True
        return False

    def update_dates(self , dates , **kwargs) -> np.ndarray:
        """update the fetcher given dates"""
        if self.check_server_down(): 
            return np.array([], dtype = int)
        if not self.db_by_name:
            assert None not in dates , f'{self.__class__.__name__} use date type but date is None'
        for date in dates: 
            DB.save(self.get_data(date) , self.DB_SRC , self.DB_KEY , date = date , indent = 1 , vb_level = 3)
        return dates

    def update_with_retries(self , timeout_wait_seconds = 20 , timeout_max_retries = 10) -> None:
        """update the fetcher with retries"""
        dates = self.target_dates()

        if len(dates) == 0: 
            Logger.skipping(f'{self.__class__.__name__} already fetched up to {CALENDAR.update_to()}!' , indent = self._stdout_indent)
            return

        updated_dates = np.array([], dtype = int)
        
        while timeout_max_retries >= 0:
            try:
                new_dates = self.update_dates(dates)
                updated_dates = np.concatenate([updated_dates , new_dates])
            except Exception as e:
                if '最多访问' in str(e):
                    if timeout_max_retries <= 0: 
                        Logger.warning(f'max retries reached: {e}')
                    else:
                        Logger.alert1(f'{e} , wait {timeout_wait_seconds} seconds' , indent = self._stdout_indent)
                        time.sleep(timeout_wait_seconds)
                elif 'Connection to api.waditu.com timed out' in str(e):
                    Logger.error(e)
                    TS.server_down = True
                    self.check_server_down()
                    raise Exception('Tushare server is down, skip today\'s update')
                else: 
                    raise e
            else:
                break
            timeout_max_retries -= 1
            dates = self.target_dates()
        Logger.success(f'{self.__class__.__name__} fetched for {Dates(updated_dates)}' , indent = self._stdout_indent)

    def locked_fetch(self , tushare_api : Callable[..., T] , *args, **kwargs) -> T:
        """locked fetch from tushare"""
        with TS.lock:
            return tushare_api(*args, **kwargs)

    def iterate_fetch(self , tushare_api : Callable[..., T] , limit = 2000 , max_fetch_times = -1 , breakpoint : bool = False , **kwargs) -> pd.DataFrame:
        """iterate fetch from tushare"""
        iterate_fetcher = TushareIterateFetcher(self.__class__.__name__ , tushare_api , limit , max_fetch_times = max_fetch_times , breakpoint = breakpoint , **kwargs)
        return iterate_fetcher.fetch()
        
    def missing_dates(self):
        """get missing dates"""
        return np.array([] , dtype = int)

    @classmethod
    def update_missing(cls):
        """update missing dates"""
        fetcher = cls()
        missing_dates = fetcher.missing_dates()
        if len(missing_dates) == 0:
            return
        fetcher.update_dates(missing_dates , step = 1)
        
class InfoFetcher(TushareFetcher):
    """base class of info fetcher , implement get_data for real use"""
    DB_TYPE = 'info'
    UPDATE_FREQ = 'd'
    DB_SRC = 'information_ts'

    def target_dates(self):
        return self._info_fetcher_update_date()

class TimeSeriesFetcher(TushareFetcher):
    """base class of time series fetcher , implement get_data for real use"""
    DB_TYPE = 'time_series'
    UPDATE_FREQ = 'd'

    def target_dates(self):
        return self._info_fetcher_update_date()

class TradeDataFetcher(TushareFetcher):
    """base class of date fetcher , implement get_data for real use"""
    DB_TYPE = 'date'
    DB_SRC = 'trade_ts'

    def target_dates(self): 
        return self._date_fetcher_update_dates()

class DayFetcher(TradeDataFetcher):
    """base class of day fetcher , implement get_data for real use"""
    UPDATE_FREQ = 'd'

    def missing_dates(self):
        """get missing dates"""
        dates = CALENDAR.range(self.START_DATE , CALENDAR.update_to() , type = 'td')
        stored_dates = DB.dates(self.DB_SRC , self.DB_KEY)
        missing_dates = np.setdiff1d(dates , stored_dates)
        return missing_dates

class WeekFetcher(TradeDataFetcher):
    """base class of week fetcher , implement get_data for real use"""
    UPDATE_FREQ = 'w'

class MonthFetcher(TradeDataFetcher):
    """base class of month fetcher , implement get_data for real use"""
    UPDATE_FREQ = 'm'

class FinaFetcher(TushareFetcher):
    """base class of fina fetcher , implement get_data for real use"""
    DB_TYPE = 'fina'
    UPDATE_FREQ = 'd'
    DB_SRC = 'financial_ts'
    DATA_FREQ : Literal['y' , 'h' , 'q'] = 'q'
    CONSIDER_FUTURE = False

    def target_dates(self):
        return self._fina_fetcher_update_dates(self.DATA_FREQ , self.CONSIDER_FUTURE)

class RollingFetcher(TushareFetcher):
    """base class of rolling fetcher , implement get_data for real use"""
    DB_TYPE = 'rolling'
    UPDATE_FREQ = 'd'

    ROLLING_BACK_DAYS = 30
    ROLLING_SEP_DAYS = 50
    ROLLING_DATE_COL = 'date'
    SAVEING_DATE_COL = True

    def __init__(self):
        super().__init__()
        assert self.ROLLING_BACK_DAYS > 0 , f'{self.__class__.__name__} ROLLING_BACK_DAYS must be positive'
        assert self.ROLLING_SEP_DAYS > 0 , f'{self.__class__.__name__} ROLLING_BACK_DAYS must be positive'

    def target_dates(self):
        """get update dates for rolling fetcher"""
        assert self.UPDATE_FREQ , f'{self.__class__.__name__} UPDATE_FREQ must be set'
        update_to = CALENDAR.update_to()
        update = self.updatable(self.last_update_date() , self.UPDATE_FREQ , update_to)
        if not update: 
            return []

        rolling_last_date = max(self.START_DATE , CALENDAR.cd(self.last_date() , -self.ROLLING_BACK_DAYS))
        update_dates = TS.dates_to_update(rolling_last_date , self.UPDATE_FREQ , update_to)
        d , dates = update_dates[0] , [update_dates[0]]
        while True:
            d = CALENDAR.cd(d , self.ROLLING_SEP_DAYS)
            dates.append(min(d , update_to))
            if d >= update_to: 
                break
        return dates

    def update_dates(self , dates) -> np.ndarray:
        """override TushareFetcher.update_with_dates because rolling fetcher needs get data by ROLLING_SEP_DAYS intervals"""
        assert self.DB_TYPE == 'rolling' , f'{self.__class__.__name__} is not a rolling fetcher'
        updated_dates = []
        for i in range(len(dates) - 1):
            start , end = CALENDAR.cd(dates[i] , 1) , dates[i+1]
            df = self.get_data(start , end)
            if df.empty: 
                continue
            assert self.ROLLING_DATE_COL in df.columns , f'{self.ROLLING_DATE_COL} not in {df.columns}'
            for date in df[self.ROLLING_DATE_COL].unique():
                subdf = df.query(f'{self.ROLLING_DATE_COL} == @date').copy()
                if not self.SAVEING_DATE_COL: 
                    subdf = subdf.drop(columns = [self.ROLLING_DATE_COL])
                DB.save(subdf , self.DB_SRC , self.DB_KEY , date = date , indent = 1 , vb_level = 3)
                updated_dates.append(date)
        return np.array(updated_dates , dtype = int)