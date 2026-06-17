"""
Tushare data fetcher base classes and paginated iterator.

``TushareFetcher`` is the abstract base for all concrete data fetcher classes.
Subclasses set ``DB_TYPE``, ``DB_SRC``, ``DB_KEY``, and ``UPDATE_FREQ`` and
implement ``get_data(date)``; they are auto-registered by ``TushareFetcherMeta``.

``TushareIterateFetcher`` handles paginated API calls with breakpoint/resume
support: partial results are saved as ``.feather`` files so that interrupted
fetches can be resumed rather than restarted.

Concrete fetcher base classes
------------------------------
``InfoFetcher``        — static info tables (no date dimension)
``TimeSeriesFetcher``  — date-series tables (one row per date)
``TradeDataFetcher``   — daily market data
``DayFetcher``         — per-date incremental updates
``WeekFetcher``        — weekly updates
``MonthFetcher``       — monthly updates
``FinaFetcher``        — financial statement tables
``RollingFetcher``     — range-fetch with per-date splitting
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from abc import abstractmethod , ABCMeta
from importlib import import_module
from pathlib import Path
from typing import Any , Literal , TypeVar , TypeAlias
from collections.abc import Callable

from src.proj import MACHINE , PATH , CALENDAR , Dates , DB , Base , Save , Load 
from src.proj.util.functional.handler import retry_call
from .core import TS

__all__ = [
    'TushareFetcher' , 'InfoFetcher' , 'TimeSeriesFetcher' , 
    'TradeDataFetcher' , 'DayFetcher' , 'WeekFetcher' , 'MonthFetcher' , 
    'FinaFetcher' , 'RollingFetcher']

T = TypeVar('T')
TushareDbTarget : TypeAlias = Literal['info' , 'time_series' , 'date' , 'fina' , 'rolling' , 'fundport' , '']

class TushareFetcherMeta(ABCMeta):
    """meta class of TushareFetcher , check if the subclass is valid and register all subclasses without abstract methods"""
    registry : dict[str , type[TushareFetcher] | Any] = {}
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

class TushareIterateFetcher(Base.BoundLogger):
    """
    Paginated Tushare API fetcher with breakpoint/resume support.

    Fetches pages of ``limit`` rows at a time, accumulating results into
    ``.feather`` files.  On the next invocation the breakpoint metadata is
    checked: if the saved data is not older than ``survival_time`` hours,
    the offset picks up where it left off rather than re-fetching from page 1.

    Typical use: pass as a helper inside a ``TushareFetcher.get_data(date)``
    implementation to handle large paginated tables (e.g. analyst reports).
    """
    base_path : Path = PATH.cache.joinpath('tushare_fetcher_breakpoint')
    base_path.mkdir(parents=True, exist_ok=True)
    survival_time : int = 4 # in hours

    def __init__(
        self , fetcher_name : str , tushare_api : Callable[..., T] , limit : int = 2000 , * ,
        max_fetch_times : int = -1 , breakpoint : bool = True , 
        indent : int = 0 , vb_level : Base.lit.VerbosityLevel = 1 , **kwargs
    ):
        """
        Parameters
        ----------
        fetcher_name : str
            Used to namespace the breakpoint directory.
        tushare_api : Callable
            The Tushare API function to call (e.g. ``TS.api.stk_holdertrade``).
        limit : int
            Page size for each paginated API call.
        max_fetch_times : int
            Cap on the total number of pages (``-1`` = unlimited).
        breakpoint : bool
            Enable breakpoint/resume (default True).
        """
        super().__init__(indent=indent, vb_level=vb_level, **kwargs)
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
        if not self.breakpoint or not self.metadata_path.exists():
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
        if not metadata or ('expiration_date' in metadata and metadata['expiration_date'] < CALENDAR.now().timestamp()):
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
            Save.df(df , self.breakpoint_path.joinpath(f'bkpt.{offset}.feather'))
        metadata = {
            'save_time' : CALENDAR.now().timestamp(),
            'next_offset' : next_offset,
            'breakpoints' : list(datas.keys()),
        }
        self.append_metadata(metadata)
        self.logger.success(f'Saved {self} to {self.breakpoint_path}')

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
                    dfs[int(bkpt)] = Load.df(p)
                self.logger.success(f'Loaded {self} from {self.breakpoint_path} , next_offset={metadata['next_offset']}')
                return metadata['next_offset'] , dfs
            except Exception as e:
                self.logger.error(f'Error loading breakpoint: {e}')
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

class TushareFetcher(Base.BoundLogger , metaclass=TushareFetcherMeta):
    """base class of TushareFetcher"""
    START_DATE  : int = 19970101
    DB_TYPE     : TushareDbTarget = ''
    UPDATE_FREQ : Base.lit.FreqUpdate | Any = None
    DB_SRC      : str = ''
    DB_KEY      : str = ''
    SKIP_ON_MACHINES : tuple[str , ...] = ()
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}Fetcher(type={self.DB_TYPE},db={self.DB_SRC}/{self.DB_KEY},start={self.START_DATE},freq={self.UPDATE_FREQ})'
    
    def __str__(self) -> str:
        if self.__class__.__name__.lower().endswith('fetcher'):
            name = self.__class__.__name__
        else:
            name = self.__class__.__name__ + 'Fetcher'
        return f'{name}()'

    @classmethod
    def load_tasks(cls):
        task_path = Path(__file__).parent.with_name('task')
        for path in sorted(task_path.rglob('*.py')):
            module_name = '.'.join(PATH.relative(path).with_suffix('').parts)
            import_module(module_name)

    @abstractmethod
    def get_data(self , date : int | Any = None , date2 : int | Any = None) -> pd.DataFrame: 
        """get required dataframe on given date""" 

    @abstractmethod
    def target_dates(self) -> Dates: 
        """get update dates"""

    def _info_fetcher_update_date(self) -> Dates:
        """update date for info fetcher"""
        assert self.UPDATE_FREQ , f'{self.__class__.__name__} UPDATE_FREQ must be set'
        update_to = CALENDAR.update_to(key = 'tushare')
        return Dates(update_to) if self.updatable(self.last_date() , self.UPDATE_FREQ , update_to) else Dates()

    def _date_fetcher_update_dates(self) -> Dates:
        """update dates for date fetcher"""
        assert self.UPDATE_FREQ , f'{self.__class__.__name__} UPDATE_FREQ must be set'
        return TS.dates_to_update(self.last_date() , self.UPDATE_FREQ) 
    
    def _fina_fetcher_update_dates(self , data_freq : Base.lit.FreqFinData = 'q' , consider_future : bool = False) -> Dates:
        """update dates for fina fetcher"""
        assert self.UPDATE_FREQ , f'{self.__class__.__name__} UPDATE_FREQ must be set'
        update_to = CALENDAR.update_to(key = 'tushare')
        update = self.updatable(self.last_update_date() , self.UPDATE_FREQ , update_to)
        if not update: 
            return Dates()

        dates = CALENDAR.qe_trailing(update_to , n_past = 3 , n_future = 4 if consider_future else 0 , another_date = self.last_date())
        if data_freq == 'y': 
            dates = [date for date in dates if date % 10000 == 1231]
        elif data_freq == 'h': 
            dates = [date for date in dates if date % 10000 in [630,1231]]

        return Dates(dates)

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

    def set_rollback_date(self , rollback_date : int | None = None) -> None:
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
            dates = self.stored_dates()
            ldate = max(dates) if len(dates) else self.START_DATE
        if self.rollback_date: 
            ldate = min(ldate , self.rollback_date)
        return ldate

    @staticmethod
    def updatable(last_date : int , freq : Base.lit.FreqUpdate , update_to : int | None = None) -> bool:
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
    def update(
        cls , * , rollback_date : int | None = None , 
        indent : int = 0 , vb_level : Base.lit.VerbosityLevel = 1 , **kwargs
    ) -> Base.UpdateFlag:
        """update the fetcher"""
        try:
            if MACHINE.name in cls.SKIP_ON_MACHINES:
                cls.logger.skipping(f'{cls.__name__} is skipped on {MACHINE.name}' , idt = 1 , add_prefix = True)
                return Base.UpdateFlag.SKIPPED
            fetcher = cls(indent = indent , vb_level = vb_level)
            fetcher.set_rollback_date(rollback_date)

            flags = Base.UpdateFlagList()
            flags += fetcher.update_with_retries()
            flags += fetcher.update_missing()
            return flags.summarize()
        except Exception as e:
            cls.logger.error(f'Update failed: {e}' if rollback_date is None else f'Update rollback failed: {e}')
            cls.logger.print_exc(e)
            return Base.UpdateFlag.FAILED

    def check_server_down(self) -> bool:
        """check if the tushare server is down"""
        if TS.server_down:
            self.logger.only_once(
                f'Will not update because Tushare server is down' , 
                object = self , mark = 'tushare_server_down' , printer = 'error')
            return True
        return False

    def update_dates(self , dates : Base.intDates , **kwargs) -> Dates:
        """update the fetcher given dates"""
        dates = Dates(dates)
        if self.check_server_down(): 
            return dates
        for date in dates: 
            DB.save(self.get_data(date) , self.DB_SRC , self.DB_KEY , date = date , indent = self.indent + 1 , vb_level = self.vb_level + 1)
        return dates

    def update_with_retries(self , timeout_wait_seconds = 20 , timeout_max_retries = 10) -> Base.UpdateFlag:
        """update the fetcher with retries"""
        dates = self.target_dates()

        if dates.empty: 
            self.logger.skipping(f'Already fetched up to {CALENDAR.update_to(key = 'tushare')}!')
            return Base.UpdateFlag.SKIPPED

        updated_dates = np.array([], dtype = int)
        
        while timeout_max_retries >= 0:
            try:
                new_dates = self.update_dates(dates)
                updated_dates = np.concatenate([updated_dates , new_dates])
            except Exception as e:
                if '最多访问' in str(e):
                    if timeout_max_retries <= 0: 
                        self.logger.warning(f'max retries reached: {e}')
                    else:
                        self.logger.alert1(f'{e} , wait {timeout_wait_seconds} seconds' , idt = 1)
                        time.sleep(timeout_wait_seconds)
                elif 'Connection to api.waditu.com timed out' in str(e):
                    self.logger.error(e)
                    TS.server_down = True
                    self.check_server_down()
                    raise Exception('Tushare server is down, skip update')
                else: 
                    raise e
            else:
                break
            timeout_max_retries -= 1
            dates = self.target_dates()
        self.logger.success(f'Fetched for {Dates(updated_dates)}' , add_prefix = True)
        return Base.UpdateFlag.SUCCESS

    def locked_fetch(self , tushare_api : Callable[..., T] , *args, **kwargs) -> pd.DataFrame:
        """fetch from tushare with threading lock"""
        with TS.lock:
            ret = tushare_api(*args, **kwargs)
        if not isinstance(ret , pd.DataFrame):
            raise TypeError(f'{self} must return a pd.DataFrame, but got {ret}')
        return ret

    def iterate_fetch(self , tushare_api : Callable[..., T] , limit = 2000 , max_fetch_times = -1 , breakpoint : bool = False , **kwargs) -> pd.DataFrame:
        """iterate fetch from tushare with threading lock (defined in TushareIterateFetcher)"""
        iterate_fetcher = TushareIterateFetcher(self.__class__.__name__ , tushare_api , limit , max_fetch_times = max_fetch_times , breakpoint = breakpoint , **kwargs)
        return iterate_fetcher.fetch()
        
    def missing_dates(self , **kwargs) -> Dates:
        """get missing dates"""
        return Dates()

    def stored_dates(self) -> Dates:
        """get stored dates"""
        return DB.dates(self.DB_SRC , self.DB_KEY)

    @classmethod
    def update_missing(cls) -> Base.UpdateFlag:
        """update missing dates"""
        fetcher = cls()
        missing_dates = fetcher.missing_dates()
        if missing_dates.empty:
            return Base.UpdateFlag.SKIPPED
        fetcher.update_dates(missing_dates , step = 1)
        return Base.UpdateFlag.SUCCESS
        
class InfoFetcher(TushareFetcher):
    """base class of info fetcher , implement get_data for real use"""
    DB_TYPE = 'info'
    UPDATE_FREQ = 'd'
    DB_SRC = 'information_ts'

    def target_dates(self) -> Dates:
        return self._info_fetcher_update_date()

class TimeSeriesFetcher(TushareFetcher):
    """base class of time series fetcher , implement get_data for real use"""
    DB_TYPE = 'time_series'
    UPDATE_FREQ = 'd'

    def target_dates(self) -> Dates:
        return self._info_fetcher_update_date()

class TradeDataFetcher(TushareFetcher):
    """base class of date fetcher , implement get_data for real use"""
    DB_TYPE = 'date'
    DB_SRC = 'trade_ts'

    def target_dates(self) -> Dates: 
        return self._date_fetcher_update_dates()

class DayFetcher(TradeDataFetcher):
    """base class of day fetcher , implement get_data for real use"""
    UPDATE_FREQ = 'd'

    def missing_dates(self , updated = True , **kwargs) -> Dates:
        """get missing dates"""
        start , end = CALENDAR.update_schedule(self.START_DATE , None , key = 'tushare')
        dates = Dates(start , end).diff(self.stored_dates())
        return dates.diff(self.stored_dates())

    def target_dates(self) -> Dates:
        """get target dates"""
        return self._date_fetcher_update_dates().intersect(self.missing_dates())

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
    DATA_FREQ : Base.lit.FreqFinData = 'q'
    CONSIDER_FUTURE = False

    def target_dates(self) -> Dates:
        return self._fina_fetcher_update_dates(self.DATA_FREQ , self.CONSIDER_FUTURE)

class RollingFetcher(TushareFetcher):
    """base class of rolling fetcher , implement get_data for real use"""
    DB_TYPE = 'rolling'
    UPDATE_FREQ = 'd'

    ROLLING_BACK_DAYS = 30
    ROLLING_SEP_DAYS = 50
    ROLLING_DATE_COL = 'date'
    SAVEING_DATE_COL = True

    def __init__(self , *args , **kwargs):
        super().__init__(*args , **kwargs)
        assert self.ROLLING_BACK_DAYS > 0 , f'{self.__class__.__name__} ROLLING_BACK_DAYS must be positive'
        assert self.ROLLING_SEP_DAYS > 0 , f'{self.__class__.__name__} ROLLING_BACK_DAYS must be positive'

    def target_dates(self) -> Dates:
        """get update dates for rolling fetcher"""
        assert self.UPDATE_FREQ , f'{self.__class__.__name__} UPDATE_FREQ must be set'
        update_to = CALENDAR.update_to(key = 'tushare')
        update = self.updatable(self.last_update_date() , self.UPDATE_FREQ , update_to)
        if not update: 
            return Dates()

        rolling_last_date = max(self.START_DATE , CALENDAR.cd(self.last_date() , -self.ROLLING_BACK_DAYS))
        update_dates = TS.dates_to_update(rolling_last_date , self.UPDATE_FREQ , update_to)
        d , dates = update_dates[0] , [update_dates[0]]
        while True:
            d = CALENDAR.cd(d , self.ROLLING_SEP_DAYS)
            dates.append(min(d , update_to))
            if d >= update_to: 
                break
        return Dates(dates)

    def update_dates(self , dates : Dates) -> Dates:
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
                DB.save(subdf , self.DB_SRC , self.DB_KEY , date = date , indent = self.indent + 1 , vb_level = self.vb_level + 1)
                updated_dates.append(date)
        return Dates(updated_dates)