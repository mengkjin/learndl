import numpy as np
import pandas as pd

from datetime import datetime , timedelta , time
from typing import Any , Literal , Sequence

import src.proj.db as DB

from .basic import BC , BJTZ
from .trade_date import TradeDate

def date_convert_to_index(date):
    if isinstance(date , TradeDate): 
        date = [date.td]
    elif isinstance(date , pd.Series): 
        date = date.to_numpy(int) #.tolist()
    return date

class CALENDAR:
    """
    trade calendar util
    """

    ME = pd.date_range(start='1997-01-01', end='2099-12-31', freq='ME').strftime('%Y%m%d').to_numpy(int)
    QE = pd.date_range(start='1997-01-01', end='2099-12-31', freq='QE').strftime('%Y%m%d').to_numpy(int)
    YE = pd.date_range(start='1997-01-01', end='2099-12-31', freq='YE').strftime('%Y%m%d').to_numpy(int)

    _update_to = None

    @staticmethod
    def now(bj_tz = True) -> datetime:
        if bj_tz:
            return datetime.now(BJTZ)
        else:
            return datetime.now()
    @classmethod
    def today(cls , offset = 0 , bj_tz = True) -> int: 
        d = cls.now(bj_tz)
        if offset != 0:
            d = d + timedelta(days=offset)
        return int(d.strftime('%Y%m%d'))
    @classmethod
    def time(cls , bj_tz = True) -> int: 
        return int(cls.now(bj_tz).strftime('%H%M%S'))
    @staticmethod
    def datetime(year : int = 2000 , month : int = 1 , day : int = 1 , * , lock_month = False) -> int:
        if month > 12 or month <= 0:
            year += (month - 1) // 12
            month = (month - 1) % 12 + 1
        date1 = datetime(year , month , 1)
        real_month = date1.month if lock_month else None
        date = date1 + timedelta(days=day - 1)
        if real_month is not None and date.month != real_month:
            if date.month < real_month:
                date = date1.replace(day = 1)
            else:
                date = date1.replace(month = real_month + 1 , day = 1) - timedelta(days = 1)
        return int(date.strftime('%Y%m%d'))
    @classmethod
    def clock(cls , date : int = 0 , h = 0 , m = 0 , s = 0):
        if date <=0:
            date_time = (cls.now(bj_tz = False) + timedelta(days=date))
        else:
            date_time = datetime.strptime(str(date) , '%Y%m%d')
        return int(date_time.replace(hour=h, minute=m, second=s).strftime('%Y%m%d%H%M%S'))
    @classmethod
    def is_updated_today(cls , modified_time : int | None , hour = 20 , minute = 0):
        if modified_time is None:
            return False
        if modified_time < 1e8: 
            modified_time = modified_time * 1000000
        current_time = cls.now(bj_tz = True)
        required_date = 0 if (current_time.hour >= hour and current_time.minute >= minute) else -1
        required_time = cls.clock(required_date , hour , minute)
        return modified_time >= required_time
    
    @classmethod
    def update_to(cls):
        if cls._update_to is None:
            cls._update_to = cls.today(-1 if cls.now(bj_tz = True).time() <= time(19, 59, 0) else 0)
        return cls._update_to
    @staticmethod
    def updated(date : int | None = None):
        updated_date = DB.max_date('trade_ts' , 'day')
        if date is not None:
            updated_date = min(updated_date , date)
        return updated_date
    
    @staticmethod
    def td(date : int | TradeDate , offset : int = 0):
        return TradeDate(date).offset(offset)
    
    @classmethod
    def td_array(cls , date , offset : int = 0 , backward = True):
        td_arr = TradeDate.as_numpy(BC.full.loc[date_convert_to_index(date) , 'td' if backward else 'td_forward'])
        if offset != 0: 
            d_index = TradeDate.as_numpy(BC.full.loc[td_arr , 'td_index']) + offset
            d_index = np.maximum(np.minimum(d_index , len(BC.trd) - 1) , 0)
            td_arr = TradeDate.as_numpy(BC.trd.loc[d_index , 'calendar'])
        return td_arr
    
    @staticmethod
    def cd(date : int | TradeDate , offset : int = 0):
        d = date.cd if isinstance(date , TradeDate) else int(date)
        if offset == 0 or d <= BC.min_date or d >= BC.max_date: 
            return d
        d = datetime.strptime(str(d) , '%Y%m%d') + timedelta(days=offset)
        return int(d.strftime('%Y%m%d'))
    
    @staticmethod
    def cd_array(date , offset : int = 0):
        cd_arr = np.array([d.cd if isinstance(d , TradeDate) else int(d) for d in date])
        if offset != 0: 
            cd_arr = np.minimum(cd_arr , CALENDAR.calendar_end())
            d_index = BC.full.loc[cd_arr , 'cd_index'] + offset
            d_index = np.maximum(np.minimum(d_index , len(BC.cal) - 1) , 0)
            cd_arr = BC.cal.loc[d_index , 'calendar'].to_numpy(int)
        return cd_arr
    
    @staticmethod
    def td_diff(date1 , date2) -> int | Any:
        diff = BC.full.loc[[date1 , date2] , 'td_index'].astype(int).diff().iloc[1]
        assert isinstance(diff , int) , f'{date1} and {date2} are not in calendar'
        return diff
    
    @staticmethod
    def cd_diff(date1 , date2) -> int | Any:
        try:
            diff = int(BC.full.loc[[date1 , date2] , 'cd_index'].diff().dropna().astype(int).item())
        except Exception:
            diff = (datetime.strptime(str(date1), '%Y%m%d') - datetime.strptime(str(date2), '%Y%m%d')).days
        return diff
    
    @classmethod
    def td_diff_array(cls , date1_arr , date2_arr) -> int | Any:
        td1_arr = TradeDate.as_numpy(BC.full.loc[date_convert_to_index(date1_arr) , 'td_index'])
        td2_arr = TradeDate.as_numpy(BC.full.loc[date_convert_to_index(date2_arr) , 'td_index'])
        return td1_arr - td2_arr
    
    @classmethod
    def cd_diff_array(cls , date1_arr , date2_arr) -> int | Any:
        cd1_arr = TradeDate.as_numpy(BC.full.loc[date_convert_to_index(date1_arr) , 'cd_index'])
        cd2_arr = TradeDate.as_numpy(BC.full.loc[date_convert_to_index(date2_arr) , 'cd_index'])
        return cd1_arr - cd2_arr
    
    @staticmethod
    def td_trailing(date , n : int):
        return np.sort(BC.trd[BC.trd['calendar'] <= date].iloc[-n:]['calendar'].to_numpy()).astype(int)
    
    @staticmethod
    def cd_trailing(date , n : int):
        return np.sort(BC.cal[BC.cal['calendar'] <= date].iloc[-n:]['calendar'].to_numpy()).astype(int)

    @classmethod
    def start_dt(cls , date : int | TradeDate | None) -> int:
        date_dt = 19900101 if date is None else int(date)
        if date_dt < 0: 
            date_dt = cls.today(date_dt)
        return date_dt
    
    @classmethod
    def end_dt(cls , date : int | TradeDate | None) -> int:
        date = 99991231 if date is None else int(date)
        if date < 0: 
            date = cls.today(date)
        return date

    @classmethod
    def td_within(cls , start_dt : int | TradeDate | None = None , 
                  end_dt : int | TradeDate | None = None , 
                  step : int = 1 , until_today = True , slice : tuple[Any,Any] | None = None , updated = False):
        dates = cls.slice(BC.trd['calendar'].to_numpy(int) , start_dt , end_dt)
        if until_today: 
            dates = dates[dates <= cls.today()]
        if updated: 
            dates = dates[dates <= cls.updated()]
        dates = dates[::step]
        if slice is not None: 
            dates = cls.slice(dates , slice[0] , slice[1])
        return dates

    @classmethod
    def diffs(cls , *args , td = True):
        '''
        return the difference between target dates and source dates
        inputs *args options:
            1. start_dt , end_dt , source_dates , target_dates will be dates within start_dt and end_dt
            2. target_dates , source_dates
        td : bool , when using start_dt and end_dt , will return TradeDate (default=True)
        '''
        assert len(args) in [2, 3] , 'Use tuple date_target must be a tuple of length 2 or 3'
        if len(args) == 2:
            target_dates , source_dates = args
        else:
            start_dt , end_dt , source_dates = args
            target_dates = cls.td_within(start_dt , end_dt) if td else cls.cd_within(start_dt , end_dt)
        if source_dates is None:
            source_dates = np.array([], dtype=int)
        return Dates(np.setdiff1d(target_dates , source_dates))
    
    @classmethod
    def td_filter(cls , date_list):
        td_list = cls.td_within(min(date_list) , max(date_list))
        return td_list[np.isin(td_list , date_list)]
    
    @classmethod
    def cd_within(cls , start_dt : int | TradeDate | None = None , end_dt : int | TradeDate | None = None , step : int = 1 , 
                  until_today = True , updated = False):    
        dates = cls.slice(BC.cal['calendar'].to_numpy(int) , start_dt , end_dt)
        if until_today: 
            dates = dates[dates <= cls.today()]
        if updated: 
            dates = dates[dates <= cls.updated()]
        return dates[::step]
    
    @staticmethod
    def calendar_start(): 
        md = np.min(BC.full.index.to_numpy())
        return int(md)

    @staticmethod
    def calendar_end(): 
        md = np.max(BC.full.index.to_numpy())
        return int(md)

    @staticmethod
    def td_start_end(reference_date , period_num : int = 1 , 
                     freq : Literal['d','w','m','q','y'] | str = 'm' , 
                     lag_num : int = 0):
        td = TradeDate(reference_date)
        pdays = {'d':1 , 'w':7 , 'm':21 , 'q':63 , 'y':252}[freq]
        start_dt = td - pdays * (period_num + lag_num) + 1
        end_dt   = td - pdays * lag_num
        return start_dt , end_dt

    @classmethod
    def cd_start_end(cls , reference_date , period_num : int = 1 , 
                     freq : Literal['d','w','m','q','y'] | str = 'm' , 
                     lag_num : int = 0):
        cd = int(reference_date)
        if freq in ['d' , 'w']:
            pdays = {'d':1 , 'w':7}[freq]
            start_dt = cd - pdays * (period_num + lag_num) + 1
            end_dt   = cd - pdays * lag_num
        elif freq in ['m' , 'q' , 'y']:
            year , month , day = cd // 10000 , cd % 10000 // 100 , cd % 100
            pmonths = {'m':1 , 'q':3 , 'y':12}[freq]
            start_month = month - pmonths * (period_num + lag_num)
            end_month = month - pmonths * lag_num
            start_dt = cls.cd(cls.datetime(year , start_month , day , lock_month = True) , 1)
            end_dt = cls.datetime(year , end_month , day , lock_month = True)
        else:
            raise ValueError(f'Invalid frequency: {freq}')
        return start_dt , end_dt
    
    @staticmethod
    def as_trade_date(date : int | Any):
        return TradeDate(date)

    @staticmethod
    def is_trade_date(date : int | TradeDate):
        return BC.full.loc[int(date) , 'trade'] == 1
    
    @staticmethod
    def trade_dates():
        return BC.trd['calendar'].to_numpy(int)

    @classmethod
    def slice(cls , dates , start_dt : int | TradeDate | None = None , end_dt : int | TradeDate | None = None , year : int | None = None) -> np.ndarray:
        if isinstance(dates , list):
            dates = np.array(dates)
        dates = dates[(dates >= cls.start_dt(start_dt)) & (dates <= cls.end_dt(end_dt))]
        if year  is not None: 
            dates = dates[(dates // 10000) == year]
        return dates

    @staticmethod
    def reformat(date : Any , old_fmt = '%Y%m%d' , new_fmt = '%Y%m%d'):
        if old_fmt == new_fmt: 
            return date
        return datetime.strptime(str(date), old_fmt).strftime(new_fmt)

    @classmethod
    def year_end(cls , date):
        return cls.end_dt(date) // 10000 * 10000 + 1231

    @classmethod
    def year_start(cls , date):
        return cls.start_dt(date) // 10000 * 10000 + 101

    @classmethod
    def quarter_end(cls , date):
        return cls.QE[cls.QE >= date][0]

    @classmethod
    def quarter_start(cls , date):
        return cls.year_start(date) // 10000 * 10000 + (cls.year_start(date) % 10000 // 300 - 2)  * 100 + 1

    @classmethod
    def month_end(cls , date):
        return cls.ME[cls.ME >= date][0]

    @classmethod
    def month_start(cls , date):
        return cls.year_start(date) // 100 * 100 + 1
    
    @classmethod
    def qe_trailing(cls , date , n_past = 1 , n_future = 0 , another_date = None , year_only = False):
        assert n_past >= 1 and n_future >= 0 , f'{n_past} and {n_future} must be greater than 1 and 0'
        if another_date is None: 
            another_date = date
        dates = cls.YE if year_only else cls.QE
        start = dates[dates <= min(date , another_date)][-n_past:]
        mid   = dates[(dates > min(date , another_date)) & (dates <= max(date , another_date))]
        end   = dates[dates >  max(date , another_date)][:n_future]
        return np.concatenate([start , mid , end])
    
    @classmethod
    def qe_within(cls , start_dt , end_dt , year_only = False):
        if year_only:
            return cls.YE[(cls.YE >= start_dt) & (cls.YE <= end_dt)]
        else:
            return cls.QE[(cls.QE >= start_dt) & (cls.QE <= end_dt)]

    @classmethod
    def qe_interpolate(cls , incomplete_qtr_ends : Sequence | Any):
        if len(incomplete_qtr_ends) == 0: 
            return np.array([] , dtype = int)
        return cls.qe_within(min(incomplete_qtr_ends) , max(incomplete_qtr_ends))
    
    @classmethod
    def check_rollback_date(cls , rollback_date : int | None , max_rollback_days : int = 10):
        if rollback_date is None: 
            return
        earliest_rollback_date = cls.td(cls.updated() , -max_rollback_days)
        assert rollback_date >= earliest_rollback_date , \
            f'rollback_date {rollback_date} is too early, must be at least {earliest_rollback_date}'

class Dates(np.ndarray[int , Any]):
    """
    Dates util , takes possible input types and returns a list representation of dates, can convert to np.ndarray
    accepts:
    0. empty input: 
       - empty dates[]
    1. only 1 input arg:
       - None : length 0 array
       - date : int / TradeDate / string of digits , length 1 array
       - dates : list / np.ndarray / pd.Series / Dates2 , directly converted to np.ndarray
    2. exact 2 inputs:
       - 2 date : must be both int / TradeDate / string of digits / None , start ~ end
    3. 3 inputs:
       - 1 dates / None , 2 date / None , must be dates , start , end
    """
    def __new__(cls , *args : Any , info = None):
        if len(args) == 0:
            dates = []
        elif len(args) == 1:
            if args[0] is None:
                dates = []
            elif isinstance(args[0] , (int , TradeDate , str)):
                dates = [int(args[0])]
            else:
                dates = np.array(args[0], dtype=int)
        elif len(args) == 2: 
            start , end = args
            dates = CALENDAR.td_within(start , end)
        elif len(args) == 3:
            dates , start , end = args
            dates = CALENDAR.slice(dates , start , end)
        else:
            raise ValueError(f'Invalid number of arguments: {len(args)}')
        obj = np.asarray(dates).view(cls)
        obj.info = info
        return obj

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.format_str})'

    def __str__(self) -> str:
        return self.format_str()

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)

    @property
    def empty(self) -> bool:
        return len(self) == 0

    def diffs(self , *others : Any) -> 'Dates':
        return Dates(np.setdiff1d(self , Dates(*others)) , info = self.info)

    def format_str(self) -> str:
        if self.empty:
            return 'Empty Dates'
        else:
            return f'{self.min()}~{self.max()}({len(self)}days)' if len(self) > 1 else str(self[0])
        