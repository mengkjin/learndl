import torch
import numpy as np
import pandas as pd

from datetime import datetime , timedelta , time
from typing import Any , Literal , Sequence

from src.proj import PATH
from src.basic import DB
from src.func.singleton import singleton

@singleton
class _Calendars:
    def __init__(self):
        self.load_calendars()

    def load_calendars(self):
        calendar = DB.load('information_ts' , 'calendar' , raise_if_not_exist = True).loc[:,['calendar' , 'trade']]
        if (res_path := PATH.conf.joinpath('glob','reserved_calendar.json')).exists():
            res_calendar = pd.read_json(res_path).loc[:,['calendar' , 'trade']]
            calendar = pd.concat([calendar , res_calendar[res_calendar['calendar'] > calendar['calendar'].max()]]).sort_values('calendar')

        trd = calendar.query('trade == 1').reset_index(drop=True)
        trd['td'] = trd['calendar']
        trd['pre'] = trd['calendar'].shift(1, fill_value=-1)
        calendar = calendar.merge(trd.drop(columns='trade') , on = 'calendar' , how = 'left').ffill()
        calendar['cd_index'] = np.arange(len(calendar))
        calendar['td_index'] = calendar['trade'].cumsum() - 1
        calendar['td_forward_index'] = calendar['td_index'] + 1 - calendar['trade']
        calendar['td_forward'] = trd.iloc[calendar['td_forward_index'].clip(upper = len(trd) - 1).to_numpy(int)]['calendar'].values
        calendar = calendar.astype(int).set_index('calendar')
        cal_cal = calendar.reset_index().set_index('cd_index')
        cal_trd = calendar[calendar['trade'] == 1].reset_index().set_index('td_index')

        self.full = calendar
        self.cal = cal_cal
        self.trd = cal_trd

        self.min_date = calendar.index.min()
        self.max_date = calendar.index.max()

        self.max_td_index : int = int(cal_trd.index.max())
    
_CLD = _Calendars()

class TradeDate:
    def __new__(cls , date : int | Any , *args , **kwargs):
        if isinstance(date , TradeDate):
            return date
        return super().__new__(cls)
    def __init__(self , date : int | Any , force_trade_date = False):
        if not isinstance(date , TradeDate):
            self.cd = int(date)
            if force_trade_date or self.cd < _CLD.min_date or self.cd > _CLD.max_date:
                self.td : int = self.cd 
            else:
                self.td = _CLD.full['td'].loc[self.cd]

    def __repr__(self):
        return str(self.td)

    def __int__(self): 
        return int(self.td)
    
    def __str__(self):
        return str(self.td)
    
    def __add__(self , n : int):
        return self.offset(n)
    
    def __sub__(self , n : int):
        return self.offset(-n)
    
    def __lt__(self , other):
        return int(self) < int(other)
    
    def __le__(self , other):
        return int(self) <= int(other)
    
    def __gt__(self , other):
        return int(self) > int(other)
    
    def __ge__(self , other):
        return int(self) >= int(other)
    
    def __eq__(self , other):
        return int(self) == int(other)

    def as_int(self):
        return int(self)
    
    def offset(self , n : int):
        return self._cls_offset(self , n)

    @classmethod
    def _cls_offset(cls , td0 , n : int):
        td0 = cls(td0)
        if n == 0: 
            return td0
        elif td0 < _CLD.min_date or td0 > _CLD.max_date:
            return td0
        assert isinstance(n , (int , np.integer)) , f'n must be a integer, got {type(n)}'
        d_index = _CLD.full['td_index'].loc[td0.td] + n
        d_index = np.maximum(np.minimum(d_index , _CLD.max_td_index) , 0)
        new_date = _CLD.trd.loc[d_index,'td']
        return cls(new_date) 

    @staticmethod
    def as_numpy(td):
        if isinstance(td , int): 
            td = np.array([td])
        elif isinstance(td , pd.Series): 
            td = td.to_numpy()
        elif isinstance(td , list): 
            td = np.array(td)
        elif isinstance(td , torch.Tensor): 
            td = td.cpu().numpy()
        return td.astype(int)

class CALENDAR:
    """
    trade calendar util
    """

    ME = pd.date_range(start='1997-01-01', end='2099-12-31', freq='ME').strftime('%Y%m%d').to_numpy(int)
    QE = pd.date_range(start='1997-01-01', end='2099-12-31', freq='QE').strftime('%Y%m%d').to_numpy(int)
    YE = pd.date_range(start='1997-01-01', end='2099-12-31', freq='YE').strftime('%Y%m%d').to_numpy(int)

    @staticmethod
    def today(offset = 0) -> int: 
        d = datetime.today() 
        if offset != 0:
            d = d + timedelta(days=offset)
        return int(d.strftime('%Y%m%d'))
    @staticmethod
    def time() -> int: 
        return int(datetime.now().strftime('%H%M%S'))
    @staticmethod
    def clock(date : int = 0 , h = 0 , m = 0 , s = 0):
        if date <=0:
            date_time = (datetime.today() + timedelta(days=date))
        else:
            date_time = datetime.strptime(str(date) , '%Y%m%d')
        return int(date_time.replace(hour=h, minute=m, second=s).strftime('%Y%m%d%H%M%S'))
    @classmethod
    def is_updated_today(cls , modified_time : int , hour = 20 , minute = 0):
        if modified_time < 1e8: 
            modified_time = modified_time * 1000000
        current_time = datetime.now()
        required_date = 0 if (current_time.hour >= hour and current_time.minute >= minute) else -1
        required_time = CALENDAR.clock(required_date , hour , minute)
        return modified_time >= required_time
    @classmethod
    def now(cls):
        return cls.today() * 1000000 + cls.time()
    @classmethod
    def update_to(cls):
        return cls.today(-1 if datetime.now().time() <= time(19, 59, 0) else 0)
    @staticmethod
    def updated():
        return DB.max_date('trade_ts' , 'day')
    @staticmethod
    def _date_convert_to_index(date):
        if isinstance(date , TradeDate): 
            date = [date.td]
        elif isinstance(date , pd.Series): 
            date = date.to_numpy(int) #.tolist()
        return date
    
    @staticmethod
    def td(date : int | TradeDate , offset : int = 0):
        return TradeDate(date).offset(offset)
    
    @classmethod
    def td_array(cls , date , offset : int = 0 , backward = True):
        td_arr = TradeDate.as_numpy(_CLD.full.loc[cls._date_convert_to_index(date) , 'td' if backward else 'td_forward'])
        if offset != 0: 
            d_index = TradeDate.as_numpy(_CLD.full.loc[td_arr , 'td_index']) + offset
            d_index = np.maximum(np.minimum(d_index , len(_CLD.trd) - 1) , 0)
            td_arr = TradeDate.as_numpy(_CLD.trd.loc[d_index , 'calendar'])
        return td_arr
    
    @staticmethod
    def cd(date : int | TradeDate , offset : int = 0):
        d = date.cd if isinstance(date , TradeDate) else int(date)
        if offset == 0 or d <= _CLD.min_date or d >= _CLD.max_date: 
            return d
        d = datetime.strptime(str(d) , '%Y%m%d') + timedelta(days=offset)
        return int(d.strftime('%Y%m%d'))
    
    @staticmethod
    def cd_array(date , offset : int = 0):
        cd_arr = np.array([d.cd if isinstance(d , TradeDate) else int(d) for d in date])
        if offset != 0: 
            cd_arr = np.minimum(cd_arr , CALENDAR.calendar_end())
            d_index = _CLD.full.loc[cd_arr , 'cd_index'] + offset
            d_index = np.maximum(np.minimum(d_index , len(_CLD.cal) - 1) , 0)
            cd_arr = _CLD.cal.loc[d_index , 'calendar'].to_numpy(int)
        return cd_arr
    
    @staticmethod
    def td_diff(date1 , date2) -> int | Any:
        diff = _CLD.full.loc[[date1 , date2] , 'td_index'].astype(int).diff().iloc[1]
        assert isinstance(diff , int) , f'{date1} and {date2} are not in calendar'
        return diff
    
    @staticmethod
    def cd_diff(date1 , date2) -> int | Any:
        try:
            diff = int(_CLD.full.loc[[date1 , date2] , 'cd_index'].diff().dropna().astype(int).item())
        except Exception:
            diff = (datetime.strptime(str(date1), '%Y%m%d') - datetime.strptime(str(date2), '%Y%m%d')).days
        return diff
    
    @classmethod
    def td_diff_array(cls , date1_arr , date2_arr) -> int | Any:
        td1_arr = TradeDate.as_numpy(_CLD.full.loc[cls._date_convert_to_index(date1_arr) , 'td_index'])
        td2_arr = TradeDate.as_numpy(_CLD.full.loc[cls._date_convert_to_index(date2_arr) , 'td_index'])
        return td1_arr - td2_arr
    
    @classmethod
    def cd_diff_array(cls , date1_arr , date2_arr) -> int | Any:
        cd1_arr = TradeDate.as_numpy(_CLD.full.loc[cls._date_convert_to_index(date1_arr) , 'cd_index'])
        cd2_arr = TradeDate.as_numpy(_CLD.full.loc[cls._date_convert_to_index(date2_arr) , 'cd_index'])
        return cd1_arr - cd2_arr
    
    @staticmethod
    def td_trailing(date , n : int):
        return np.sort(_CLD.trd[_CLD.trd['calendar'] <= date].iloc[-n:]['calendar'].to_numpy()).astype(int)
    
    @staticmethod
    def cd_trailing(date , n : int):
        return np.sort(_CLD.cal[_CLD.cal['calendar'] <= date].iloc[-n:]['calendar'].to_numpy()).astype(int)

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
        dates = cls.slice(_CLD.trd['calendar'].to_numpy(int) , start_dt , end_dt)
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
        return np.setdiff1d(target_dates , source_dates)
    
    @classmethod
    def td_filter(cls , date_list):
        td_list = cls.td_within(min(date_list) , max(date_list))
        return td_list[np.isin(td_list , date_list)]
    
    @classmethod
    def cd_within(cls , start_dt : int | TradeDate | None = None , end_dt : int | TradeDate | None = None , step : int = 1 , 
                  until_today = True , updated = False):    
        dates = cls.slice(_CLD.cal['calendar'].to_numpy(int) , start_dt , end_dt)
        if until_today: 
            dates = dates[dates <= cls.today()]
        if updated: 
            dates = dates[dates <= cls.updated()]
        return dates[::step]
    
    @staticmethod
    def calendar_start(): 
        md = np.min(_CLD.full.index.to_numpy())
        return int(md)

    @staticmethod
    def calendar_end(): 
        md = np.max(_CLD.full.index.to_numpy())
        return int(md)

    @staticmethod
    def td_start_end(reference_date , period_num : int , 
                     freq : Literal['d','w','m','q','y'] = 'm' , 
                     lag_num : int = 0):
        td = TradeDate(reference_date)
        pdays = {'d':1 , 'w':7 , 'm':21 , 'q':63 , 'y':252}[freq]
        start_dt = td - pdays * (period_num + lag_num) + 1
        end_dt   = td - pdays * lag_num
        return start_dt , end_dt
    
    @staticmethod
    def as_trade_date(date : int | Any):
        return TradeDate(date)

    @staticmethod
    def is_trade_date(date : int | TradeDate):
        return _CLD.full.loc[int(date) , 'trade'] == 1
    
    @staticmethod
    def trade_dates():
        return _CLD.trd['calendar'].to_numpy(int)

    @classmethod
    def slice(cls , dates , start_dt : int | TradeDate | None = None , end_dt : int | TradeDate | None = None , year : int | None = None) -> np.ndarray:
        dates = dates[(dates >= cls.start_dt(start_dt)) & (dates <= cls.end_dt(end_dt))]
        if year  is not None: 
            dates = dates[(dates // 10000) == year]
        return dates

    @staticmethod
    def format(date : Any , old_fmt = '%Y%m%d' , new_fmt = '%Y%m%d'):
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
            return np.array([]).astype(int)
        return cls.qe_within(min(incomplete_qtr_ends) , max(incomplete_qtr_ends))
    
    @classmethod
    def check_rollback_date(cls , rollback_date : int | None , max_rollback_days : int = 10):
        if rollback_date is None: 
            return
        earliest_rollback_date = cls.td(cls.updated() , -max_rollback_days)
        assert rollback_date >= earliest_rollback_date , \
            f'rollback_date {rollback_date} is too early, must be at least {earliest_rollback_date}'

    @classmethod
    def dates_str(cls , dates : np.ndarray | list[int | None] | Any) -> str:
        if isinstance(dates , np.ndarray):
            d0 , d1 = min(dates) , max(dates)
            dstr = f'{d0}~{d1}' if d0 != d1 else str(d0)
        elif isinstance(dates , list):
            valid_dates = [d for d in dates if d is not None]
            if None in dates:
                d0 , d1 = min(valid_dates) , 99991231
            else:
                d0 , d1 = min(valid_dates) , max(valid_dates)
            dstr = f'{d0}~{d1}' if d0 != d1 else str(d0)
        else:
            dstr = str(dates) if dates is not None else 'None'
        return dstr