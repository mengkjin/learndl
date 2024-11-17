import torch
import numpy as np
import pandas as pd

from datetime import datetime , timedelta , time
from pathlib import Path
from typing import Any , Literal

from .. import path as PATH

def today(offset = 0 , astype : Any = int):
    d = datetime.today() + timedelta(days=offset)
    return astype(d.strftime('%Y%m%d'))

def load_calendar():
    calendar = PATH.db_load('information_ts' , 'calendar' , raise_if_not_exist = True).loc[:,['calendar' , 'trade']]
    if (res_path := Path('./reserved_calendar.json')).exists():
        res_calendar = pd.read_json(res_path).loc[:,['calendar' , 'trade']]
        calendar = pd.concat([calendar , res_calendar[res_calendar['calendar'] > calendar['calendar'].max()]]).sort_values('calendar')

    trd = calendar[calendar['trade'] == 1].reset_index(drop=True)
    trd['td'] = trd['calendar']
    trd['pre'] = trd['calendar'].shift(1, fill_value=-1)
    calendar = calendar.merge(trd.drop(columns='trade') , on = 'calendar' , how = 'left').ffill()
    calendar['cd_index'] = np.arange(len(calendar))
    calendar['td_index'] = calendar['trade'].cumsum() - 1
    calendar['td_forward_index'] = calendar['td_index'] + 1 - calendar['trade']
    calendar['td_forward'] = trd.iloc[calendar['td_forward_index'].clip(upper = len(trd) - 1).to_numpy()]['calendar'].values
    calendar = calendar.astype(int).set_index('calendar')
    cal_cal = calendar.reset_index().set_index('cd_index')
    cal_trd = calendar[calendar['trade'] == 1].reset_index().set_index('td_index')
    return calendar , cal_cal , cal_trd

_CALENDAR , _CALENDAR_CAL , _CALENDAR_TRD = load_calendar()

class TradeDate(int):
    def __init__(self , date : int | Any , force_trade_date = False):
        self.cd = int(date)
        self.td : int = self.cd if force_trade_date else _CALENDAR['td'].loc[self.cd]

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
        if n == 0: return cls(td0)
        d_index = _CALENDAR['td_index'].loc[td0.td] + n
        d_index = np.maximum(np.minimum(d_index , len(_CALENDAR) - 1) , 0)
        new_date = _CALENDAR[_CALENDAR['td_index'] == d_index].iloc[0]['td']
        return cls(new_date) 

    @staticmethod
    def as_numpy(td):
        if isinstance(td , int): td = np.array([td])
        elif isinstance(td , pd.Series): td = td.to_numpy()
        elif isinstance(td , list): td = np.array([td])
        elif isinstance(td , torch.Tensor): td = td.cpu().numpy()
        return td.astype(int)

class TradeCalendar:
    _instance = None

    def __new__(cls , *args , **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def calendar(): return _CALENDAR
    @staticmethod
    def cal_cal(): return _CALENDAR_CAL
    @staticmethod
    def cal_trd(): return _CALENDAR_TRD
    @staticmethod
    def today():   return today()
    @staticmethod
    def update_to():
        return today(-1 if datetime.now().time() <= time(19, 0, 0) else 0)
    
    @staticmethod
    def _date_convert_to_index(date):
        if isinstance(date , TradeDate): date = [date.td]
        elif isinstance(date , pd.Series): date = date.to_numpy().astype(int).tolist()
        return date
    
    @staticmethod
    def td(date : int | TradeDate , offset : int = 0):
        return TradeDate(date).offset(offset)
    
    @classmethod
    def td_array(cls , date , offset : int = 0 , backward = True):
        td_arr = TradeDate.as_numpy(_CALENDAR.loc[cls._date_convert_to_index(date) , 'td' if backward else 'td_forward'])
        if offset != 0: 
            d_index = TradeDate.as_numpy(_CALENDAR.loc[td_arr , 'td_index']) + offset
            d_index = np.maximum(np.minimum(d_index , len(_CALENDAR_TRD) - 1) , 0)
            td_arr = TradeDate.as_numpy(_CALENDAR_TRD.loc[d_index , 'calendar'])
        return td_arr
    
    @staticmethod
    def cd(date : int | TradeDate , offset : int = 0):
        d = date.cd if isinstance(date , TradeDate) else date
        if offset == 0: return d
        d = datetime.strptime(str(d) , '%Y%m%d') + timedelta(days=offset)
        return int(d.strftime('%Y%m%d'))
    
    @staticmethod
    def cd_array(date , offset : int = 0):
        cd_arr = np.array([d.cd if isinstance(d , TradeDate) else int(d) for d in date])
        if offset != 0: 
            d_index = _CALENDAR.loc[cd_arr , 'cd_index'] + offset
            d_index = np.maximum(np.minimum(d_index , len(_CALENDAR_CAL) - 1) , 0)
            cd_arr = _CALENDAR_CAL.loc[d_index , 'calendar'].astype(int).to_numpy()
        return cd_arr
    
    @staticmethod
    def td_diff(date1 , date2) -> int | Any:
        diff = _CALENDAR.loc[[date1 , date2] , 'td_index'].astype(int).diff().iloc[1]
        assert isinstance(diff , int) , f'{date1} and {date2} are not in calendar'
        return diff
    
    @staticmethod
    def cd_diff(date1 , date2) -> int | Any:
        diff = _CALENDAR.loc[[date1 , date2] , 'cd_index'].astype(int).diff().iloc[1]
        assert isinstance(diff , int) , f'{date1} and {date2} are not in calendar'
        return diff
    
    @staticmethod
    def td_trailing(date , n : int):
        return np.sort(_CALENDAR_TRD[_CALENDAR_TRD['calendar'] <= date].iloc[-n:]['calendar'].to_numpy())
    
    @staticmethod
    def cd_trailing(date , n : int):
        return np.sort(_CALENDAR_CAL[_CALENDAR_CAL['calendar'] <= date].iloc[-n:]['calendar'].to_numpy())
    
    @staticmethod
    def td_within(start_dt : int | TradeDate | None = -1 , end_dt : int | TradeDate | None = 99991231 , step : int = 1 , until_today = True):
        if start_dt is None: start_dt = -1
        if end_dt is None: end_dt = 99991231
        start_dt , end_dt = int(start_dt) , int(end_dt)
        dates = _CALENDAR_TRD['calendar'].to_numpy()
        if until_today: end_dt = min(end_dt , today())
        return dates[(dates >= start_dt) & (dates <= end_dt)][::step]
    
    @classmethod
    def td_filter(cls , date_list):
        td_list = cls.td_within(min(date_list) , max(date_list))
        return td_list[np.isin(td_list , date_list)]
    
    @staticmethod
    def cd_within(start_dt : int | TradeDate = -1 , end_dt : int | TradeDate = 99991231 , step : int = 1 , until_today = True):
        start_dt , end_dt = int(start_dt) , int(end_dt)
        dates = _CALENDAR_CAL['calendar'].to_numpy()
        if until_today: end_dt = min(end_dt , today())
        return dates[(dates >= start_dt) & (dates <= end_dt)][::step]
    
    @staticmethod
    def calendar_start(): return _CALENDAR.index.min()

    @staticmethod
    def calendar_end(): return _CALENDAR.index.max()

    @staticmethod
    def td_start_end(reference_date , period_num : int , 
                     freq : Literal['d','w','m','q','y'] = 'm' , 
                     lag_num : int = 0):
        td = TradeDate(reference_date)
        pdays = {'d':1 , 'w':7 , 'm':21 , 'q':63 , 'y':252}[freq]
        start_date = td - pdays * (period_num + lag_num) + 1
        end_date   = td - pdays * lag_num
        return start_date , end_date
    
    @staticmethod
    def as_trade_date(date : int | Any):
        return TradeDate(date)

    @staticmethod
    def is_trade_date(date : int | TradeDate):
        return _CALENDAR.loc[int(date) , 'trade'] == 1
    
    @staticmethod
    def trade_dates():
        return _CALENDAR_TRD['calendar'].to_numpy()

CALENDAR = TradeCalendar()