import torch
import numpy as np
import pandas as pd
from typing import Any , Literal

from ....basic import PATH
from ....func.time import today
from ....func.singleton import singleton

_calendar = pd.read_feather(PATH.get_target_path('information_ts' , 'calendar')).loc[:,['calendar' , 'trade']]
_trd = _calendar[_calendar['trade'] == 1].reset_index(drop=True)
_trd['td'] = _trd['calendar']
_trd['pre'] = _trd['calendar'].shift(1, fill_value=-1)
_calendar = _calendar.merge(_trd.drop(columns='trade') , on = 'calendar' , how = 'left').ffill()
_calendar['cd_index'] = np.arange(len(_calendar))
_calendar['td_index'] = _calendar['trade'].cumsum() - 1
_calendar = _calendar.astype(int).set_index('calendar')
_cal_cal = _calendar.reset_index().set_index('cd_index')
_cal_trd = _calendar[_calendar['trade'] == 1].reset_index().set_index('td_index')

class TradeDate:
    def __init__(self , date : int | Any , force_trade_date = False):
        self.calendar_date = int(date)
        self.trade_date : int = self.calendar_date if force_trade_date else _calendar['td'].loc[self.calendar_date]

    def __repr__(self):
        return str(self.trade_date)

    def __int__(self): 
        return int(self.trade_date)
    
    def __str__(self):
        return str(self.trade_date)
    
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
        if n == 0: return self
        d_index = _calendar['td_index'].loc[self.trade_date] + n
        d_index = np.maximum(np.minimum(d_index , len(_calendar) - 1) , 0)
        return self.__class__(_calendar[_calendar['td_index'] == d_index]['td'].iloc[0] , force_trade_date = True) 

    @staticmethod
    def as_numpy(td):
        if isinstance(td , int): td = np.array([td])
        elif isinstance(td , pd.Series): td = td.to_numpy()
        elif isinstance(td , list): td = np.array([td])
        elif isinstance(td , torch.Tensor): td = td.cpu().numpy()
        return td.astype(int)

@singleton
class TradeCalendar:

    @property
    def calendar(self):
        return _calendar
    
    @property
    def cal_cal(self):
        return _cal_cal
    
    @property
    def cal_trd(self):
        return _cal_trd

    @property
    def today(self):
        return today()
    
    @classmethod
    def td(cls , date : int | TradeDate , offset : int = 0):
        return TradeDate(date).offset(offset)
        
    @classmethod
    def tds(cls , date):
        return TradeDate.as_numpy(_calendar.loc[date , 'td'])
    
    @classmethod
    def previous(cls , date):
        return TradeDate.as_numpy(_calendar.loc[date , 'pre'])
    
    @classmethod
    def offset(cls , date , n : Any = 0 , type : Literal['t' , 'c'] = 't'):
        if type == 't':
            d_index = _calendar.loc[date , 'td_index'] + n
            d_index = np.maximum(np.minimum(d_index , len(_cal_trd) - 1) , 0)
            td = _cal_trd.loc[d_index , 'calendar']
        else:
            d_index = _calendar.loc[date , 'cd_index'] + n
            d_index = np.maximum(np.minimum(d_index , len(_cal_cal) - 1) , 0)
            td = _cal_cal.loc[d_index , 'calendar']
        return TradeDate.as_numpy(td)
    
    @classmethod
    def trailing(cls , date , n : int , type : Literal['t' , 'c'] = 't' , ):
        if type == 't':
            td = _cal_trd[_cal_trd['calendar'] <= date]
        else:
            td = _cal_cal[_cal_cal['calendar'] <= date]
        return np.sort(td[-n:]['calendar'].to_numpy())
    
    @classmethod
    def td_within(cls , start_dt : int | TradeDate = -1 , end_dt : int | TradeDate = 99991231 , step : int = 1 , until_today = True):
        start_dt , end_dt = int(start_dt) , int(end_dt)
        dates = _cal_trd['calendar'].to_numpy()
        if until_today: end_dt = min(end_dt , today())
        return dates[(dates >= start_dt) & (dates <= end_dt)][::step]
    
    @classmethod
    def calendar_start(cls): return _calendar.index.min()

    @classmethod
    def calendar_end(cls): return _calendar.index.max()

    @classmethod
    def td_start_end(cls , reference_date , period_num : int , 
                     period_type : Literal['d','w','m','q','y'] = 'm' , 
                     lag_num : int = 0):
        td = TradeDate(reference_date)
        pdays = {'d':1 , 'w':7 , 'm':21 , 'q':63 , 'y':252}[period_type]
        start_date = td - pdays * (period_num + lag_num) + 1
        end_date   = td - pdays * lag_num
        return start_date , end_date
    
    @classmethod
    def is_trade_date(cls , date : int | TradeDate):
        return _calendar.loc[int(date) , 'trade'] == 1
