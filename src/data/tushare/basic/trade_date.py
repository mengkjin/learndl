import torch
import numpy as np
import pandas as pd
from typing import Any , Literal

from ....basic import PATH
from ....func.time import today
from ....func.singleton import singleton

def load_calendar():
    calendar = PATH.db_load('information_ts' , 'calendar').loc[:,['calendar' , 'trade']]
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

class TradeDate:
    def __init__(self , date : int | Any , force_trade_date = False):
        self.calendar_date = int(date)
        self.trade_date : int = self.calendar_date if force_trade_date else _CALENDAR['td'].loc[self.calendar_date]

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
        d_index = _CALENDAR['td_index'].loc[self.trade_date] + n
        d_index = np.maximum(np.minimum(d_index , len(_CALENDAR) - 1) , 0)
        return self.__class__(_CALENDAR[_CALENDAR['td_index'] == d_index]['td'].iloc[0] , force_trade_date = True) 

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
        return _CALENDAR
    
    @property
    def cal_cal(self):
        return _CALENDAR_CAL
    
    @property
    def cal_trd(self):
        return _CALENDAR_TRD

    @property
    def today(self):
        return today()
    
    @staticmethod
    def date_convert(date):
        if isinstance(date , TradeDate): date = int(date)
        elif isinstance(date , pd.Series): date = date.to_numpy().astype(int).tolist()
        return date
    
    @classmethod
    def td(cls , date : int | TradeDate , offset : int = 0):
        return TradeDate(date).offset(offset)
        
    @classmethod
    def tds(cls , date):
        return TradeDate.as_numpy(_CALENDAR.loc[cls.date_convert(date) , 'td'])
    
    @classmethod
    def tds_forward(cls , date):
        return TradeDate.as_numpy(_CALENDAR.loc[cls.date_convert(date) , 'td_forward'])
    
    @classmethod
    def previous(cls , date):
        return TradeDate.as_numpy(_CALENDAR.loc[cls.date_convert(date) , 'pre'])
    
    @classmethod
    def offset(cls , date , n : Any = 0 , type : Literal['t' , 'c'] = 't'):
        if type == 't':
            d_index = _CALENDAR.loc[cls.date_convert(date) , 'td_index'] + n
            d_index = np.maximum(np.minimum(d_index , len(_CALENDAR_TRD) - 1) , 0)
            td = _CALENDAR_TRD.loc[d_index , 'calendar']
        else:
            d_index = _CALENDAR.loc[cls.date_convert(date) , 'cd_index'] + n
            d_index = np.maximum(np.minimum(d_index , len(_CALENDAR_CAL) - 1) , 0)
            td = _CALENDAR_CAL.loc[d_index , 'calendar']
        return TradeDate.as_numpy(td)
    
    @classmethod
    def trailing(cls , date , n : int , type : Literal['t' , 'c'] = 't' , ):
        if type == 't':
            td = _CALENDAR_TRD[_CALENDAR_TRD['calendar'] <= date]
        else:
            td = _CALENDAR_CAL[_CALENDAR_CAL['calendar'] <= date]
        return np.sort(td[-n:]['calendar'].to_numpy())
    
    @classmethod
    def td_within(cls , start_dt : int | TradeDate = -1 , end_dt : int | TradeDate = 99991231 , step : int = 1 , until_today = True):
        start_dt , end_dt = int(start_dt) , int(end_dt)
        dates = _CALENDAR_TRD['calendar'].to_numpy()
        if until_today: end_dt = min(end_dt , today())
        return dates[(dates >= start_dt) & (dates <= end_dt)][::step]
    
    @classmethod
    def calendar_start(cls): return _CALENDAR.index.min()

    @classmethod
    def calendar_end(cls): return _CALENDAR.index.max()

    @classmethod
    def td_start_end(cls , reference_date , period_num : int , 
                     freq : Literal['d','w','m','q','y'] = 'm' , 
                     lag_num : int = 0):
        td = TradeDate(reference_date)
        pdays = {'d':1 , 'w':7 , 'm':21 , 'q':63 , 'y':252}[freq]
        start_date = td - pdays * (period_num + lag_num) + 1
        end_date   = td - pdays * lag_num
        return start_date , end_date
    
    @classmethod
    def is_trade_date(cls , date : int | TradeDate):
        return _CALENDAR.loc[int(date) , 'trade'] == 1
