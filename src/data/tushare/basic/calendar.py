import torch
import numpy as np
import pandas as pd
from typing import Any , Literal

from ....basic import PATH
from ....func.time import today
from ....func.singleton import singleton_threadsafe

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

@singleton_threadsafe
class TradeCalendar:
    def __init__(self) -> None:
        self.calendar = _calendar
        self.cal_cal = _cal_cal
        self.cal_trd = _cal_trd
        
    def td(self , date):
        return TradeDate.as_numpy(self.calendar.loc[date , 'td'])
    
    def pre(self , date):
        return TradeDate.as_numpy(self.calendar.loc[date , 'pre'])
    
    def offset(self , date , n : Any = 0 , type : Literal['t' , 'c'] = 't'):
        if type == 't':
            d_index = self.calendar.loc[date , 'td_index'] + n
            d_index = np.maximum(np.minimum(d_index , len(self.cal_trd) - 1) , 0)
            td = self.cal_trd.loc[d_index , 'calendar']
        else:
            d_index = self.calendar.loc[date , 'cd_index'] + n
            d_index = np.maximum(np.minimum(d_index , len(self.cal_cal) - 1) , 0)
            td = self.cal_cal.loc[d_index , 'calendar']
        return TradeDate.as_numpy(td)
    
    def trailing(self , date , n : int , type : Literal['t' , 'c'] = 't' , ):
        if type == 't':
            td = self.cal_trd[self.cal_trd['calendar'] <= date]
        else:
            td = self.cal_cal[self.cal_cal['calendar'] <= date]
        return np.sort(td[-n:]['calendar'].to_numpy())
    
    def td_within(self , start_dt : int = -1 , end_dt : int = 99991231 , step : int = 1 , until_today = True):
        dates = self.cal_trd['calendar'].to_numpy()
        if until_today: end_dt = min(end_dt , today())
        return dates[(dates >= start_dt) & (dates <= end_dt)][::step]
    
    @property
    def calendar_start(self): return self.calendar.index.min()
    @property
    def calendar_end(self): return self.calendar.index.max()
