import time
import pandas as pd
import numpy as np

from datetime import date , datetime , timedelta
from typing import Any , Callable

from ..env import CONF

def today(offset = 0 , astype : Any = int):
    d = datetime.today() + timedelta(days=offset)
    return astype(d.strftime('%Y%m%d'))

def this_day(astype : Any = str):
    return astype(date.today().strftime('%Y%m%d'))

def date_offset(date , offset = 0 , astype : Any = int):
    if isinstance(date , (np.ndarray,pd.Index,pd.Series,list,tuple,np.ndarray)):
        is_scalar = False
        new_date = pd.DatetimeIndex(np.array(date).astype(str))
    else:
        is_scalar = True
        new_date = pd.DatetimeIndex([str(date)])
    if offset == 0:
        new_date = new_date.strftime('%Y%m%d') #type:ignore
    else:
        new_date = (new_date + pd.DateOffset(offset)).strftime('%Y%m%d') #type:ignore
    new_date = new_date.astype(astype)
    return new_date[0] if is_scalar else new_date.values

def date_diff(date1 : int | str , date2 : int | str):
    return (datetime.strptime(str(date1), '%Y%m%d') - datetime.strptime(str(date2), '%Y%m%d')).days

def date_seg(start_dt , end_dt , freq='Q' , astype : Any = int):
    dt_list = pd.date_range(str(start_dt) , str(end_dt) , freq=freq).strftime('%Y%m%d').astype(int)
    dt_starts = [date_offset(start_dt) , *date_offset(dt_list[:-1],1)]
    dt_ends = [*dt_list[:-1] , date_offset(end_dt)]
    return [(astype(s),astype(e)) for s,e in zip(dt_starts , dt_ends)]

class Timer:
    '''simple timer to print out time'''
    def __init__(self , *args , newline = False): 
        self.newline = newline
        self.key = '/'.join(args)
    def __enter__(self):
        self.start_time = time.time()
        if not CONF.SILENT: print(self.key , end=' start!\n' if self.newline else '...')
    def __exit__(self, type, value, trace):
        if not CONF.SILENT: print(self.key if self.newline else '...' , f'finished! Cost {time.time()-self.start_time:.2f} secs')

class PTimer:
    '''process timer , call to record and .summarize() to print out summary'''
    def __init__(self , record = True) -> None:
        self.recording = record
        self.recorder = {} if record else None
    class ptimer:
        def __init__(self , target : dict[str,list[float]] | None , key):
            self.target , self.key = target , key
            if self.target is not None and key not in self.target.keys(): self.target[self.key] = []
        def __enter__(self):
            if self.target is not None: self.start_time = time.time()
        def __exit__(self, type, value, trace):
            if self.target is not None: self.target[self.key].append(time.time() - self.start_time)

    def func_timer(self , func : Callable):
        def wrapper(*args , **kwargs):
            with self.ptimer(self.recorder , func.__name__):
                return func(*args , **kwargs)
        return wrapper if self.recording else func

    def __call__(self , *args):
        return self.ptimer(self.recorder , '/'.join(args))
    
    def summarize(self):
        if self.recorder is not None:
            tb = pd.DataFrame([[k , len(self.recorder[k]) , np.sum(self.recorder[k])] for k in self.recorder.keys()] ,
                                columns = ['keys' , 'num_calls', 'total_time'])
            tb['avg_time'] = tb['total_time'] / tb['num_calls']
            print(tb.sort_values(by=['total_time'],ascending=False))

class BigTimer:
    '''big timer to print out time'''
    def __init__(self , printer = print , name = None):
        self.printer = printer
        self.name = name if name else 'Whole Process'
    def __enter__(self):
        self.start_time = time.time()
    def __exit__(self, *args): 
        if not CONF.SILENT: self.printer(f'{self.name} Finished! Cost {self.time_str(time.time()-self.start_time)}')
    @staticmethod
    def time_str(seconds : float | int):
        time_str = ''
        if (hours := seconds // 3600) > 0: time_str += f'{hours:.0f} Hours '
        if (minutes := (seconds - hours * 3600) // 60) > 0: time_str += f'{minutes:.0f} Minutes '
        time_str += f'{seconds - hours * 3600 - minutes * 60:.1f} Seconds'
        return time_str

