import time
import pandas as pd
import numpy as np

from datetime import date , datetime , timedelta
from typing import Any

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

def date_seg(start_dt , end_dt , freq='Q' , astype : Any = int):
    dt_list = pd.date_range(str(start_dt) , str(end_dt) , freq=freq).strftime('%Y%m%d').astype(int)
    dt_starts = [date_offset(start_dt) , *date_offset(dt_list[:-1],1)]
    dt_ends = [*dt_list[:-1] , date_offset(end_dt)]
    return [(astype(s),astype(e)) for s,e in zip(dt_starts , dt_ends)]

class Timer:
    '''simple timer to print out time'''
    def __init__(self , *args): self.key = '/'.join(args)
    def __enter__(self):
        self.start_time = time.time()
        print(self.key , '...', end='')
    def __exit__(self, type, value, trace): print(f'... cost {time.time()-self.start_time:.2f} secs')