import pandas as pd
import numpy as np

from collections.abc import Iterable
from datetime import date , datetime , timedelta
from typing import Any , Callable

def today(offset = 0 , astype : Any = int):
    d = datetime.today() + timedelta(days=offset)
    return astype(d.strftime('%Y%m%d'))

def this_day(astype : Any = str):
    return astype(date.today().strftime('%Y%m%d'))

def date_offset(date : Any , offset : int = 0 , astype = int):
    iterable_input = isinstance(date , Iterable)
    date = pd.DatetimeIndex(np.array(date).astype(str) if iterable_input else [str(date)])
    dseries : pd.DatetimeIndex = (date + pd.DateOffset(n=offset))
    new_date = dseries.strftime('%Y%m%d').astype(astype).to_numpy()
    return new_date if iterable_input else new_date[0]

def date_diff(date1 : int | str , date2 : int | str):
    return (datetime.strptime(str(date1), '%Y%m%d') - datetime.strptime(str(date2), '%Y%m%d')).days

def date_seg(start_dt , end_dt , freq='QE' , astype : Any = int):
    if start_dt >= end_dt: return []
    dt_list = pd.date_range(str(start_dt) , str(end_dt) , freq=freq).strftime('%Y%m%d').astype(int)
    dt_starts = [date_offset(start_dt) , *date_offset(dt_list[:-1],1)]
    dt_ends = [*dt_list[:-1] , date_offset(end_dt)]
    return [(astype(s),astype(e)) for s,e in zip(dt_starts , dt_ends)]