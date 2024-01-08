import pandas as pd
import numpy as np
from datetime import date , datetime

def this_day(astype = str):
    return astype(date.today().strftime('%Y%m%d'))

def date_offset(date , offset = 0 , astype = str):
    if isinstance(date , (np.ndarray,pd.Index,pd.Series,list,tuple,np.ndarray)):
        is_scalar = False
        new_date = pd.DatetimeIndex(np.array(date).astype(str))
    else:
        is_scalar = True
        new_date = pd.DatetimeIndex([str(date)])
    if offset == 0:
        new_date = new_date.strftime('%Y%m%d')
    else:
        new_date = (new_date + pd.DateOffset(offset)).strftime('%Y%m%d')
    new_date = new_date.astype(astype)
    return new_date[0] if is_scalar else new_date.values

def date_seg(start_dt , end_dt , freq='Q' , astype = str):
    dt_list = pd.date_range(str(start_dt) , str(end_dt) , freq=freq).strftime('%Y%m%d').astype(int)
    dt_starts = [date_offset(start_dt) , *date_offset(dt_list[:-1],1)]
    dt_ends = [*dt_list[:-1] , date_offset(end_dt)]
    return [(astype(s),astype(e)) for s,e in zip(dt_starts , dt_ends)]
