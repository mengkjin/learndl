"""
Tushare API utilities: token management, rate-limiting lock, and code normalisation.

The module-level ``TS`` singleton is the sole interface to the Tushare API.
All Tushare fetchers should call ``TS.api`` for the Tushare Pro API handle and
use ``TS.lock`` (via ``TS.locked()``) to serialise API calls.
"""
import threading
import functools

import tushare as ts
import numpy as np
import pandas as pd

from typing import Literal , Callable , TypeVar
from src.proj import CALENDAR , MACHINE
from src.data.util import secid_adjust

_server_down = False
T = TypeVar('T')

class TushareUtils:
    """
    Singleton utility class for the Tushare Pro API.

    Provides lazy-loaded API handle, thread-safe lock, and helper methods:
    - ``updatable(last_date, freq)``  — check whether an update is due
    - ``dates_to_update(last_date, freq)`` — list of dates needing an update
    - ``code_to_secid(df)``           — normalise ts_code → integer secid
    """
    """
    parameters for tushare
        token: token for tushare
        pro: tushare pro api
        server_down: whether the tushare server is down
    """

    @property
    def token(self):
        return MACHINE.secret['accounts']['tushare']['token']

    @property
    def api(self):
        if not hasattr(self , '_pro'):
            self._pro = ts.pro_api(self.token)
        return self._pro

    @property
    def lock(self) -> threading.Lock:
        if not hasattr(self , '_lock'):
            self._lock = threading.Lock()
        return self._lock

    def locked(self , func : Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                return func(*args, **kwargs)
        return wrapper

    @property
    def server_down(self):
        if not hasattr(self , '_server_down'):
            self._server_down = _server_down
        return self._server_down

    @server_down.setter
    def server_down(self , value: bool):
        self._server_down = value
        
    def get_api(self):
        return ts.pro_api(self.token)

    @classmethod
    def code_to_secid(cls ,df : pd.DataFrame , code_col = 'ts_code' , drop_old = True , ashare = True):
        """switch old symbol into secid"""
        if code_col not in df.columns.values: 
            return df
        if ashare: 
            valid = df[code_col].astype(str).str.split('.').str[-1].str.lower().isin(['sh' , 'sz' , 'bj'])
        else:
            valid = None
        df = secid_adjust(df , code_cols = code_col , drop_old = drop_old)
        if valid is not None: 
            df['secid'] = df['secid'].where(valid , -1)
        return df

    @classmethod
    def updatable(cls , last_date : int , freq : Literal['d' , 'w' , 'm'] , update_to : int | None = None):
        """check if the date is updatable given last date and frequency"""
        update_to = update_to or CALENDAR.update_to()
        if freq == 'd':
            return update_to > last_date
        elif freq == 'w':
            return update_to > CALENDAR.cd(last_date , 6)
        elif freq == 'm':
            return ((update_to // 100) % 100) != ((last_date // 100) % 100)
    
    @classmethod
    def dates_to_update(cls , last_date : int , freq : Literal['d' , 'w' , 'm'] , update_to : int | None = None):
        """get dates to update given last date and frequency"""
        update_to = update_to or CALENDAR.update_to()
        if last_date >= update_to: 
            return np.array([] , dtype=int)
        if freq == 'd':
            date_list = pd.date_range(str(last_date) , str(update_to)).strftime('%Y%m%d').to_numpy(int)[1:]
        elif freq == 'w':
            date_list = pd.date_range(str(last_date) , str(update_to)).strftime('%Y%m%d').to_numpy(int)[::7][1:]
        elif freq == 'm':
            date_list = pd.date_range(str(last_date) , str(update_to) , freq='ME').strftime('%Y%m%d').to_numpy(int)
            if last_date in date_list: 
                date_list = date_list[1:]
        return np.sort(date_list)

    @classmethod
    def adjust_precision(cls , df : pd.DataFrame , tol : float = 1e-8 , dtype_float = np.float32 , dtype_int = np.int64):
        """adjust precision for df columns"""
        for col in df.columns:
            if np.issubdtype(df[col].to_numpy().dtype , np.floating): 
                df[col] = df[col].astype(dtype_float)
                df[col] *= (df[col].abs() > tol)
            if np.issubdtype(df[col].to_numpy().dtype , np.integer): 
                df[col] = df[col].astype(dtype_int)
        return df

    @classmethod
    def get_func_name(cls , func : Callable):
        if isinstance(func , functools.partial):
            groups = [cls.get_func_name(func.func) , '.'.join(func.args) , '.'.join(f'{k}={v}' for k,v in func.keywords.items())]
            return '.'.join([s for s in groups if s])
        else:
            return func.__name__

TS = TushareUtils()