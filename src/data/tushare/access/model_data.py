import torch
import numpy as np
import pandas as pd

from typing import Any , Callable , Literal

from .abstract_access import DateDataAccess

from .stock_info import INFO
from ....basic import PATH , CALENDAR , TradeDate
from ....func.singleton import singleton
    
@singleton
class RiskModelAccess(DateDataAccess):
    MAX_LEN = 2000
    DATA_TYPE_LIST = ['res' , 'exp']
    
    def loader_func(self , date , data_type):
        if data_type == 'res': 
            df = PATH.db_load('models' , 'tushare_cne5_res' , date)
        elif data_type == 'exp': 
            df = PATH.db_load('models' , 'tushare_cne5_exp' , date)
        else:
            raise KeyError(data_type)
        if df is not None: df = df.reset_index().assign(date = date)
        return df

    def get_res(self , date , field = None , drop_old = False):
        return self.get_df(date , 'res' , field , drop_old)
    
    def get_exp(self , date , field = None , drop_old = False):
        return self.get_df(date , 'exp' , field , drop_old)
    
    def get_exret(self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
                  mask = True , pivot = True):
        return self.get_specific_data(start_dt , end_dt , 'res' , 'resid' , prev = False , mask = mask , pivot = pivot)

RISK = RiskModelAccess()