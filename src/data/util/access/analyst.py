import torch
import numpy as np
import pandas as pd

from typing import Any , Callable , Literal

from src.basic import PATH , CALENDAR , TradeDate
from src.func.singleton import singleton

from .abstract_access import DateDataAccess
from ..basic import INFO

@singleton
class AnalystDataAccess(DateDataAccess):
    MAX_LEN = 2000
    DATA_TYPE_LIST = ['report']

    def data_loader(self , date , data_type):
        db_key_dict = {'report' : 'report'}
        if data_type in db_key_dict: 
            df = PATH.db_load('analyst_ts' , db_key_dict[data_type] , date , verbose = False)
        else:
            raise KeyError(data_type)
        return df

    def get_report(self , date , field = None):
        return self.get(date , 'report' , field)

    def get_trailing_reports(self , date : int , n_month : int = 3 , lag_month : int = 0, latest = False , **filter_kwargs):
        d0 = CALENDAR.cd(date , -30 * (n_month + lag_month))
        d1 = CALENDAR.cd(date , -30 * lag_month) 
        dates = CALENDAR.cd_within(d0 , d1)
        reports : list[pd.DataFrame] = []
        for date in dates:
            df = self.get_report(date)
            if df.empty: continue
            for key , value in filter_kwargs.items():
                df = df[df[key] == value]
            reports.append(df)
        df = pd.concat(reports).reset_index(drop = True)
        df['report_date'] = df['report_date'].astype(int)
        if latest:
            df = df.sort_values(['secid' , 'report_date']).groupby(['secid' , 'org_name' , 'quarter']).last().reset_index(drop = False)
        return df
    
    @staticmethod
    def weighted_val(df : pd.DataFrame , end_date : int , col : str , half_life : int = 180):
        df = df.assign(_w = np.exp(-np.log(2) * CALENDAR.cd_diff_array(end_date , df['report_date']) / half_life))
        return df.groupby('secid').apply(lambda x:(x[col] * x['_w']).sum() / x['_w'].sum(),include_groups=False)
    
    @staticmethod
    def val_multiplier(val : str):
        return 1e4 if val in ['sales' , 'op' , 'np' , 'tp'] else 1
    
    def get_val_est(self , date : int , year : int , val : Literal['sales' , 'np' , 'tp' , 'op' , 'eps' , 'roe'] , 
                    n_month : int = 12 , lag_month : int = 0):
        col = {'sales' : 'op_rt' , 'np' : 'np' , 'tp' : 'tp' , 'op' : 'op_pr' , 'eps' : 'eps' , 'roe' : 'roe'}[val]
        multiplier = self.val_multiplier(val)
        end_date = CALENDAR.cd(date , -30 * lag_month)
        df = self.get_trailing_reports(end_date , n_month , lag_month , latest = True , quarter = f'{year}Q4')
        est = self.weighted_val(df , end_date , col) * multiplier
        return est
    
    def get_val_ftm(self , date : int , val : Literal['sales' , 'np' , 'tp' , 'op' , 'eps' , 'roe'] , n_month : int = 12 , lag_month : int = 0):
        month = date // 100 % 100
        year = date // 10000
        val0 = self.get_val_est(date , year , val , n_month , lag_month)
        val1 = self.get_val_est(date , year + 1 , val , n_month , lag_month)
        ftm = ((12 - month) * val0 + month * val1) / 12
        return ftm
    
    def target_price(self , date : int , n_month : int = 12 , lag_month : int = 0):
        end_date = CALENDAR.cd(date , -30 * lag_month)
        df = self.get_trailing_reports(end_date , n_month , lag_month , latest = True)
        df = df[df['max_price'].notna() | df['min_price'].notna()]
        df = df[df['quarter'] == f'{date // 10000}Q4']
        df['target_price'] = df.loc[:,['max_price' , 'min_price']].mean(axis = 1)
        v = df.groupby('secid')['target_price'].mean()
        return v
        
ANALYST = AnalystDataAccess()