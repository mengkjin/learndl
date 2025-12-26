import pandas as pd

from typing import Literal

from src.basic import TradeDate , DB
from src.math.singleton import singleton
from src.data.util import INFO

from .access import DateDataAccess

@singleton
class ExposureAccess(DateDataAccess):
    MAX_LEN = 300
    DATA_TYPE_LIST = ['daily_risk']
    
    def data_loader(self , date , data_type):
        if data_type in self.DATA_TYPE_LIST: 
            df = DB.load('exposure' , data_type , date , vb_level = 99 , use_alt = True)
            if not df.empty: 
                secid = INFO.get_secid(date) # noqa
                df = df.query('secid in @secid')
        else:
            raise KeyError(data_type)
        return df
    
    def get_daily_risk(self , date):
        return self.get(date , 'daily_risk')

    def get_risks(
        self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
        field : Literal['true_range' , 'turnover' , 'large_buy_pdev' , 'small_buy_pct' ,
        'sqrt_avg_size' , 'open_close_pct' , 'ret_volatility' , 'ret_skewness'] | str | list , prev = False ,
        mask = False , pivot = False , **kwargs
    ) -> pd.DataFrame:
        qte = self.get_specific_data(start_dt , end_dt , 'daily_risk' , field = field , prev = prev , 
                                     mask = mask , pivot = False , drop_old = True)
        
        if pivot:
            qte = qte.pivot_table(field , 'date' , 'secid')
        return qte
        
EXPO= ExposureAccess()