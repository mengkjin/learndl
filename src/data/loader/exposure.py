import pandas as pd

from typing import Literal

from src.proj import TradeDate , DB
from src.proj.func import singleton
from src.data.util import INFO

from .access import DateDataAccess

@singleton
class ExposureAccess(DateDataAccess):
    MAX_LEN = 300
    DB_SRC = 'exposure'
    DB_KEYS = {'daily_risk' : 'daily_risk'}
    
    def data_loader(self , date , data_type):
        df : pd.DataFrame = DB.load(self.DB_SRC , self.DB_KEYS[data_type] , date , vb_level = 99 , use_alt = True)
        if not df.empty: 
            df = df[df['secid'].isin(INFO.get_secid(date))]
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