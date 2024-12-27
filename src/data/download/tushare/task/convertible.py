import numpy as np
import pandas as pd

from typing import Any

from src.basic import IS_SERVER
from src.data.download.tushare.basic import pro , code_to_secid , CALENDAR , InfoFetcher , TushareFetcher , updatable , DateFetcher

class ConvertibleBasic(InfoFetcher):
    DB_KEY = 'cb_basic'
    UPDATE_FREQ = 'w'

    def get_data(self , date):
        df = self.iterate_fetch(pro.cb_basic , limit = 5000)
        return df

class ConvertibleDailyQuote(DateFetcher):
    START_DATE = 20200101 if IS_SERVER else 20241215
    DB_KEY = 'cb_day'
    def get_data(self , date : int):
        date_str = str(date)
        
        quote = self.iterate_fetch(pro.cb_daily , limit = 2000 , trade_date=date_str)
        if quote.empty: return quote
        quote = quote.rename(columns={'pre_close':'preclose','vol':'volume' , 'pre_settle':'presettle'})
        quote['amount'] = quote['amount'] * 10000
        
        return quote