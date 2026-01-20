# do not use relative import in this file because it will be running in top-level directory
import numpy as np
import pandas as pd

from src.proj import MACHINE
from src.data.download.tushare.basic import InfoFetcher , DayFetcher

class FuturesCalendar(InfoFetcher):
    """futures calendar"""
    DB_KEY = 'futures_calendar'
    def get_data(self , date):
        markets = ['CFFEX-中金所' , 'DCE-大商所' , 'CZCE-郑商所' , 'SHFE-上期所' , 'INE-上海国际能源交易中心' , 'GFEX-广州期货交易所']
        dfs : list[pd.DataFrame] = []
        for market in markets:
            df = self.iterate_fetch(self.pro.trade_cal , limit = 5000 , exchange=market.split('-')[0])
            dfs.append(df)

        df = pd.concat([d for d in dfs if not d.empty])
        return df

class FuturesBasic(InfoFetcher):
    """futures basic infomation"""
    DB_KEY = 'futures_basic'
    UPDATE_FREQ = 'w'

    def get_data(self , date):
        markets = ['CFFEX-中金所' , 'DCE-大商所' , 'CZCE-郑商所' , 'SHFE-上期所' , 'INE-上海国际能源交易中心' , 'GFEX-广州期货交易所']
        dfs : list[pd.DataFrame] = []
        for market in markets:
            df = self.iterate_fetch(self.pro.fut_basic , limit = 5000 , exchange=market.split('-')[0])
            dfs.append(df)

        df = pd.concat([d for d in dfs if not d.empty])
        return df

class FuturesDailyQuote(DayFetcher):
    """futures daily quote"""
    START_DATE = 20180101 if MACHINE.updatable else 20241215
    DB_KEY = 'fut_day'
    def get_data(self , date : int):
        date_str = str(date)
        
        quote = self.iterate_fetch(self.pro.fut_daily , limit = 2000 , trade_date=date_str)
        if quote.empty: 
            return quote
        quote = quote.rename(columns={'pre_close':'preclose','vol':'volume' , 'pre_settle':'presettle'})
        quote['amount'] = quote['amount'] * 10000
        quote['vwap'] = np.where(quote['volume'] == 0 , quote['close'] , quote['amount'] / quote['volume'])

        return quote