import numpy as np
import pandas as pd

from typing import Any
from src.basic import IS_SERVER
from src.data.download.tushare.basic import pro , code_to_secid , CALENDAR , InfoFetcher , TushareFetcher , updatable , DateFetcher

class FundInfo(InfoFetcher):
    DB_KEY = 'mutual_fund_info'
    UPDATE_FREQ = 'w'

    def get_data(self , date):
        renamer = {'ts_code' : 'fund_id'}
        df1 = self.iterate_fetch(pro.fund_basic , limit = 5000 , market='E')
        df2 = self.iterate_fetch(pro.fund_basic , limit = 5000 , market='O')

        df = pd.concat([df1 , df2]).rename(columns=renamer)
        for col in ['found_date' , 'due_date' , 'list_date' , 'issue_date' , 'delist_date']:
            df[col] = df[col].fillna(99991231).astype(int)
        df = df.reset_index(drop=True)
        return df

class FundPortfolioFetcher(TushareFetcher):
    START_DATE = 20070101
    DB_TYPE = 'fundport'
    UPDATE_FREQ = 'm'
    DB_SRC = 'holding_ts'
    DB_KEY = 'mutual_fund'
    DATA_FREQ = 'q'
    CONSIDER_FUTURE = False

    def update_dates(self):
        this_date , last_date , last_update_date = CALENDAR.today() , self.last_date() , self.last_update_date()

        update = updatable(this_date , last_update_date , self.UPDATE_FREQ)
        dates = CALENDAR.qe_trailing(this_date , n_past = 1 , another_date=last_date)

        if not update and len(dates) <= 1: dates = []
        return dates
    
    def get_data(self , date):
        renamer = {'ts_code' : 'fund_id'}
        df = self.iterate_fetch(pro.fund_portfolio , limit = 3000 , period = str(date) , max_fetch_times=500)
        df = code_to_secid(df.rename(columns=renamer) , code_col='symbol' , retain = True)
        for col in ['ann_date' , 'end_date']:
            df[col] = df[col].fillna(99991231).astype(int)
        df = df.reset_index(drop=True)
        return df

class ETFDailyQuote(DateFetcher):
    START_DATE = 20200101 if IS_SERVER else 20241215
    DB_KEY = 'etf_day'
    def get_data(self , date : int):
        date_str = str(date)
        
        quote = self.iterate_fetch(pro.fund_daily , limit = 2000 , trade_date=date_str)
        if quote.empty: return quote
        quote = quote.rename(columns={'pct_change':'pctchange','pre_close':'preclose','vol':'volume'})
        quote['volume'] = quote['volume'] * 1000
        quote['amount'] = quote['amount'] * 10000
        quote['vwap'] = np.where(quote['volume'] == 0 , quote['close'] , quote['amount'] / quote['volume'])
        return quote