import numpy as np
import pandas as pd

from typing import Any

from src.data.download.tushare.basic import pro , code_to_secid , CALENDAR , InfoFetcher , TushareFetcher , updatable , quarter_ends

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
        dates = quarter_ends(this_date , last_date , trailing_quarters = 1)  
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