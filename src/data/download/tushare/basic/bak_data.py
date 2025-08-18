import pandas as pd
import numpy as np
from typing import Literal

from src.basic import PATH , CALENDAR
from .func import ts_code_to_secid

class TSBackUpDataTransform():
    '''Daily Quote'''
    BACKUP_DIR = PATH.main.joinpath('bak_data')
    BACKUP_RECORD_DIR = PATH.main.joinpath('bak_data_record')
    REQUIRED_KEYS = ['adj_factor' , 'daily' , 'daily_basic' , 'moneyflow' , 'stk_limit']
    
    BACKUP_DIR.mkdir(parents=True , exist_ok=True)
    BACKUP_RECORD_DIR.mkdir(parents=True , exist_ok=True)

    def get_bak_data(self , date : int , key : Literal['adj_factor' , 'daily' , 'daily_basic' , 'moneyflow' , 'stk_limit']):
        date_str = str(date)
        record_file = self.BACKUP_DIR.joinpath(f'{key}_{date_str}.csv')
        df = pd.read_csv(record_file)
        df.columns = df.columns.str.lower()
        return df

    def day(self , date : int):
        adj = self.get_bak_data(date , 'adj_factor').rename(columns={'adj_factor':'adjfactor'})

        quote = self.get_bak_data(date , 'daily').rename(columns={'pct_change':'pctchange','pre_close':'preclose','vol':'volume'})
        quote['volume'] = quote['volume'] / 10. # to 10^3
        quote['vwap'] = np.where(quote['volume'] == 0 , quote['close'] , quote['amount'] / quote['volume'])

        shr = self.get_bak_data(date , 'daily_basic').loc[:,['ts_code','trade_date' , 'total_share','float_share','free_share']]
        shr.loc[:,['total_share','float_share','free_share']] *= 1e4
        shr.loc[shr['free_share'].isna() , 'free_share'] = shr.loc[shr['free_share'].isna() , 'float_share']

        limit = self.get_bak_data(date , 'stk_limit')
        if len(limit) == 0:
            limit = quote.loc[:,['ts_code' , 'trade_date' , 'close']].copy()
            limit['up_limit'] = (limit['close'] * 1.1).round(2)
            limit['down_limit'] = (limit['close'] * 0.9).round(2)
            limit = limit.drop(columns=['close'])

        susp = pd.DataFrame(columns=['ts_code' , 'trade_date'])

        mutual_col = ['ts_code' , 'trade_date']
        trade = quote.merge(adj,on=mutual_col,how='left').\
            merge(limit,on=mutual_col,how='left').\
            merge(shr,on=mutual_col,how='left')
        trade['status'] = 1.0 * ~trade['ts_code'].isin(susp['ts_code']).fillna(0)
        trade['limit'] = 1.0 * (trade['close'] >= trade['up_limit']).fillna(0) - 1.0 * (trade['close'] <= trade['down_limit']).fillna(0)
        trade['turn_tt'] = (trade['volume'] / trade['total_share'] * 1e5).fillna(0)
        trade['turn_fl'] = (trade['volume'] / trade['float_share'] * 1e5).fillna(0)
        trade['turn_fr'] = (trade['volume'] / trade['free_share'] * 1e5).fillna(0)

        trade = ts_code_to_secid(trade).set_index('secid').sort_index().reset_index().loc[
            :,['secid', 'adjfactor', 'open', 'high', 'low', 'close', 'amount','volume', 'vwap', 
            'status', 'limit', 'pctchange', 'preclose', 'turn_tt','turn_fl', 'turn_fr']]
        return trade
    
    def day_val(self , date : int):
        val = self.get_bak_data(date , 'daily_basic').loc[:,['ts_code','trade_date' , 'total_share','float_share','free_share','total_mv','circ_mv']]
        val.loc[:,['total_share','float_share','free_share','total_mv','circ_mv']] *= 1e4
        val = ts_code_to_secid(val).set_index('secid').sort_index().reset_index().drop(columns='trade_date')
        return val
    
    def day_moneyflow(self , date : int):
        mf = self.get_bak_data(date , 'moneyflow')
        mf = ts_code_to_secid(mf).set_index('secid').sort_index().reset_index().drop(columns='trade_date')
        return mf
    
    def day_limit(self , date : int):
        lmt = self.get_bak_data(date , 'stk_limit')
        lmt = ts_code_to_secid(lmt).set_index('secid').sort_index().reset_index().drop(columns='trade_date')
        return lmt
    
    def transform(self , date : int):
        db_src = 'trade_ts'
        day = self.day(date)
        day_val = self.day_val(date)
        day_moneyflow = self.day_moneyflow(date)
        day_limit = self.day_limit(date)
        self.path_record(date).touch()
        PATH.db_save(day , db_src , 'day' , date)
        PATH.db_save(day_val , db_src , 'day_val' , date)
        PATH.db_save(day_moneyflow , db_src , 'day_moneyflow' , date)
        PATH.db_save(day_limit , db_src , 'day_limit' , date)

    def clear_day(self , date : int):
        if self.path_record(date).exists():
            db_src = 'trade_ts'
            for db_key in ['day' , 'day_val' , 'day_moneyflow' , 'day_limit']:
                path = PATH.db_path(db_src , db_key , date)
                new_path = path.with_name(f'.{path.name}.bak')
                if path.exists():
                    if not new_path.exists():
                        path.rename(new_path)
                    else:
                        path.unlink()
            self.path_record(date).unlink()

    def path_record(self , date : int):
        return self.BACKUP_RECORD_DIR.joinpath(f'{date}.backed')

    def get_baked_dates(self):
        return [int(file.stem) for file in self.BACKUP_RECORD_DIR.glob('*.backed')]

    def get_bakable_dates(self):
        dates = CALENDAR.td_within(start_dt = CALENDAR.td(CALENDAR.updated() , 1) , end_dt = CALENDAR.update_to())
        dates = [date for date in dates if self.bakable_date(date)]
        return dates
    
    def bakable_date(self , date : int):
        paths = [self.BACKUP_DIR.joinpath(f'{key}_{date}.csv') for key in self.REQUIRED_KEYS]
        return all([path.exists() for path in paths])
    
    @classmethod
    def update(cls):
        transformer = cls()
        for date in transformer.get_bakable_dates():
            transformer.transform(date)

    @classmethod
    def clear(cls):
        transformer = cls()
        for date in transformer.get_baked_dates():
            transformer.clear_day(date)