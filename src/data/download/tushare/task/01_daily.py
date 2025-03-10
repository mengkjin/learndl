import numpy as np
import pandas as pd

from src.data.download.tushare.basic import DateFetcher , pro , ts_code_to_secid

class DailyQuote(DateFetcher):
    '''Daily Quote'''
    DB_KEY = 'day'
    def get_data(self , date : int):
        date_str = str(date)
        adj = pro.query('adj_factor',  trade_date=date_str).rename(columns={'adj_factor':'adjfactor'})

        quote = pro.daily(trade_date=date_str).rename(columns={'pct_chg':'pctchange','pre_close':'preclose','vol':'volume'})
        quote['volume'] = quote['volume'] / 10. # to 10^3
        quote['vwap'] = np.where(quote['volume'] == 0 , quote['close'] , quote['amount'] / quote['volume'])

        shr = pro.daily_basic(trade_date=date_str).loc[:,['ts_code','trade_date' , 'total_share','float_share','free_share']]
        shr.loc[:,['total_share','float_share','free_share']] *= 1e4
        shr.loc[shr['free_share'].isna() , 'free_share'] = shr.loc[shr['free_share'].isna() , 'float_share']

        limit = pro.stk_limit(trade_date=date_str)
        if len(limit) == 0:
            limit = quote.loc[:,['ts_code' , 'trade_date' , 'close']].copy()
            limit['up_limit'] = (limit['close'] * 1.1).round(2)
            limit['down_limit'] = (limit['close'] * 0.9).round(2)
            limit = limit.drop(columns=['close'])

        susp = pro.suspend_d(suspend_type='S', trade_date=date_str)

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
    
class DailyValuation(DateFetcher):
    '''Daily Valuation'''
    DB_KEY = 'day_val'   
    def get_data(self , date : int):
        val = ts_code_to_secid(pro.daily_basic(trade_date=str(date)))
        val.loc[:,['total_share','float_share','free_share','total_mv','circ_mv']] *= 1e4

        val = ts_code_to_secid(val).set_index('secid').sort_index().reset_index().drop(columns='trade_date')
        return val
    
class DailyMoneyFlow(DateFetcher):
    '''Daily Money Flow'''
    START_DATE = 20100101
    DB_KEY = 'day_moneyflow'  
    def get_data(self , date : int):
        mf = ts_code_to_secid(pro.moneyflow(trade_date=str(date)))
        mf = ts_code_to_secid(mf).set_index('secid').sort_index().reset_index().drop(columns='trade_date')
        return mf
    
class DailyLimit(DateFetcher):
    '''Daily Price Limit Infomation'''
    START_DATE = 20070101
    DB_KEY = 'day_limit'       
    def get_data(self , date : int):
        lmt = ts_code_to_secid(pro.stk_limit(trade_date=str(date)))
        lmt = ts_code_to_secid(lmt).set_index('secid').sort_index().reset_index().drop(columns='trade_date')
        return lmt
"""
class DailyOpenAuction(DateFetcher):
    '''Daily Open Auction Infomation'''
    START_DATE = 20070101
    DB_KEY = 'day_open_auction'       
    def get_data(self , date : int):
        auc = code_to_secid(pro.stk_auction_o(trade_date=str(date)))
        auc = code_to_secid(auc).set_index('secid').sort_index().reset_index().drop(columns='trade_date')
        return auc
    
class DailyCloseAuction(DateFetcher):
    '''Daily Close Auction Infomation'''
    START_DATE = 20070101
    DB_KEY = 'day_close_auction'       
    def get_data(self , date : int):
        auc = code_to_secid(pro.stk_auction_c(trade_date=str(date)))
        auc = code_to_secid(auc).set_index('secid').sort_index().reset_index().drop(columns='trade_date')
        return auc
"""
