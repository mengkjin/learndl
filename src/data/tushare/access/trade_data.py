import torch
import numpy as np
import pandas as pd

from typing import Any , Callable , Literal

from .abstract import DateDataAccess
from .calendar import CALENDAR
from .stock_info import INFO

from ..basic.trade_date import TradeDate
from ....basic import PATH
from ....func.singleton import singleton_threadsafe

@singleton_threadsafe
class TradeDataAccess(DateDataAccess):
    MAX_LEN = 2000
    DATA_TYPE_LIST = ['trd' , 'val' , 'mf']
    
    def loader_func(self , date , data_type):
        if data_type == 'trd': 
            df = PATH.load_target_file('trade_ts' , 'day' , date)
        elif data_type == 'val': 
            df = PATH.load_target_file('trade_ts' , 'day_val' , date)
        elif data_type == 'mf': 
            df = PATH.load_target_file('trade_ts' , 'day_moneyflow' , date)
        else:
            raise KeyError(data_type)
        if df is not None: df['date'] = date
        return df

    def get_val(self , date , cols = None , drop_old = False):
        return self.get_df(date , 'val' , cols , drop_old)
    
    def get_trd(self , date , cols = None , drop_old = False):
        return self.get_df(date , 'trd' , cols , drop_old)
    
    def get_mf(self , date , cols = None , drop_old = False):
        return self.get_df(date , 'mf' , cols , drop_old)

    def get_quotes(self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
                   cols : Literal['adjfactor','open','high','low','close','amount','volume',
                                  'vwap','status','limit','pctchange','preclose','turn_tt',
                                  'turn_fl','turn_fr'] | list , 
                   mask = False , pivot = False , drop_old = True , adj_price = True):
        qte = self.get_specific_data(start_dt , end_dt , 'trd' , cols , prev = False , 
                                     mask = mask , pivot = pivot , drop_old = drop_old)
        if adj_price:
            price_list = ['open','high','low','close','vwap']
            if (isinstance(cols , str) and cols in price_list) or any(col in cols for col in price_list):
                adj = self.get_adjfactor(start_dt , end_dt , pivot = pivot , drop_old = drop_old)
                if isinstance(cols , str):
                    qte *= adj
                else:
                    for col in cols:
                        if col in price_list: qte[col] *= adj
        return qte
    
    def get_adjfactor(self , start_dt : int | TradeDate , end_dt : int | TradeDate , pivot = False , drop_old = True):
        return self.get_specific_data(start_dt , end_dt , 'trd' , 'adjfactor' , prev = False , 
                                     mask = False , pivot = pivot , drop_old = drop_old)
    
    def get_val_data(self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
                     cols : Literal[
                         'turnover_rate','turnover_rate_f','volume_ratio','pe','pe_ttm','pb',
                         'ps','ps_ttm','dv_ratio','dv_ttm','total_share','float_share',
                         'free_share','total_mv','circ_mv'] | list , 
                     prev = True , mask = False , pivot = False , drop_old = True):
        return self.get_specific_data(start_dt , end_dt , 'val' , cols , 
                                      prev = prev , mask = mask , pivot = pivot , drop_old = drop_old)
    
    def get_mf_data(self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
                    cols : Literal[
                        'buy_sm_vol','buy_sm_amount','sell_sm_vol','sell_sm_amount','buy_md_vol',
                        'buy_md_amount','sell_md_vol','sell_md_amount','buy_lg_vol','buy_lg_amount',
                        'sell_lg_vol','sell_lg_amount','buy_elg_vol','buy_elg_amount','sell_elg_vol',
                        'sell_elg_amount','net_mf_vol','net_mf_amount'] | list , 
                    mask = False , pivot = False , drop_old = True):
        return self.get_specific_data(start_dt , end_dt , 'mf' , cols , 
                                      prev = False , mask = mask , pivot = pivot , drop_old = drop_old)
    
    def get_returns(self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
                    return_type : Literal['close' , 'vwap' , 'open' , 'intraday' , 'overnight'] = 'close' , 
                    pivot = True , mask = True):
        symbol = 'pctchange'
        if return_type == 'close':
            rets = self.get_quotes(start_dt , end_dt , symbol , mask = False , pivot = False) / 100
        elif return_type == 'intraday':
            rets = self.get_quotes(start_dt , end_dt , ['open' , 'close'] , mask = False , pivot = False)
            rets[symbol] = rets['close'] / rets['open'] - 1
        elif return_type == 'overnight':
            rets = self.get_quotes(start_dt , end_dt , ['open' , 'preclose'] , mask = False , pivot = False)
            rets[symbol] = rets['open'] / rets['preclose'] - 1
        elif return_type in ['vwap' , 'open']:
            price_symbol = 'vwap' if return_type == 'vwap' else 'open'
            rets = self.get_quotes(CALENDAR.td(start_dt , -1) , end_dt , ['adjfactor' , price_symbol] , mask = False , pivot = False).reset_index('date')
            rets['adjfactor'] = rets['adjfactor'].groupby('secid').ffill().fillna(1)
            rets['adjp'] = rets[price_symbol] * rets['adjfactor']
            rets[symbol] = rets['adjp'].pct_change()
            rets = rets[rets['date'] >= start_dt].reset_index().set_index(['date' , 'secid']).sort_index()
        else:
            raise KeyError(return_type)
        if pivot: rets = rets.pivot_table(symbol , 'date' , 'secid')
        rets = INFO.mask_list_dt(rets , mask)
        return rets
    
    def get_volumes(self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
                    volume_type : Literal['amount' , 'volume' , 'turn_tt' , 'turn_fl' , 'turn_fr'] = 'volume' , pivot = True , mask = True):
        volumes = self.get_quotes(start_dt , end_dt , volume_type , mask = mask , pivot = pivot)
        return volumes
    
    def get_turnovers(self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
                      turnover_type : Literal['tt' , 'fl' , 'fr'] = 'fr' , pivot = True , mask = True):
        symbol : Literal['turn_tt' , 'turn_fl' , 'turn_fr'] | Any = f'turn_{turnover_type}'
        turns = self.get_volumes(start_dt , end_dt , symbol , mask = mask , pivot = pivot) / 100
        return turns
    
    def get_mv(self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
               mv_type : Literal['circ_mv' , 'total_mv'] = 'circ_mv' , prev = True , pivot = False , drop_old = True):
        return self.get_val_data(start_dt , end_dt , mv_type , prev = prev , pivot = pivot , drop_old = drop_old)

    def get_market_return(self , start_dt : int | TradeDate , end_dt : int | TradeDate):
        '''
        dates = CALENDAR.td_within(CALENDAR.td(start_dt , -1) , end_dt)
        df_list = []
        for i in range(len(dates[1:])):
            trd = self.get_trd(dates[1 + i] , ['secid','pctchange']) 
            val = self.get_val(dates[0 + i] , ['secid','circ_mv'])
            ret = trd.merge(val , on = 'secid').assign(date = dates[1 + i])
            df_list.append(ret)
        df = pd.concat(df_list)
        '''

        rets = self.get_returns(start_dt , end_dt , return_type = 'close' , mask = False , pivot = False)
        circ = self.get_mv(start_dt , end_dt , mv_type = 'circ_mv' , pivot = False)
        rets = rets.merge(circ , on = ['date' , 'secid'])
        rets['mv_change'] = rets['pctchange'] * rets['circ_mv']
        mkt_ret = rets.groupby('date').apply(lambda x:(x['mv_change']).sum()/x['circ_mv'].sum(),include_groups=False).\
            rename('market').to_frame()
        return mkt_ret
    
TRADE = TradeDataAccess()