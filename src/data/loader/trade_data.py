import pandas as pd
from typing import Any , Literal

from src.basic import CALENDAR , TradeDate , DB
from src.func.singleton import singleton
from src.data.util import INFO

from .access import DateDataAccess

db_key_dict = {'trd' : 'day' , 'val' : 'day_val' , 'mf' : 'day_moneyflow' , 'limit' : 'day_limit'}
@singleton
class TradeDataAccess(DateDataAccess):
    MAX_LEN = 2000
    DATA_TYPE_LIST = ['trd' , 'val' , 'mf' , 'limit']
    
    def data_loader(self , date , data_type):
        if data_type in db_key_dict: 
            df = DB.db_load('trade_ts' , db_key_dict[data_type] , date , verbose = False)
        else:
            raise KeyError(data_type)
        return df
    
    def latest_date(self , data_type : str , date : int | None = None):
        if data_type in db_key_dict:
            dates = DB.db_dates('trade_ts' , db_key_dict[data_type])
            if date: 
                dates = dates[dates <= date]
            return dates.max()
        else:
            raise KeyError(data_type)

    def get_val(self , date , field = None):
        return self.get(date , 'val' , field)
    
    def get_trd(self , date , field = None):
        return self.get(date , 'trd' , field)
    
    def get_mf(self , date , field = None):
        return self.get(date , 'mf' , field)
    
    def get_limit(self , date , field = None):
        return self.get(date , 'limit' , field)

    def get_quotes(
        self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
        field : Literal['adjfactor','open','high','low','close','amount','volume',
                        'vwap','status','limit','pctchange','preclose','turn_tt',
                        'turn_fl','turn_fr'] | list , 
        mask = False , pivot = False , drop_old = False , adj_price = True
    ) -> pd.DataFrame:
        qte = self.get_specific_data(start_dt , end_dt , 'trd' , field , prev = False , 
                                     mask = mask , pivot = False , drop_old = drop_old)
        if adj_price:
            prices = [p for p in ([field] if isinstance(field , str) else field) if p in ['open','high','low','close','vwap','preclose']]
            if prices:
                adj = self.get_adjfactor(start_dt , end_dt , pivot = False , drop_old = drop_old)['adjfactor'].fillna(1)
                for p in prices:
                    qte[p] = qte[p] * adj
        if pivot: 
            qte = qte.pivot_table(field , 'date' , 'secid')
        return qte
    
    def get_adjfactor(
        self , start_dt : int | TradeDate , end_dt : int | TradeDate , pivot = False , drop_old = False
    ) -> pd.DataFrame:
        return self.get_specific_data(start_dt , end_dt , 'trd' , 'adjfactor' , prev = False , 
                                      mask = False , pivot = pivot , drop_old = drop_old)
    
    def get_val_data(
        self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
        field : Literal[
            'turnover_rate','turnover_rate_f','volume_ratio','pe','pe_ttm','pb',
            'ps','ps_ttm','dv_ratio','dv_ttm','total_share','float_share',
            'free_share','total_mv','circ_mv'] | list , 
        prev = True , mask = False , pivot = False , drop_old = False
    ) -> pd.DataFrame:
        return self.get_specific_data(start_dt , end_dt , 'val' , field , 
                                      prev = prev , mask = mask , pivot = pivot , drop_old = drop_old)
    
    def get_mf_data(
        self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
        field : Literal[
            'buy_sm_vol','buy_sm_amount','sell_sm_vol','sell_sm_amount','buy_md_vol',
            'buy_md_amount','sell_md_vol','sell_md_amount','buy_lg_vol','buy_lg_amount',
            'sell_lg_vol','sell_lg_amount','buy_elg_vol','buy_elg_amount','sell_elg_vol',
            'sell_elg_amount','net_mf_vol','net_mf_amount'] | list , 
        mask = False , pivot = False , drop_old = False
    ) -> pd.DataFrame:
        return self.get_specific_data(start_dt , end_dt , 'mf' , field , 
                                      prev = False , mask = mask , pivot = pivot , drop_old = drop_old)
    
    def get_limit_data(
        self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
        field : Literal['up_limit','down_limit','pre_close',] | list , 
        mask = False , pivot = False , drop_old = False
    ) -> pd.DataFrame:
        return self.get_specific_data(start_dt , end_dt , 'limit' , field , 
                                      prev = False , mask = mask , pivot = pivot , drop_old = drop_old)
    
    def get_returns(
        self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
        return_type : Literal['close' , 'vwap' , 'open' , 'intraday' , 'overnight'] = 'close' , 
        pivot = True , mask = True
    ) -> pd.DataFrame:
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
        if pivot: 
            rets = rets.pivot_table(symbol , 'date' , 'secid')
        rets = INFO.mask_list_dt(rets , mask)
        return rets
    
    def get_volumes(
        self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
        volume_type : Literal['amount' , 'volume' , 'turn_tt' , 'turn_fl' , 'turn_fr'] = 'volume' , pivot = True , mask = True
    ) -> pd.DataFrame:
        volumes = self.get_quotes(start_dt , end_dt , volume_type , mask = mask , pivot = pivot)
        return volumes
    
    def get_turnovers(
        self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
        turnover_type : Literal['tt' , 'fl' , 'fr'] = 'fr' , pivot = True , mask = True
    ) -> pd.DataFrame:
        symbol : Literal['turn_tt' , 'turn_fl' , 'turn_fr'] | Any = f'turn_{turnover_type}'
        turns = self.get_volumes(start_dt , end_dt , symbol , mask = mask , pivot = pivot) / 100
        return turns
    
    def get_mv(
        self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
        mv_type : Literal['circ_mv' , 'total_mv'] = 'circ_mv' , prev = True , pivot = False , drop_old = False
    ) -> pd.DataFrame:
        return self.get_val_data(start_dt , end_dt , mv_type , prev = prev , pivot = pivot , drop_old = drop_old)

    def get_market_return(
        self , start_dt : int | TradeDate , end_dt : int | TradeDate , 
        return_type : Literal['close' , 'vwap' , 'open' , 'intraday' , 'overnight'] = 'close'
    ) -> pd.DataFrame:
        rets = self.get_returns(start_dt , end_dt , return_type = return_type , mask = False , pivot = False)
        circ = self.get_mv(start_dt , end_dt , mv_type = 'circ_mv' , pivot = False)
        rets = rets.merge(circ , on = ['date' , 'secid'])
        rets['mv_change'] = rets['pctchange'] * rets['circ_mv']
        mkt_ret : pd.Series | Any = rets.groupby('date').apply(lambda x,**kwg:(x['mv_change']).sum()/x['circ_mv'].sum() , include_groups = False)
        return mkt_ret.rename('market').to_frame()
    
    def get_market_amount(
        self , start_dt : int | TradeDate , end_dt : int | TradeDate
    ) -> pd.DataFrame:
        amount = self.get_volumes(start_dt , end_dt , volume_type = 'amount' , mask = False , pivot = False)
        mkt_amt : pd.Series | Any = amount.groupby('date')['amount'].sum()
        return mkt_amt.rename('market').to_frame()
        
TRADE = TradeDataAccess()