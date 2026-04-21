"""
Daily market data access singleton for the Chinese A-share market.

Provides adjusted OHLCV, valuation, moneyflow, and limit data from the
``trade_ts`` database.  Exported as the ``TRADE`` singleton.
"""
import pandas as pd
import numpy as np
from typing import Any , Literal

from src.proj import CALENDAR , TradeDate , DB , singleton
from src.data.util import INFO

from .access import DateDataAccess

@singleton
class TradeDataAccess(DateDataAccess):
    """
    Singleton data access object for daily Chinese A-share market data.

    Covers four database tables (``DB_KEYS``):
    - ``trd``   : OHLCV, adjfactor, pctchange, preclose, turnover
    - ``val``   : valuation multiples (PE/PB/PS), market cap, shares outstanding
    - ``mf``    : money flow (large/medium/small buy-sell volumes and amounts)
    - ``limit`` : daily limit-up / limit-down prices

    Access via the module-level ``TRADE`` singleton.
    """
    MAX_LEN = 5000
    DB_SRC = 'trade_ts'
    DB_KEYS = {'trd' : 'day' , 'val' : 'day_val' , 'mf' : 'day_moneyflow' , 'limit' : 'day_limit'}
    
    def data_loader(self , date , data_type) -> pd.DataFrame:
        """Load a single-date slice for ``data_type`` from the database."""
        df = DB.load(self.DB_SRC , self.DB_KEYS[data_type] , date , vb_level = 'never')
        return df

    def closest_date(self , data_type : str , date : int | None = None) -> int:
        """Return the most recent date available in the DB for ``data_type``, optionally capped at ``date``."""
        dates = DB.dates(self.DB_SRC , self.DB_KEYS[data_type])
        if date: 
            dates = dates[dates <= date]
        return dates.max() if len(dates) else -1

    def db_loads_callback(self , df : pd.DataFrame , db_src : str , db_key : str):
        """when DB.loads is called with fill_datavendor=True , this function will be called to add the data to the collection"""
        if df.empty or db_src != self.DB_SRC:
            return
        for data_type , data_key  in self.DB_KEYS.items():
            if data_key == db_key:
                self.collections[data_type].add_long_frame(df.set_index(self.DATE_KEY))

    def loads(self , dates: list[int | TradeDate] | np.ndarray | int | None , data_type : str):
        dates = CALENDAR.td_array(dates)
        if len(dates) == 0:
            return
        df = DB.loads(self.DB_SRC , self.DB_KEYS[data_type] , dates , vb_level = 'never')
        self.collections[data_type].add_long_frame(df.set_index(self.DATE_KEY))

    def get_val(self , date , field = None) -> pd.DataFrame:
        """Return valuation data for a single ``date`` (convenience wrapper)."""
        return self.get(date , 'val' , field)

    def get_trd(self , date , field = None) -> pd.DataFrame:
        """Return trade (OHLCV) data for a single ``date`` (convenience wrapper)."""
        return self.get(date , 'trd' , field)

    def get_mf(self , date , field = None) -> pd.DataFrame:
        """Return money-flow data for a single ``date`` (convenience wrapper)."""
        return self.get(date , 'mf' , field)

    def get_limit(self , date , field = None) -> pd.DataFrame:
        """Return limit-price data for a single ``date`` (convenience wrapper)."""
        return self.get(date , 'limit' , field)

    def get_quotes(
        self , start : int | TradeDate , end : int | TradeDate ,
        field : Literal['adjfactor','open','high','low','close','amount','volume',
                        'vwap','status','limit','pctchange','preclose','turn_tt',
                        'turn_fl','turn_fr'] | list ,
        mask = False , pivot = False , drop_old = False , adj_price = True
    ) -> pd.DataFrame:
        """
        Return daily quote data for the requested fields.

        When ``adj_price=True`` (default), price columns (open/high/low/close/vwap/preclose)
        are multiplied by the rolling adjfactor.  Use ``mask=True`` to apply the
        listing-date mask.  Use ``pivot=True`` for a (date × secid) wide frame.
        """
        qte = self.get_specific_data(start , end , 'trd' , field , prev = False , 
                                     mask = mask , pivot = False , drop_old = drop_old)
        if adj_price:
            prices = [p for p in ([field] if isinstance(field , str) else field) if p in ['open','high','low','close','vwap','preclose']]
            if prices:
                adj = self.get_adjfactor(start , end , pivot = False , drop_old = drop_old)['adjfactor'].fillna(1)
                for p in prices:
                    qte[p] = qte[p] * adj
        if pivot: 
            qte = qte.pivot_table(field , 'date' , 'secid')
        return qte
    
    def get_adjfactor(
        self , start : int | TradeDate , end : int | TradeDate , pivot = False , drop_old = False
    ) -> pd.DataFrame:
        """Return the daily adjustment factor series for ``[start, end]``."""
        return self.get_specific_data(start , end , 'trd' , 'adjfactor' , prev = False , 
                                      mask = False , pivot = pivot , drop_old = drop_old)
    
    def get_val_data(
        self , start : int | TradeDate , end : int | TradeDate ,
        field : Literal[
            'turnover_rate','turnover_rate_f','volume_ratio','pe','pe_ttm','pb',
            'ps','ps_ttm','dv_ratio','dv_ttm','total_share','float_share',
            'free_share','total_mv','circ_mv'] | list , 
        prev = True , mask = False , pivot = False , drop_old = False
    ) -> pd.DataFrame:
        """Return valuation data (PE, PB, market cap, etc.) with point-in-time shifting."""
        return self.get_specific_data(start , end , 'val' , field ,
                                      prev = prev , mask = mask , pivot = pivot , drop_old = drop_old)

    def get_mf_data(
        self , start : int | TradeDate , end : int | TradeDate ,
        field : Literal[
            'buy_sm_vol','buy_sm_amount','sell_sm_vol','sell_sm_amount','buy_md_vol',
            'buy_md_amount','sell_md_vol','sell_md_amount','buy_lg_vol','buy_lg_amount',
            'sell_lg_vol','sell_lg_amount','buy_elg_vol','buy_elg_amount','sell_elg_vol',
            'sell_elg_amount','net_mf_vol','net_mf_amount'] | list , 
        mask = False , pivot = False , drop_old = False
    ) -> pd.DataFrame:
        """Return money-flow data (large/medium/small buy-sell classifications)."""
        return self.get_specific_data(start , end , 'mf' , field ,
                                      prev = False , mask = mask , pivot = pivot , drop_old = drop_old)

    def get_limit_data(
        self , start : int | TradeDate , end : int | TradeDate ,
        field : Literal['up_limit','down_limit','pre_close',] | list ,
        mask = False , pivot = False , drop_old = False
    ) -> pd.DataFrame:
        """Return limit-price data (daily limit-up / limit-down prices and previous close)."""
        return self.get_specific_data(start , end , 'limit' , field ,
                                      prev = False , mask = mask , pivot = pivot , drop_old = drop_old)
    
    def get_returns(
        self , start : int | TradeDate , end : int | TradeDate ,
        return_type : Literal['close' , 'vwap' , 'open' , 'intraday' , 'overnight'] = 'close' ,
        pivot = True , mask = True
    ) -> pd.DataFrame:
        """
        Return stock returns for ``[start, end]``.

        Return types
        ------------
        ``'close'``
            Daily close-to-close percentage change (``pctchange`` / 100).
        ``'intraday'``
            Intra-day return: ``close / open - 1``.
        ``'overnight'``
            Overnight return: ``open / prev_close - 1``.
        ``'vwap'``
            VWAP-to-VWAP return computed via ``pct_change`` on
            adjfactor-adjusted VWAP series.
        ``'open'``
            Open-to-open return computed via ``pct_change`` on
            adjfactor-adjusted open series.

        When ``pivot=True`` (default) returns a (date × secid) wide DataFrame.
        When ``mask=True`` applies listing-date masking.
        """
        symbol = 'pctchange'
        if return_type == 'close':
            rets = self.get_quotes(start , end , symbol , mask = False , pivot = False) / 100
        elif return_type == 'intraday':
            rets = self.get_quotes(start , end , ['open' , 'close'] , mask = False , pivot = False)
            rets[symbol] = rets['close'] / rets['open'] - 1
        elif return_type == 'overnight':
            rets = self.get_quotes(start , end , ['open' , 'preclose'] , mask = False , pivot = False)
            rets[symbol] = rets['open'] / rets['preclose'] - 1
        elif return_type in ['vwap' , 'open']:
            price_symbol = 'vwap' if return_type == 'vwap' else 'open'
            rets = self.get_quotes(CALENDAR.td(start , -1) , end , ['adjfactor' , price_symbol] , mask = False , pivot = False).reset_index('date')
            rets['adjfactor'] = rets['adjfactor'].groupby('secid').ffill().fillna(1)
            rets['adjp'] = rets[price_symbol] * rets['adjfactor']
            rets[symbol] = rets['adjp'].pct_change(fill_method=None)
            rets = rets[rets['date'] >= start].reset_index().set_index(['date' , 'secid']).sort_index()
        else:
            raise KeyError(return_type)
        if pivot: 
            rets = rets.pivot_table(symbol , 'date' , 'secid')
        rets = INFO.mask_list_dt(rets , mask)
        return rets
    
    def get_volumes(
        self , start : int | TradeDate , end : int | TradeDate ,
        volume_type : Literal['amount' , 'volume' , 'turn_tt' , 'turn_fl' , 'turn_fr'] = 'volume' , pivot = True , mask = True
    ) -> pd.DataFrame:
        """Return daily trading volume / amount / turnover for the requested type."""
        volumes = self.get_quotes(start , end , volume_type , mask = mask , pivot = pivot)
        return volumes

    def get_turnovers(
        self , start : int | TradeDate , end : int | TradeDate ,
        turnover_type : Literal['tt' , 'fl' , 'fr'] = 'fr' , pivot = True , mask = True
    ) -> pd.DataFrame:
        """Return turnover rate (0-1 scale) for ``'tt'`` / ``'fl'`` / ``'fr'`` types."""
        symbol : Literal['turn_tt' , 'turn_fl' , 'turn_fr'] | Any = f'turn_{turnover_type}'
        turns = self.get_volumes(start , end , symbol , mask = mask , pivot = pivot) / 100
        return turns

    def get_mv(
        self , start : int | TradeDate , end : int | TradeDate ,
        mv_type : Literal['circ_mv' , 'total_mv'] = 'circ_mv' , prev = True , pivot = False , drop_old = False
    ) -> pd.DataFrame:
        """Return circulating or total market cap (unit: 万元 / 10,000 CNY)."""
        return self.get_val_data(start , end , mv_type , prev = prev , pivot = pivot , drop_old = drop_old)

    def get_market_return(
        self , start : int | TradeDate , end : int | TradeDate ,
        return_type : Literal['close' , 'vwap' , 'open' , 'intraday' , 'overnight'] = 'close'
    ) -> pd.DataFrame:
        """Return the cap-weighted market return series indexed by date."""
        rets = self.get_returns(start , end , return_type = return_type , mask = False , pivot = False)
        circ = self.get_mv(start , end , mv_type = 'circ_mv' , pivot = False)
        rets = rets.merge(circ , on = ['date' , 'secid'])
        rets['mv_change'] = rets['pctchange'] * rets['circ_mv']
        mkt_ret : pd.Series | Any = rets.groupby('date').apply(lambda x,**kwg:(x['mv_change']).sum()/x['circ_mv'].sum() , include_groups = False)
        return mkt_ret.rename('market').to_frame()
    
    def get_market_amount(
        self , start : int | TradeDate , end : int | TradeDate
    ) -> pd.DataFrame:
        """Return total market trading amount (sum across all stocks) indexed by date."""
        amount = self.get_volumes(start , end , volume_type = 'amount' , mask = False , pivot = False)
        mkt_amt : pd.Series | Any = amount.groupby('date')['amount'].sum()
        return mkt_amt.rename('market').to_frame()
        
TRADE = TradeDataAccess()