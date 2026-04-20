"""
Central aggregation facade for all data access singletons in the data pipeline.

``DATAVENDOR`` is the primary entry point for computed and cross-source data:
price-adjusted OHLCV blocks, forward returns, risk model exposures, financial
statement expressions, and day-level quote retrievals.

All underlying singletons (``TRADE``, ``RISK``, ``BS``/``IS``/``CF``/``INDI``/``FINA``,
``ANALYST``, ``MKLINE``, ``EXPO``, ``INFO``) are exposed as class attributes so
callers can access specialised methods directly if needed.
"""
import torch
import numpy as np
import pandas as pd
import polars as pl

from typing import Any , Literal

from src.proj import Logger , CALENDAR , Proj , DB , Dates , singleton , Const
from src.data.util import DataBlock , INFO

from .financial_data import BS , IS , CF , INDI , FINA , FinData
from .analyst import ANALYST
from .min_kline import MKLINE
from .model_data import RISK
from .trade_data import TRADE
from .exposure import EXPO

@singleton
class DataVendor:
    """
    Singleton aggregation facade for factor analysis and portfolio construction.

    Exposes all data singletons as class attributes plus high-level computed
    methods that combine multiple sources into ``DataBlock`` tensors:

    Key methods
    -----------
    ``secid(date)``                : listed security universe
    ``get_quotes_block(dates)``    : price-adjusted OHLCV DataBlock
    ``get_returns_block(s, e)``    : close/vwap daily return DataBlock
    ``nday_fut_ret(secid, date)``  : n-day compounded forward return DataBlock
    ``ffmv(secid, date)``          : float market cap (risk model weight)
    ``risk_style_exp(secid, date)``: CNE5 style factor exposures
    ``get_fin_latest(expr, date)`` : FinData expression → latest Series
    ``get_fin_hist(expr, date)``   : FinData expression → history DataFrame
    ``day_quote(date, price)``     : single-day adjusted price per secid
    ``get_quote_ret(d0, d1)``      : single-period return between two dates
    """
    CALENDAR = CALENDAR
    TRADE = TRADE
    INFO = INFO
    RISK = RISK
    
    INDI = INDI
    BS   = BS
    IS   = IS
    CF   = CF
    FINA = FINA

    ANALYST = ANALYST

    MKLINE = MKLINE
    EXPO = EXPO

    CUSTOM_DATA = {}
    
    def __init__(self):
        """Initialise per-date caches and load the full stock listing at startup."""
        self.day_quotes : dict[int,pd.DataFrame] = {}
        self.day_secids : dict[int,np.ndarray] = {}

        self.init_stocks()

    def data_storage_control(self):
        """Truncate all underlying singleton caches to free memory."""
        self.TRADE.truncate()
        self.RISK.truncate()

        self.INDI.truncate()
        self.BS.truncate()
        self.IS.truncate()
        self.CF.truncate()
        self.FINA.truncate()

    def init_stocks(self , listed = True , exchange = ['SZSE', 'SSE', 'BSE']):
        """Load the full stock description and ST status into ``all_stocks``/``st_stocks``."""
        with Proj.silence:
            self.all_stocks = self.INFO.get_desc(set_index=False , listed = listed , exchange = exchange)
            self.st_stocks = self.INFO.get_st()

    def secid(self , date : int | None = None) -> np.ndarray: 
        """get secid of all stocks or at a specific date"""
        if date is None: 
            return np.unique(self.all_stocks['secid'].to_numpy(int))
        if date not in self.day_secids:
            self.day_secids[date] = self.INFO.get_secid(date)
        return self.day_secids[date]

    def db_loads_callback(self , df : pd.DataFrame | pl.DataFrame , db_src : str , db_key : str):
        """Receive bulk-loaded trade_ts data from ``DB.loads`` and forward it to ``TRADE``'s cache."""
        if not db_src == 'trade_ts' or len(df) == 0:
            return
        for data_type , data_key  in self.TRADE.DB_KEYS.items():
            if data_key == db_key:
                if isinstance(df , pl.DataFrame):
                    df = df.to_pandas()
                self.TRADE.collections[data_type].add_long_frame(df.reset_index().drop(columns = ['index'] , errors = 'ignore').set_index(self.TRADE.DATE_KEY))
        
    @classmethod
    def td_within(cls , start : int | None = None , end : int | None = None , step : int = 1 , updated = False , extend = 0):
        """Return a trading-day date array for [start, end] with optional step and extension."""
        if extend > 0:
            if end is not None:
                end = cls.cd(end , extend)
            if start is not None:
                start = cls.cd(start , -extend)
        dates = CALENDAR.range(start , end , 'td' , step = step , updated = updated)
        return dates
    
    @staticmethod
    def td_array(date , offset : int = 0): return CALENDAR.td_array(date , offset)
    
    @staticmethod
    def td(date , offset : int = 0): return CALENDAR.td(date , offset).as_int()

    @staticmethod
    def cd(date , offset : int = 0): return CALENDAR.cd(date , offset)

    @classmethod
    def real_factor(cls , factor_type : Literal['pred' , 'factor'] , names : str | list[str] | np.ndarray ,
                    start = 20240101 , end = 20240531 , step = 5):
        """Load named factor or model prediction DataBlocks from the DB and merge them."""
        if isinstance(names , str): 
            names = [names]
        dates = DATAVENDOR.td_within(start , end , step)

        # values = [DB.loads(factor_type , name , dates , key_column = 'date') for name in names]
        # values = [v.set_index(['secid','date']) for v in values if not v.empty]
        # if values:
        #     return DataBlock.from_pandas(pd.concat(values , axis=1).sort_index())

        values = [DB.loads_pl(factor_type , name , dates , key_column = 'date') for name in names]
        values = [v for v in values if len(v) > 0]
        if values:
            value = values[0]
            for add in values[1:]:
                value = value.join(add , on = ['secid' , 'date'] , how = 'full')
            return DataBlock.from_polars(value)
        
        Logger.alert1(f'None of {factor_type} {names} found in {start} ~ {end}')
        return DataBlock()

    @classmethod
    def stock_factor(cls , factor_names : str | list[str] | np.ndarray , start = 20240101 , end = 20240531 , step = 5):
        """Load named stock factors from the ``factor`` DB into a DataBlock."""
        return cls.real_factor('factor' , factor_names , start , end , step)

    @classmethod
    def model_preds(cls , model_names : str | list[str] | np.ndarray , start = 20240101 , end = 20240531 , step = 5):
        """Load named model predictions from the ``pred`` DB into a DataBlock."""
        return cls.real_factor('pred' , model_names , start , end , step)

    def random_factor(self , start = 20240101 , end = 20240531 , step = 5 , nfactor = 2):
        """Generate a random-value DataBlock (for testing)."""
        date  = self.td_within(start , end , step)
        secid = self.secid()
        return DataBlock(np.random.randn(len(secid),len(date),1,nfactor),
                         secid,date,[f'factor{i+1}' for i in range(nfactor)])

    def update_named_data_block(
        self,
        data_key: Literal["daily_quotes", "risk_exp"],
        db_src: str,
        db_key: str,
        dates: np.ndarray | list[int] | int | None = None,
        extend=0,
    ):
        """
        Lazily extend an internal DataBlock attribute to cover the requested dates.

        Maintains the attribute ``_block_{data_key}`` and extends it with any
        missing dates by calling ``DataBlock.extend_to``.  Only saves if the
        new target range is wider than the currently loaded range.
        For ``daily_quotes``, applies price adjustment after extension.
        """
        if dates is None or (not isinstance(dates , int) and len(dates) == 0):
            return
        if isinstance(dates , int):
            target_start , target_end = dates , dates
        elif isinstance(dates , np.ndarray):
            target_start , target_end = dates.min() , dates.max()
        elif isinstance(dates , list):
            target_start , target_end = min(dates) , max(dates)
        else:
            raise ValueError(f'Unknown dates type: {type(dates)}')
        target_start , target_end = self.cd(target_start , -extend) , self.cd(target_end , extend)
        target_start , target_end = min(target_start , CALENDAR.update_to()) , min(target_end , CALENDAR.update_to())

        block0 : DataBlock = getattr(self , f'_block_{data_key}' , DataBlock())
        loaded_start , loaded_end = block0.min_date , block0.max_date
        
        block0 = block0.extend_to(db_src , db_key , target_start , target_end , inplace = True)

        if data_key == 'daily_quotes':
            block0 = block0.adjust_price()
        
        Logger.success(f'DATAVENDOR.{data_key} expand from {Dates(loaded_start,loaded_end)} to {Dates(target_start,target_end)}')
        setattr(self , f'_block_{data_key}' , block0)

    def update_return_block(self , start : int , end : int):
        """Compute and cache the daily return DataBlock for [start, end] if not already loaded."""
        td_within = self.td_within(start , end , updated = True)
        daily_ret = getattr(self , f'_block_daily_ret' , DataBlock())
        if daily_ret.date is None or not np.isin(td_within , daily_ret.date).all():
            pre_start_dt = CALENDAR.cd(start , -20)
            extend_td_within = self.td_within(pre_start_dt , end)
            blk = self.get_quotes_block(extend_td_within).align(date = extend_td_within , feature = ['close' , 'vwap']).ffill()
            rtn = torch.nn.functional.pad(blk.values[:,1:] / blk.values[:,:-1] - 1 , (0,0,0,0,1,0) , value = torch.nan)
            blk.update(values = torch.where(rtn.isinf() , torch.nan , rtn))
            blk = blk.align_date(blk.date_within(start , end) , inplace = True)
            setattr(self , f'_block_daily_ret' , blk)

    def get_quotes_block(self , dates : np.ndarray | list[int] | int | None = None , * , extend = 0) -> DataBlock:
        """Return a price-adjusted OHLCV DataBlock covering ``dates`` (lazy-loaded and cached)."""
        with Proj.silence:
            self.update_named_data_block('daily_quotes' , 'trade_ts' , 'day' , dates , extend = extend)
        return getattr(self , f'_block_daily_quotes' , DataBlock())

    def get_risk_exp(self , dates : np.ndarray | list[int] | int | None = None , * , extend = 0) -> DataBlock:
        """Return the CNE5 risk model exposure DataBlock covering ``dates``."""
        with Proj.silence:
            self.update_named_data_block('risk_exp' , 'models' , 'tushare_cne5_exp' , dates , extend = extend)
        return getattr(self , f'_block_risk_exp' , DataBlock())

    def get_returns_block(self , start : int , end : int):
        """Return the daily close/vwap return DataBlock for [start, end] (lazy-loaded)."""
        with Proj.silence:
            self.update_return_block(start , end)
        return getattr(self , f'_block_daily_ret' , DataBlock())
    
    def day_quote(self , date : int | Any , price : Literal['close' , 'vwap' , 'open'] = 'close'):
        """Return a ``(secid, price)`` DataFrame for a single date, with adjfactor applied."""
        df = self.TRADE.get_trd(date , ['secid' , 'adjfactor' , price])
        if not df.empty:
            df['price'] = df[price] * df['adjfactor'].fillna(1)
            return df.loc[:,['secid' , 'price']]
        else:
            return pd.DataFrame(columns = ['secid' , 'price'])
    
    def get_quote_ret(self , date0 , date1 , 
                      price0 : Literal['close' , 'vwap' , 'open'] = 'close' ,
                      price1 : Literal['close' , 'vwap' , 'open'] = 'close' ,
                      secid : np.ndarray | pd.Series | Any | None = None):
        """
        get ret of single date0 and date1
        using DataFrame method is much faster than DataBlock method
        slicing of smaller df is much faster than slicing of larger array
        """
        q0 = self.day_quote(date0 , price0)
        q1 = self.day_quote(date1 , price1)
        if q0.empty or q1.empty: 
            return pd.DataFrame(columns = ['secid' , 'ret']).set_index('secid')
        q = q0[q0['price'] != 0].merge(q1 , on = 'secid')
        q['ret'] = q['price_y'] / q['price_x'] - 1
        q = q[['secid' , 'ret']].set_index('secid')
        if secid is not None:
            q = q.reindex(secid).fillna(0)
        return q

    def get_quote_ret_new(self , date0 , date1 , 
                      price0 : Literal['close' , 'vwap' , 'open'] = 'close' ,
                      price1 : Literal['close' , 'vwap' , 'open'] = 'close' ,
                      secid : np.ndarray | pd.Series | Any | None = None):
        blk = self.get_quotes_block([date0 , date1])
        p0 = blk.loc(date = date0 , feature = price0).flatten()
        p1 = blk.loc(date = date1 , feature = price1).flatten()
        if len(p0) == 0 or len(p1) == 0:
            return pd.DataFrame(columns = ['secid' , 'ret']).set_index('secid')
        q = pd.DataFrame({'secid' : blk.secid , 'ret' : p1 / np.where(p0 == 0 , np.nan , p0) - 1}).set_index('secid')
        if secid is not None:
            q = q.reindex(secid).fillna(0)
        return q

    def get_miscel_ret(self , df : pd.DataFrame , ret_type : Literal['close' , 'vwap'] = 'close') -> pd.DataFrame:
        """Return per-row returns for arbitrary (secid, start, end) combinations.

        ``df`` must contain ``'secid'``, ``'start'``, and ``'end'`` columns.
        Returns the same DataFrame with an added ``'ret'`` column.
        """
        """get ret of miscel secids and dates, df must contain 'secid' , 'start' , 'end' columns"""
        assert 'secid' in df.columns and 'start' in df.columns and 'end' in df.columns , \
            f'df must contain "secid" , "start" , "end" columns : {df.columns}'
        df['prev'] = CALENDAR.td_array(df['start'] , -1)
        dates = np.unique(np.concatenate([df['prev'].to_numpy() , df['end'].to_numpy()]))
        quotes = DB.loads('trade_ts' , 'day' , dates).filter(items = ['secid' , 'date' , ret_type , 'adjfactor'])
        quotes[ret_type] = quotes[ret_type] * quotes['adjfactor']

        q0 = df.merge(quotes , left_on = ['secid' , 'prev'] , right_on = ['secid' , 'date'] , how = 'left')[ret_type]
        q1 = df.merge(quotes , left_on = ['secid' , 'end'] , right_on = ['secid' , 'date'] , how = 'left')[ret_type]
        ret = q1 / q0 - 1
        df['ret'] = ret
        df.drop(columns = ['prev'] , inplace = True)
        return df

    def nday_fut_ret(self , secid : np.ndarray , date : np.ndarray , nday : int = 10 , lag : int = 2 ,
                     ret_type : Literal['close' , 'vwap'] = 'close'):
        """
        Compute n-day compounded forward returns for each (secid, date).

        Returns a DataBlock aligned to (secid, date) with the ``'ret'`` feature
        representing the product of daily returns over the ``[lag, lag+nday)``
        window after each date.  ``lag`` must be >= 1 to avoid look-ahead bias.
        """
        assert lag > 0 , f'lag must be positive : {lag}. If you want to use next day\'s return, set lag = 1'
        date_min = self.td(date.min() , -10)
        date_max = self.td(int(date.max()) , nday + lag + 10)
        full_date = self.td_within(date_min , date_max)
        blk = self.get_returns_block(date_min , date_max).align(secid , full_date , ret_type)
        values = torch.nn.functional.pad(blk.values[:,lag:] , (0,0,0,0,0,lag) , value = torch.nan).unfold(1 , nday , 1).exp().prod(dim = -1) - 1
        blk.update(values = values , date = full_date[:values.shape[1]] , feature = ['ret']).align_date(date , inplace = True)
        return blk
    
    def ffmv(self , secid : np.ndarray , date : np.ndarray , prev = True):
        """
        Return float market cap (``weight`` feature) aligned to (secid, date).

        When ``prev=True`` (default) loads from the previous trading day and
        relabels dates forward to maintain point-in-time correctness.
        """
        if prev :
            date = self.td_array(date , -1)
        blk = self.get_risk_exp(date).align(secid , date , ['weight'])
        if prev :
            blk.date = self.td_array(blk.date , 1)
        return blk

    def risk_style_exp(self , secid : np.ndarray , date : np.ndarray):
        """Return CNE5 style factor exposures aligned to (secid, date)."""
        blk = self.get_risk_exp(date).align(secid , date , Const.Factor.RISK.style)
        return blk

    def risk_industry_exp(self , secid : np.ndarray , date : np.ndarray):
        """Return CNE5 industry factor exposures aligned to (secid, date)."""
        blk = self.get_risk_exp(date).align(secid , date , Const.Factor.RISK.indus)
        return blk

    def get_ffmv(self , secid : np.ndarray , d : int):
        """Return float market cap weights as a 1-D numpy array for the given (secid, date)."""
        if not CALENDAR.is_trade_date(d):
            return None
        blk = self.get_risk_exp(d)
        value = blk.loc(secid = secid , date = d , feature = 'weight').flatten()
        return value

    def get_cp(self , secid : np.ndarray | list[int] , d : int):
        """Return closing prices for the given secids on trading day ``d``."""
        return self.TRADE.get_trd(self.td(d) , ['secid' , 'close']).set_index('secid').reindex(secid)['close'].to_numpy()

    def get_fin_latest(self , expression : str , date : int , new_name : str | None = None , **kwargs) -> pd.Series:
        """Evaluate a ``FinData`` expression and return the latest per-secid value as a Series."""
        fin_data = FinData(expression , **kwargs)
        return fin_data.get_latest(date , new_name)

    def get_fin_hist(self, expression : str , date : int , lastn : int , new_name : str | None = None , **kwargs) -> pd.DataFrame:
        """Evaluate a ``FinData`` expression and return the trailing ``lastn`` quarterly history."""
        fin_data = FinData(expression , **kwargs)
        return fin_data.get_hist(date , lastn , new_name)

    def get_fin_qoq(self , expression : str , date : int , lastn : int , method : Literal['pct' , 'diff'] = 'pct' , **kwargs) -> pd.DataFrame:
        """Compute quarter-on-quarter changes for a FinData expression."""
        data = FinData(expression , **kwargs).get_hist(date = date , lastn = lastn + 2)
        full_index = pd.MultiIndex.from_product([data.index.get_level_values('secid').unique() ,
                                                 data.index.get_level_values('end_date').unique()])
        df_yoy = data.reindex(full_index)
        df_yoy_base = df_yoy.groupby('secid').shift(1)
        df_yoy = (df_yoy - df_yoy_base) 
        if method == 'pct':
            df_yoy = df_yoy / df_yoy_base.abs()

        df_yoy = df_yoy.reindex(data.index).where(~data.isna() , np.nan).replace([np.inf , -np.inf] , np.nan)
        return df_yoy.groupby('secid').tail(lastn)
    
    def get_fin_yoy(self , expression : str , date : int , lastn : int , method : Literal['pct' , 'diff'] = 'pct' , **kwargs) -> pd.DataFrame:
        """Compute year-on-year changes (4-quarter lag) for a FinData expression."""
        data = FinData(expression , **kwargs).get_hist(date = date , lastn = lastn + 5)
        full_index = pd.MultiIndex.from_product([data.index.get_level_values('secid').unique() ,
                                                 data.index.get_level_values('end_date').unique()])
        df_yoy = data.reindex(full_index)
        df_yoy_base = df_yoy.groupby('secid').shift(4)
        df_yoy = (df_yoy - df_yoy_base) 
        if method == 'pct':
            df_yoy = df_yoy / df_yoy_base.abs()

        df_yoy = df_yoy.reindex(data.index).where(~data.isna() , np.nan).replace([np.inf , -np.inf] , np.nan)
        return df_yoy.groupby('secid').tail(lastn)
    
DATAVENDOR = DataVendor()