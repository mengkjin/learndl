"""
Minute and 5-minute bar data access singleton for the Chinese A-share market.

Provides 1-minute and 5-minute OHLCV bar data from the ``trade_ts`` database.
Uses a ``PLDFCollection`` (Polars) cache for memory efficiency.
Exported as the ``MKLINE`` singleton.
"""
import polars as pl

from typing import Literal

from src.proj import DB , singleton
from src.data.util import INFO

from .access import DateDataAccess

@singleton
class MinKLineAccess(DateDataAccess):
    """
    Singleton data access object for intraday minute-bar data.

    Stores data as Polars DataFrames (``LOAD_AS_PL = ['min', '5min']``).
    Caches up to 60 dates (``MAX_LEN = 60``).
    """
    MAX_LEN = 60
    DB_SRC = 'trade_ts'
    DB_KEYS = {
        'min' : 'min' , 
        '5min' : '5min'
    }
    LOAD_AS_PL = ['min' , '5min']
    
    def get_secid(self , date : int):
        """Return the listed security universe for ``date`` via ``INFO.get_secid``."""
        return INFO.get_secid(date)

    def data_loader(self , date , data_type):
        """Load a single-date minute-bar slice, filtered to listed securities."""
        df = DB.load(self.DB_SRC , self.DB_KEYS[data_type] , date , vb_level = 'never' , use_alt = True)
        if not df.empty:
            df = df.query('secid in @self.get_secid(@date)')
        return df

    def db_loads_callback(self , *args , **kwargs):
        """No-op: bulk preload is not supported for minute-bar data."""
        return

    @staticmethod
    def reform_mkline(df : pl.DataFrame , with_ret : bool = False):
        """
        Standardise a minute-bar Polars DataFrame.

        Casts ``secid`` to ``Int64``.  When ``with_ret=True``, adds a ``preclose``
        column (previous bar's close, or open for the first bar of the day) and
        computes ``ret = (close - preclose) / preclose``.
        """
        df = df.with_columns(pl.col('secid').cast(pl.Int64))
        if with_ret:
            df = df.with_columns(
                pl.when(pl.col('minute') == 0).then(pl.col('open')).\
                    otherwise(pl.col('close').shift(1)).over('secid').alias('preclose')
            ).with_columns(
                ((pl.col('close') - pl.col('preclose')) / pl.col('preclose')).alias('ret')
            )
        return df

    def get_1min(self , date , with_ret = False):
        """Return 1-minute bar data for ``date`` as a Polars DataFrame."""
        df = self.get_pl(date , 'min')
        df = self.reform_mkline(df , with_ret)
        return df

    def get_5min(self , date , with_ret = False):
        """Return 5-minute bar data for ``date`` as a Polars DataFrame."""
        df = self.get_pl(date , '5min')
        df = self.reform_mkline(df , with_ret)
        return df

    def get_kline(self , date , with_ret = False):
        """Return 1-min data, falling back to 5-min if 1-min is unavailable."""
        df = self.get_1min(date , with_ret)
        if df.shape[0] == 0:
            df = self.get_5min(date , with_ret)
        return df

    def get_inday_corr(self , date : int ,
                       val1 : Literal['ret' , 'volume' , 'mkt' , 'vwap'] | str ,
                       val2 : Literal['ret' , 'volume' , 'mkt' , 'vwap'] | str ,
                       lag1 : int = 0 , lag2 : int = 0 ,
                       rename : str = 'value' , beta = False):
        """
        Compute per-stock intraday correlation (or beta) between two bar-level signals.

        Parameters
        ----------
        val1, val2 : str
            Column names to correlate.  ``'mkt'`` triggers computation of the
            market mean return across stocks.
        lag1, lag2 : int
            Lag (in bars) to shift each series per stock before correlating.
        rename : str
            Output column name for the result.
        beta : bool
            If True, returns the OLS beta (corr × std(val1) / std(val2))
            instead of the plain correlation.
        """
        
        assert val1 != val2 or lag1 != lag2 , \
            f'val1 and val2 must be different or lag1 and lag2 must be different , got {val1} , {val2} , {lag1} , {lag2}'

        df = self.get_kline(date , with_ret = (val1 in ['ret' , 'mkt']) or (val2 in ['ret' , 'mkt']))
        if val1 == 'mkt' or val2 == 'mkt':
            mkt = df.group_by('minute').agg(pl.col('ret').mean().alias('mkt'))
            df = df.join(mkt, on='minute')

        df = df.with_columns(
            pl.col(val1).shift(lag1).over('secid').alias('val1') ,
            pl.col(val2).shift(lag2).over('secid').alias('val2') ,
        )
        if beta:
            df = df.group_by('secid').agg([(pl.corr('val1', 'val2') * pl.std('val1') / pl.std('val2')).alias(rename)])
        else:
            df = df.group_by('secid').agg([pl.corr('val1', 'val2').alias(rename)])
        return df
        
MKLINE = MinKLineAccess()