"""
Daily microstructure risk feature updater.

Computes 8 per-stock daily features from 5-minute bar data and day-level
market data, and stores them in the ``exposure/daily_risk`` database.

Features
--------
``true_range``        : normalised daily true range (high-low/preclose)
``turnover``          : free-float turnover rate
``large_buy_pdev``    : large-buy VWAP deviation from stock VWAP
``small_buy_pct``     : small-buy amount as a fraction of total amount (×10)
``sqrt_avg_size``     : square root of average trade size
``open_close_pct``    : open + close period volume fraction (first+last 5min bars)
``ret_volatility``    : cross-bar 5-min return standard deviation
``ret_skewness``      : cross-bar 5-min return skewness
"""
import pandas as pd
import numpy as np

from typing import Any , Literal , Callable
from src.proj import Logger , CALENDAR , DB , Dates , Proj

from src.data.update.custom.basic import BasicCustomUpdater

class DailyRiskUpdater(BasicCustomUpdater):
    """Registered updater for per-stock daily microstructure risk features."""
    START_DATE = max(20100101 , DB.min_date('trade_ts' , '5min' , use_alt=True))
    DB_SRC = 'exposure'
    DB_KEY = 'daily_risk'

    @classmethod
    def update_all(cls , update_type : Literal['recalc' , 'update' , 'rollback'] , indent : int = 1 , vb_level : Any = 1):
        """Update daily risk features for all missing dates up to today."""
        vb_level = Proj.vb(vb_level)
        if update_type == 'recalc':
            Logger.warning(f'Recalculate all custom index is supported , but beware of the performance for {cls.__name__}!')
            stored_dates = np.array([])
        elif update_type == 'update':
            stored_dates = DB.dates(cls.DB_SRC , cls.DB_KEY)
        elif update_type == 'rollback':
            rollback_date = CALENDAR.td(cls._rollback_date)
            stored_dates = CALENDAR.slice(DB.dates(cls.DB_SRC , cls.DB_KEY) , 0 , rollback_date - 1)
        else:
            raise ValueError(f'Invalid update type: {update_type}')
            
        end = min(CALENDAR.updated() , DB.max_date('trade_ts' , '5min' , use_alt=True))
        update_dates = CALENDAR.diffs(cls.START_DATE , end , stored_dates)
        if len(update_dates) == 0:
            Logger.skipping(f'{cls.DB_SRC}/{cls.DB_KEY} is up to date' , indent = indent , vb_level = vb_level)
            return

        for date in update_dates:
            cls.update_one(date , indent = indent + 1 , vb_level = vb_level + 2)

        Logger.success(f'Update {cls.DB_SRC}/{cls.DB_KEY} at {Dates(update_dates)}' , indent = indent , vb_level = vb_level)

    @classmethod
    def update_one(cls , date : int , indent : int = 2 , vb_level : Any = 2):
        """Compute and save daily risk features for a single ``date``."""
        DB.save(calc_daily_risk(date) , cls.DB_SRC , cls.DB_KEY , date , indent = indent , vb_level = vb_level)

def fillinf(series : pd.Series , fill_value : Any = 0) -> pd.Series:
    """Replace non-finite values (NaN, Inf) in a Series with ``fill_value``."""
    return series.where(np.isfinite(series) , fill_value)

def get_inputs(date : int) -> dict[str , pd.DataFrame]:
    """
    Load the three data sources needed for daily risk calculation on ``date``.

    Returns a dict with keys ``'quote'`` (daily OHLCV), ``'moneyflow'`` (daily
    money-flow), and ``'min'`` (5-minute bars), each indexed by ``secid``.
    """
    inputs : dict[str , pd.DataFrame] = {
        'quote' : DB.load('trade_ts' , 'day' , date) ,
        'moneyflow' : DB.load('trade_ts' , 'day_moneyflow' , date),
        'min' : DB.load('trade_ts' , '5min' , date , use_alt=True)
    }
    for name , df in inputs.items():
        if not df.empty:
            inputs[name] = df.set_index('secid')
    return inputs

def calc_daily_risk(date : int):
    """
    Dispatch all 8 daily risk feature functions for ``date`` and concatenate results.

    Returns a DataFrame with one row per secid, aligned to the quote universe.
    """
    inputs = get_inputs(date)
    funcs : list[Callable[[pd.DataFrame,] , pd.Series]] = [
        day_true_range , 
        day_turnover ,
        day_largebuy_price_deviation , 
        day_smallbuy_percentage , 
        day_sqrt_avg_size , 
        day_open_close_percentage , 
        day_5min_ret_volatility , 
        day_5min_ret_skewness
    ]
    results = {func.__name__ : func(**inputs) for func in funcs}
    result = pd.concat([df.reindex(inputs['quote'].index) for df in results.values()] , axis = 1).reset_index()
    return result

def day_true_range(quote : pd.DataFrame , **kwargs) -> pd.Series:
    """Normalised true range: max(high-low, |high-preclose|, |low-preclose|) / preclose."""
    tr = pd.concat([quote['high'] - quote['low'] , (quote['high'] - quote['preclose']).abs() , (quote['low'] - quote['preclose']).abs()] , axis = 1).max(axis = 1)
    tr = fillinf((tr / quote['preclose']).rename('true_range') , 0)
    return tr

def day_turnover(quote : pd.DataFrame , **kwargs) -> pd.Series:
    """Free-float turnover rate (turn_fl / 100)."""
    turnover = quote['turn_fl'] / 100
    return turnover.rename('turnover')

def day_largebuy_price_deviation(quote : pd.DataFrame , moneyflow : pd.DataFrame , **kwargs) -> pd.Series:
    """Absolute deviation of large-buy VWAP from stock VWAP, normalised by VWAP."""
    q = quote.join(moneyflow.loc[:,['buy_elg_amount' , 'buy_elg_vol' , 'buy_lg_amount' , 'buy_lg_vol']])
    q['lbp'] = (q['buy_elg_amount'] + q['buy_lg_amount']) / (q['buy_elg_vol'] + q['buy_lg_vol']) * 100
    q['large_buy_pdev'] = fillinf(abs(q['lbp'] - q['vwap']) / q['vwap'] , np.nan)
    return q['large_buy_pdev']

def day_smallbuy_percentage(quote : pd.DataFrame , moneyflow : pd.DataFrame , **kwargs) -> pd.Series:
    """Small-buy amount as a fraction of total daily amount, scaled by 10."""
    q = quote.join(moneyflow.loc[:,['buy_sm_amount']])
    q['small_buy_pct'] = fillinf((q['buy_sm_amount']) / q['amount'] * 10 , 0)
    return q['small_buy_pct']

def day_sqrt_avg_size(quote : pd.DataFrame , min : pd.DataFrame , moneyflow : pd.DataFrame , **kwargs) -> pd.Series:
    """Square root of average trade size (total amount / estimated trade count)."""
    if not min.empty and 'num_trades' in min.columns:
        num_trades = min.groupby('secid')['num_trades'].sum()
    else:
        avg_size = [5 , 20 , 100 , 500]
        mf_buy_size = moneyflow.loc[:,['buy_sm_amount' , 'buy_md_amount' , 'buy_lg_amount' , 'buy_elg_amount']]
        mf_buy_num = (mf_buy_size / avg_size).sum(axis = 1)

        mf_sell_size = moneyflow.loc[:,['sell_sm_amount' , 'sell_md_amount' , 'sell_lg_amount' , 'sell_elg_amount']]
        mf_sell_num = (mf_sell_size / avg_size).sum(axis = 1)

        num_trades = (mf_buy_num + mf_sell_num) / 2 * 10
        num_trades = num_trades.rename('num_trades')

    q = quote.join(num_trades)
    q['sqrt_avg_size'] = fillinf((q['amount'] / q['num_trades']).pow(0.5) , np.nan)
    return q['sqrt_avg_size']

def day_open_close_percentage(quote : pd.DataFrame , min : pd.DataFrame , **kwargs) -> pd.Series:
    """Fraction of daily amount traded in the opening and closing auction periods (÷1000)."""
    ocamount = min.query('minute <= 5 or minute >= 42').groupby('secid')['amount'].sum().rename('open_close_amount')
    q = quote.join(ocamount)
    q['open_close_pct'] = fillinf(q['open_close_amount'] / q['amount'] / 1000 , 0)
    return q['open_close_pct']

def day_5min_ret_volatility(quote : pd.DataFrame , min : pd.DataFrame , **kwargs) -> pd.Series:
    """Standard deviation of 5-minute bar returns across the trading day."""
    min = min.assign(ret = lambda x: x['close'] / x['close'].shift(1) - 1)
    ret_volatility = min.query('minute >= 1').groupby('secid')['ret'].std().rename('ret_volatility')
    return quote.join(ret_volatility)['ret_volatility']

def day_5min_ret_skewness(quote : pd.DataFrame , min : pd.DataFrame , **kwargs) -> pd.Series:
    """Skewness of 5-minute bar returns across the trading day."""
    min = min.assign(ret = lambda x: x['close'] / x['close'].shift(1) - 1)
    ret_skewness = min.query('minute >= 1').groupby('secid')['ret'].skew().rename('ret_skewness')
    return quote.join(ret_skewness)['ret_skewness']
