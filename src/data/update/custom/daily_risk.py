"""
Daily microstructure risk feature updater.

Computes 8 per-stock daily features from 5-minute bar data and day-level
market data, and stores them in the ``exposure/daily_risk`` database.

Features
--------
``true_range``        : normalised daily true range (high-low/preclose)
``turnover``          : free-float turnover rate
``large_buy_pdev``    : large-buy VWAP deviation from stock VWAP
``small_buy_pct``     : small-buy amount as a fraction of total amount (x10)
``sqrt_avg_size``     : square root of average trade size
``open_close_pct``    : open + close period volume fraction (first+last 5min bars)
``ret_volatility``    : cross-bar 5-min return standard deviation
``ret_skewness``      : cross-bar 5-min return skewness
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import polars as pl

from typing import Any

from src.proj.db.io.output_lock import output_lock
from src.proj import CALENDAR , DB , Base , Dates
from src.data.update.custom.basic import BasicCustomUpdater

__all__ = ['DailyRiskUpdater']

class DailyRiskUpdater(BasicCustomUpdater):
    """Registered updater for per-stock daily microstructure risk features."""
    START_DATE = max(20100101 , DB.min_date('trade_ts' , '5min' , use_alt=True))
    DB_SRC = 'exposure'
    DB_KEY = 'daily_risk'

    @classmethod
    def proceed_update(cls , start : int | None = None , end : int | None = None , overwrite : bool = False , **kwargs) -> Base.UpdateFlag:
        """Update daily risk features for all missing dates up to update_to."""
        start = max(start or cls.START_DATE , cls.START_DATE)
        end = end or min(CALENDAR.updated() , DB.max_date('trade_ts' , '5min' , use_alt=True))
        stored_dates = Dates() if overwrite else DB.dates(cls.DB_SRC , cls.DB_KEY)
        target_dates = Dates(start , end).diff(stored_dates)
        
        if target_dates.empty:
            return Base.UpdateFlag.SKIPPED

        for date in target_dates:
            cls.update_one(date)
        return Base.UpdateFlag.SUCCESS

    @classmethod
    def update_one(cls , date : int):
        """Compute and save daily risk features for a single ``date``."""
        df = calc_daily_risk(date)
        with output_lock(DB.path(cls.DB_SRC, cls.DB_KEY, date)):
            DB.save(df , cls.DB_SRC , cls.DB_KEY , date , indent = cls.logger.indent + 2 , vb_level = cls.logger.vb_level + 2)

def _fillinf(series : pd.Series , fill_value : Any = 0) -> pd.Series:
    """Replace non-finite values (NaN, Inf) in a Series with ``fill_value``."""
    return series.where(np.isfinite(series) , fill_value)

def _get_inputs(date: int) -> dict[str, pd.DataFrame | pl.DataFrame]:
    return {
        'quote': DB.load('trade_ts', 'day', date).set_index('secid'),
        'moneyflow': DB.load('trade_ts', 'day_moneyflow', date).set_index('secid'),
        'min': DB.load_pl('trade_ts', '5min', date, use_alt=True),
    }


def calc_daily_risk(date: int):
    """Compute daily risk features aligned to the daily quote universe."""
    inputs = _get_inputs(date)
    quote, moneyflow = inputs['quote'], inputs['moneyflow']
    daily = [day_true_range(quote), day_turnover(quote),
             day_largebuy_price_deviation(quote, moneyflow),
             day_smallbuy_percentage(quote, moneyflow)]
    minute = _minute_risk(quote, inputs['min'], moneyflow)
    return pd.concat([*daily, minute], axis=1).reset_index()


def day_true_range(quote : pd.DataFrame , **kwargs) -> pd.Series:
    """Normalised true range: max(high-low, abs(high-preclose), abs(low-preclose)) / preclose."""
    tr = pd.concat([quote['high'] - quote['low'] , (quote['high'] - quote['preclose']).abs() , (quote['low'] - quote['preclose']).abs()] , axis = 1).max(axis = 1)
    tr = _fillinf((tr / quote['preclose']).rename('true_range') , 0)
    return tr

def day_turnover(quote : pd.DataFrame , **kwargs) -> pd.Series:
    """Free-float turnover rate (turn_fl / 100)."""
    turnover = quote['turn_fl'] / 100
    return turnover.rename('turnover')

def day_largebuy_price_deviation(quote : pd.DataFrame , moneyflow : pd.DataFrame , **kwargs) -> pd.Series:
    """Absolute deviation of large-buy VWAP from stock VWAP, normalised by VWAP."""
    q = quote.join(moneyflow.loc[:,['buy_elg_amount' , 'buy_elg_vol' , 'buy_lg_amount' , 'buy_lg_vol']])
    q['lbp'] = (q['buy_elg_amount'] + q['buy_lg_amount']) / (q['buy_elg_vol'] + q['buy_lg_vol']) * 100
    q['large_buy_pdev'] = _fillinf(abs(q['lbp'] - q['vwap']) / q['vwap'] , np.nan)
    return q['large_buy_pdev']

def day_smallbuy_percentage(quote : pd.DataFrame , moneyflow : pd.DataFrame , **kwargs) -> pd.Series:
    """Small-buy amount as a fraction of total daily amount, scaled by 10."""
    q = quote.join(moneyflow.loc[:,['buy_sm_amount']])
    q['small_buy_pct'] = _fillinf((q['buy_sm_amount']) / q['amount'] * 10 , 0)
    return q['small_buy_pct']

def _minute_risk(quote: pd.DataFrame, bars: pl.DataFrame, moneyflow: pd.DataFrame) -> pd.DataFrame:
    daily = pl.from_pandas(quote[['amount']].reset_index())
    if bars.is_empty():
        bars = pl.DataFrame(schema={'secid': daily.schema['secid'], 'minute': pl.Int64,
                                    'amount': daily.schema['amount'], 'close': pl.Float64})
    bars = bars.fill_nan(None)
    trades_from_bars = 'num_trades' in bars.columns
    # Preserve the historical positional shift before dropping each opening bar.
    returns = bars.with_columns((pl.col('close') / pl.col('close').shift(1) - 1)
                                .fill_nan(None).alias('ret')).filter(pl.col('minute') >= 1)
    moments = returns.group_by('secid').agg(
        pl.col('ret').std(ddof=1).alias('ret_volatility'),
        pl.when(pl.col('ret').is_infinite().any()).then(None)
        .when((pl.col('ret').count() >= 3) & (pl.col('ret').max() == pl.col('ret').min())).then(0.0)
        .otherwise(pl.col('ret').skew(bias=False)).alias('ret_skewness'),
    )
    open_close = bars.filter((pl.col('minute') <= 5) | (pl.col('minute') >= 42)).group_by('secid').agg(
        pl.col('amount').sum().alias('open_close_amount'),
    )
    if trades_from_bars:
        counts = bars.group_by('secid').agg(pl.col('num_trades').sum())
    else:
        fields = [(f'{side}_{size}_amount', scale) for side in ('buy', 'sell')
                  for size, scale in zip(('sm', 'md', 'lg', 'elg'), (5, 20, 100, 500))]
        counts = pl.from_pandas(moneyflow[[name for name, _ in fields]].reset_index()).select(
            'secid', (pl.sum_horizontal([(pl.col(name) / scale).fill_null(0)
                                         for name, scale in fields]) / 2 * 10).alias('num_trades'),
        )
    ratio = pl.col('open_close_amount') / pl.col('amount') / 1000
    sqrt_size = (pl.col('amount') / pl.col('num_trades')).sqrt()
    result = (
        daily.join(counts, on='secid', how='left', maintain_order='left')
        .join(open_close, on='secid', how='left', maintain_order='left')
        .join(moments, on='secid', how='left', maintain_order='left')
        .select('secid', pl.when(sqrt_size.is_finite()).then(sqrt_size).alias('sqrt_avg_size'),
                pl.when(ratio.is_finite()).then(ratio).otherwise(0).alias('open_close_pct'),
                'ret_volatility', 'ret_skewness')
    )
    return result.to_pandas().set_index('secid').reindex(quote.index)
