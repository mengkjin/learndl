"""Isolated dd51a3cb 15-minute experiment; never repairs infinite VWAP.

The two aggregation functions below are frozen from dd51a3cb. Do not route
this experiment through the production aggregator or VWAP repair mixin.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import polars as pl
from collections.abc import Callable

from src.proj import CALENDAR, DB
from src.data.util import DataBlock
from src.data.loader import BlockLoader
from ..core import TradePreProcessor
from .registry import TRADE_ALLOW_INACTIVE

BASELINE_COMMIT = "dd51a3cbf48c20470e518c06c9c74f3c005fb234"

def trade_min_fillna(df : pd.DataFrame):
    """
    Fill missing values in minute-bar trade data.

    - ``amount`` and ``volume``: NaN → 0 (no trades occurred in that bar).
    - Price columns (``open``, ``high``, ``low``, ``vwap``, ``close``): NaN →
      the previous bar's close price (carry-forward within each security).

    The DataFrame must have columns ``secid``, ``minute``, ``open``, ``high``,
    ``low``, ``close``, ``amount``, ``volume``, and optionally ``vwap``.
    """
    #'amount' , 'volume' to 0
    df['amount'] = df['amount'].where(~df['amount'].isna() , 0)
    df['volume'] = df['volume'].where(~df['volume'].isna() , 0)

    # close price
    df1 = df.loc[:,['secid','minute','close']].copy().rename(columns={'close':'fix'})
    df1['minute'] = df1['minute'] - df1['minute'].min()
    df = df.merge(df1,on=['secid','minute'],how='left')
    df['fix'] = df['fix'].where(~df['fix'].isna() , df['open'])

    # prices to last time price (min != 0)
    for feat in ['open','high','low','vwap','close']:
        df[feat] = df[feat].where(~df[feat].isna() , df['fix'])
    df = df.drop(columns=['fix'])
    return df

def trade_min_reform(df : pd.DataFrame , x_min_new : int , x_min_old = 1):
    """
    Resample minute-bar data from ``x_min_old``-minute bars to ``x_min_new``-minute bars.

    ``trade_min_fillna`` is applied first to ensure clean inputs.  Bars are
    grouped by ``(secid, new_minute_index)`` and aggregated with OHLCV rules:
    open=first, high=max, low=min, close=last, amount/volume=sum.
    VWAP is recomputed as ``amount / volume`` after aggregation.

    ``x_min_new`` must be a multiple of ``x_min_old``.
    """
    df = trade_min_fillna(df)
    if x_min_new == x_min_old:
        return df
    assert x_min_new % x_min_old == 0 , f'{x_min_new} % {x_min_old} != 0'
    by = x_min_new // x_min_old
    max_nbar = 240 // x_min_old
    assert df['minute'].max() in [max_nbar-1 , max_nbar] , f'{df["minute"].max()} should be {max_nbar-1} or {max_nbar}'
    if df['minute'].max() == max_nbar:
        df['minute'] = df['minute'] -1
    if x_min_new != 1:
        df = df[df['minute'] >= 0].copy()
    df['minute'] = df['minute'].clip(lower=0) // by
    agg_dict = {'open':'first','high':'max','low':'min','close':'last','amount':'sum','volume':'sum'}
    if 'num_trades' in df.columns:
        agg_dict['num_trades'] = 'sum'
    data_new = df.groupby(['secid' , 'minute']).agg(agg_dict)
    assert isinstance(data_new , pd.DataFrame) , f'data_new is not a DataFrame: {data_new}'
    if 'vwap' in df.columns:
        data_new['vwap'] = data_new['amount'] / data_new['volume']
    return data_new.reset_index(drop=False)

class PrePro_15m_original(TradePreProcessor):
    """Old aggregation and preprocessing under an independent cache key."""
    AllowInactive = TRADE_ALLOW_INACTIVE
    # Explicit experiment only; routine daily updates must not extend the snapshot.
    ENABLED = False
    DateChunkYears = 1
    snapshot: DataBlock | None = None
    require_continuous_dates: bool = False
    source_dates: np.ndarray | None = None
    source_observer: Callable[[dict], None] | None = None

    def block_loaders(self, **kwargs) -> dict[str, BlockLoader]:
        return {'day': BlockLoader('trade_ts', 'day', ['turn_fl', 'preclose'], **kwargs)}

    def load_blocks(self, start=None, end=None, secid=None, **kwargs):
        dates = CALENDAR.range(start, end)
        parts = []
        for date in dates:
            if self.source_dates is not None and date not in self.source_dates:
                if self.require_continuous_dates:
                    raise ValueError(f'15m_original: missing min/day source {date}')
                if self.source_observer is not None:
                    self.source_observer({'date': int(date), 'status': 'missing_min_or_day_source'})
                continue
            raw = DB.load('trade_ts', 'min', int(date), use_alt=True, vb_level='never')
            if raw.empty:
                if self.require_continuous_dates:
                    raise ValueError(f'15m_original: missing minute source {date}')
                if self.source_observer is not None:
                    self.source_observer({'date': int(date), 'status': 'empty_minute_source'})
                continue
            required = {'secid', 'minute', 'open', 'high', 'low', 'close', 'amount', 'volume', 'vwap'}
            if not required.issubset(raw.columns):
                raise ValueError(f'15m_original: missing minute columns {required - set(raw.columns)} on {date}')
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                bars = trade_min_reform(raw.copy(), 15)
            if set(bars.minute.unique()) != set(range(16)):
                raise ValueError(f'15m_original: incomplete bar axis on {date}')
            if self.source_observer is not None:
                self.source_observer({'date': int(date), 'minute_rows': len(raw),
                    'amount_dtype': str(raw['amount'].dtype), 'volume_dtype': str(raw['volume'].dtype),
                    'bar_rows': len(bars)})
            bars['date'] = int(date)
            parts.append(bars[['secid', 'date', 'minute', 'open', 'close', 'high', 'low', 'vwap', 'volume']])
        if not parts:
            return {'15m': DataBlock(), 'day': DataBlock()}
        intraday = DataBlock.from_polars(pl.from_pandas(pd.concat(parts, ignore_index=True)))
        intraday = intraday.align(secid=secid, date=dates, inplace=True)
        day = self.block_loaders(**kwargs)['day'].load(start, end).align(
            secid=intraday.secid, date=dates, inplace=True)
        return {'15m': intraday, 'day': day}

    def process(self, blocks):
        data_block = blocks['15m']
        db_day = blocks['day'].align(data_block.secid, data_block.date, inplace=True)
        data_block = data_block.adjust_price(divide=db_day.loc(feature='preclose'))
        return self.allocate_turn_fl_by_bar_volume(data_block, db_day.loc(feature='turn_fl'))

    def pre_process(self, start=None, end=None, *, secid=None, **kwargs):
        if self.snapshot is None or self.snapshot.empty:
            return super().pre_process(start, end, secid=secid, **kwargs)
        dates = CALENDAR.range(start, end)
        existing = self.snapshot.slice_date(start, end).align_secid(secid)
        missing = np.flatnonzero(~np.isin(dates, existing.date))
        pieces = [existing]
        # Recompute only uncovered spans; saved values are already masked/cast.
        for group in np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1):
            if len(group):
                pieces.append(super().pre_process(int(dates[group[0]]), int(dates[group[-1]]),
                                                  secid=secid, **kwargs))
        return DataBlock.merge(pieces)
