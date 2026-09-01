"""
Shared minute-bar prep and follow-min update helpers for ``min_chars`` stages.

Not a ``BasicCustomUpdater`` (filename starts with ``_`` so ``import_updaters``
skips it).  Daily / roll / tag stages import from here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from src.proj import DB , Dates , CALENDAR , Base
from src.func.basic import DIV_TOL

N_SESS = 8
MAX_MINUTE = 239
START_DATE = 20100101
# daily_update only fills this many trailing trading days (history → backfill script).
INCR_LOOKBACK_TD = 20
DB_MIN_SRC = 'trade_ts'
MIN_KEY = 'min'
DB_SRC = 'min_chars'
INT_COLS : tuple[str, ...] = ('date' , 'secid' , 'n')
_F32_MAX = float(np.finfo(np.float32).max)

RET_PANEL_COLS : tuple[str, ...] = ('secid' , 'ret' , 'volume' , 'amount')


def safe_div(num : pl.Expr , den : pl.Expr) -> pl.Expr:
    """Divide, returning null when the denominator is ~0."""
    return pl.when(den.abs() <= DIV_TOL).then(None).otherwise(num / den)


def ret_path_expr(ret : pl.Expr , * , pct : bool = True) -> pl.Expr:
    """Path return ``∏(1+ret)−1`` via log-sum so a bad tick cannot overflow."""
    path = ret.log1p().sum().exp() - 1
    return path * 100 if pct else path


class MinCharsSchedule:
    """
    Incremental ``daily_update`` window: machine schedule ∩ last ``INCR_LOOKBACK_TD`` days.

    Historical fill must call ``proceed_update(start=..., end=...)`` directly
    (see ``scripts/2_data/4_backfill_min_chars.py``).
    """
    START_DATE = START_DATE

    @classmethod
    def parse_update_input(
        cls , update_type : Base.UpdateType , rollback_date : int | None = None ,
        start : int | None = None , end : int | None = None , **kwargs ,
    ) -> dict[str , Any]:
        """Schedule-clamped window, then cut to the recent lookback for UPDATE."""
        if update_type == Base.UpdateType.UPDATE:
            start , end = CALENDAR.update_schedule(cls.START_DATE)
            start = max(int(start) , int(CALENDAR.td(end , -INCR_LOOKBACK_TD).as_int()))
            overwrite = False
        elif update_type == Base.UpdateType.ROLLBACK:
            assert rollback_date is not None , 'rollback_date is required for rollback'
            start , end = CALENDAR.update_schedule(rollback_date)
            overwrite = True
        elif update_type == Base.UpdateType.RECALC:
            assert start is not None and end is not None , 'start and end are required for recalculate'
            start , end = CALENDAR.update_schedule(max(int(start) , cls.START_DATE) , end)
            overwrite = True
        else:
            raise ValueError(f'Invalid update type: {update_type}')
        return {'start' : start , 'end' : end , 'overwrite' : overwrite}


def prepare_ret_bars(raw : pd.DataFrame) -> pl.DataFrame:
    """
    Standard 1-minute bars: session id, close-to-close ``ret``, buy/sell weights.

    First bar of each stock-day uses ``open`` as preclose.  ``px`` is bar VWAP
    falling back to close.
    """
    df = pl.from_pandas(raw)
    if 'vwap' not in df.columns:
        df = df.with_columns(pl.col('close').alias('vwap'))
    return (
        df.sort(['secid' , 'minute'])
        .with_columns(
            pl.col('secid').cast(pl.Int64) ,
            pl.col('minute').clip(0 , MAX_MINUTE) ,
            pl.col('amount').fill_null(0.0) ,
            pl.col('volume').fill_null(0.0) ,
        )
        .with_columns(
            (pl.col('minute') // 30).clip(0 , N_SESS - 1).alias('sess') ,
            pl.col('vwap').fill_null(pl.col('close')).alias('px') ,
        )
        .with_columns(
            pl.when(pl.int_range(pl.len()).over('secid') == 0)
            .then(pl.col('open'))
            .otherwise(pl.col('close').shift(1).over('secid'))
            .alias('preclose')
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('preclose') , pl.col('preclose')).alias('ret')
        )
        .with_columns(
            pl.when(pl.col('ret') > 0).then(1.0)
            .when(pl.col('ret') < 0).then(0.0)
            .otherwise(0.5)
            .alias('buy_w')
        )
        .with_columns(
            (pl.col('amount') * pl.col('buy_w')).alias('buy_amt') ,
            (pl.col('amount') * (1 - pl.col('buy_w'))).alias('sell_amt') ,
        )
    )


def load_ret_panel(date : int) -> pl.DataFrame:
    """Load one date of ``secid/ret/volume/amount``; empty frame if min is missing."""
    raw = DB.load(DB_MIN_SRC , MIN_KEY , date , use_alt = True , vb_level = 'never')
    if raw.empty:
        return pl.DataFrame(schema = {c : pl.Float64 if c != 'secid' else pl.Int64 for c in RET_PANEL_COLS})
    return prepare_ret_bars(raw).select(list(RET_PANEL_COLS))


def min_dates() -> Dates:
    """Stored 1-minute dates (``use_alt=True``)."""
    return DB.dates(DB_MIN_SRC , MIN_KEY , use_alt = True)


def trailing_min_dates(date : int , n : int) -> list[int]:
    """Last ``n`` minute-bar dates at or before ``date``; empty if fewer than ``n``."""
    return trailing_aligned_dates(date , n)


def trailing_aligned_dates(date : int , n : int , *extra_keys : str) -> list[int]:
    """Last ``n`` dates that exist in min and every extra ``min_chars`` key."""
    hist = min_dates()
    for key in extra_keys:
        hist = hist.intersect(DB.dates(DB_SRC , key))
    if hist.empty:
        return []
    arr = hist.dates
    arr = arr[arr <= int(date)]
    if arr.size < n:
        return []
    return [int(x) for x in arr[-n:]]


def load_daily_chars(date : int) -> pd.DataFrame:
    """Load one date of ``min_chars/min_chars``; empty if missing."""
    return DB.load(DB_SRC , 'min_chars' , date , vb_level = 'never')


def follow_source_dates(
    db_key : str ,
    * ,
    source : Dates ,
    start : int | None ,
    end : int | None ,
    overwrite : bool ,
    start_floor : int = START_DATE ,
) -> Dates:
    """
    Dates in ``source`` not yet stored under ``min_chars/{db_key}``.

    Does not invent calendar days that have no source file.
    """
    if source.empty:
        return Dates()
    start = max(int(start or start_floor) , start_floor)
    stored = Dates() if overwrite else DB.dates(DB_SRC , db_key)
    return Dates(source , start , end).diff(stored)


def save_stage_df(
    df : pd.DataFrame ,
    db_key : str ,
    date : int ,
    * ,
    indent : int ,
    vb_level : int ,
) -> bool:
    """Save ``df``; return False when empty so the caller can skip without raising."""
    if df.empty:
        return False
    DB.save(df , DB_SRC , db_key , int(date) , indent = indent , vb_level = vb_level)
    return True


def to_date_secid(pdf : pd.DataFrame , date : int , columns : tuple[str, ...]) -> pd.DataFrame:
    """Cast keys to int64, feature columns to float32, and lock column order."""
    missing = [c for c in columns if c not in pdf.columns]
    if missing:
        for c in missing:
            pdf[c] = None
    pdf = pdf.loc[: , list(columns)].copy()
    pdf['date'] = int(date)
    for col in pdf.columns:
        if col in INT_COLS:
            pdf[col] = pd.to_numeric(pdf[col] , errors = 'coerce').fillna(0).astype(np.int64)
        else:
            arr = np.asarray(pd.to_numeric(pdf[col] , errors = 'coerce') , dtype = np.float64)
            out = np.full(arr.shape , np.nan , dtype = np.float32)
            ok = np.isfinite(arr) & (np.abs(arr) <= _F32_MAX)
            out[ok] = arr[ok].astype(np.float32 , copy = False)
            pdf[col] = out
    return pdf
