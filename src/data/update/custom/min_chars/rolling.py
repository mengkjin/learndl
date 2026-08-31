"""
Rolling window stats: pooled minute distributions + hf trailing of daily chars.

Pooled block: last ``ROLL_WINDOW`` min dates (inclusive) of minute ret/volume/amount.
Trail block: same window of ``min_chars/min_chars`` daily columns, aggregated as
the hf factors do (mean / std / cv / sum; maxdd uses 5 days).
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from src.proj import DB , Base
from src.data.update.custom.basic import BasicCustomUpdater
from src.data.update.custom.min_chars._common import (
    DB_SRC ,
    START_DATE ,
    follow_source_dates ,
    load_daily_chars ,
    load_ret_panel ,
    min_dates ,
    safe_div ,
    save_stage_df ,
    to_date_secid ,
    trailing_aligned_dates ,
)

__all__ = [
    'MinCharsRollUpdater' ,
    'calc_min_chars_roll' ,
    'ROLL_COLUMNS' ,
    'ROLL_WINDOW' ,
    'TRAIL_SPECS' ,
    'QUANTILES' ,
]

ROLL_WINDOW = 20
MAXDD_WINDOW = 5
MIN_STAT_SAMPLES = 3
DAILY_KEY = 'min_chars'
QUANTILES : tuple[float, ...] = (0.01 , 0.05 , 0.50 , 0.95 , 0.99)
_Q_NAMES : tuple[str, ...] = ('p01' , 'p05' , 'p50' , 'p95' , 'p99')
_POOL_MOMENTS : tuple[str, ...] = ('mean' , 'std' , 'skew' , 'kurt')
_SERIES : tuple[str, ...] = ('ret' , 'vol' , 'amt')
_SERIES_COL : dict[str , str] = {'ret' : 'ret' , 'vol' : 'volume' , 'amt' : 'amount'}

# (daily column, agg, window, output name).  One row per unique output — hf
# factors that share the same 20-day mean (e.g. vol_cv) are not duplicated.
TRAIL_SPECS : tuple[tuple[str , str , int , str], ...] = (
    # correlation mean + std
    ('mkt_beta' , 'mean' , 20 , 'mkt_beta_ma20') ,
    ('mkt_corr' , 'mean' , 20 , 'mkt_corr_ma20') ,
    ('ret_autocorr' , 'mean' , 20 , 'ret_autocorr_ma20') ,
    ('vol_autocorr' , 'mean' , 20 , 'vol_autocorr_ma20') ,
    ('vol_retlag_corr' , 'mean' , 20 , 'vol_retlag_corr_ma20') ,
    ('vol_vwap_corr' , 'mean' , 20 , 'vol_vwap_corr_ma20') ,
    ('mkt_beta' , 'std' , 20 , 'mkt_beta_std20') ,
    ('mkt_corr' , 'std' , 20 , 'mkt_corr_std20') ,
    ('ret_autocorr' , 'std' , 20 , 'ret_autocorr_std20') ,
    ('vol_autocorr' , 'std' , 20 , 'vol_autocorr_std20') ,
    ('vol_retlag_corr' , 'std' , 20 , 'vol_retlag_corr_std20') ,
    ('vol_vwap_corr' , 'std' , 20 , 'vol_vwap_corr_std20') ,
    # volatility mean (1min + 5min)
    ('ret_topk_mean' , 'mean' , 20 , 'ret_topk_mean_ma20') ,
    ('ret_std' , 'mean' , 20 , 'ret_std_ma20') ,
    ('ret_std5' , 'mean' , 20 , 'ret_std5_ma20') ,
    ('ret_skew' , 'mean' , 20 , 'ret_skew_ma20') ,
    ('ret_skew5' , 'mean' , 20 , 'ret_skew5_ma20') ,
    ('ret_kurt' , 'mean' , 20 , 'ret_kurt_ma20') ,
    ('ret_kurt5' , 'mean' , 20 , 'ret_kurt5_ma20') ,
    ('ret_vardown' , 'mean' , 20 , 'ret_vardown_ma20') ,
    ('ret_vardown5' , 'mean' , 20 , 'ret_vardown5_ma20') ,
    ('vol_cv' , 'mean' , 20 , 'vol_cv_ma20') ,
    ('vol_cv5' , 'mean' , 20 , 'vol_cv5_ma20') ,
    ('ret_maxdd' , 'max' , 5 , 'ret_maxdd_max5') ,
    # liquidity
    ('smart_money' , 'mean' , 20 , 'smart_money_ma20') ,
    ('stupid_money' , 'mean' , 20 , 'stupid_money_ma20') ,
    ('vol_std' , 'cv' , 20 , 'vol_std_cv20') ,
    ('vol_end15_share' , 'mean' , 20 , 'vol_end15_share_ma20') ,
    ('vol_open5_share' , 'mean' , 20 , 'vol_open5_share_ma20') ,
    ('vol_highrank_share' , 'mean' , 20 , 'vol_highrank_share_ma20') ,
    ('vol_lowrank_share' , 'mean' , 20 , 'vol_lowrank_share_ma20') ,
    ('vol_highdev_share' , 'mean' , 20 , 'vol_highdev_share_ma20') ,
    # momentum (simple trailing only; lm_resid / rank-select stay in the factor layer)
    ('high_time' , 'mean' , 20 , 'high_time_ma20') ,
    ('incvol_ret' , 'sum' , 20 , 'incvol_ret_sum20') ,
    ('vwap_trend' , 'mean' , 20 , 'vwap_trend_ma20') ,
    ('vwap_trend' , 'std' , 20 , 'vwap_trend_std20') ,
    ('vwap_hlvol' , 'mean' , 20 , 'vwap_hlvol_ma20') ,
    ('conf_persist' , 'mean' , 20 , 'conf_persist_ma20') ,
    ('conf_persist' , 'std' , 20 , 'conf_persist_std20') ,
)

_TRAIL_OUT : tuple[str, ...] = tuple(spec[3] for spec in TRAIL_SPECS)
_assert_unique = len(_TRAIL_OUT) == len(set(_TRAIL_OUT))
if not _assert_unique:
    raise ValueError('TRAIL_SPECS output names must be unique')

POOL_COLUMNS : tuple[str, ...] = (
    'n' ,
) + tuple(
    f'{stem}_{name}'
    for stem in _SERIES
    for name in _Q_NAMES
) + tuple(
    f'{stem}_pool_{stat}'
    for stem in _SERIES
    for stat in _POOL_MOMENTS
)

ROLL_COLUMNS : tuple[str, ...] = ('date' , 'secid') + POOL_COLUMNS + _TRAIL_OUT


def _pool_aggs() -> list[pl.Expr]:
    """Percentiles and moments of pooled minute series."""
    aggs : list[pl.Expr] = [pl.len().alias('n')]
    for stem in _SERIES:
        x = pl.col(_SERIES_COL[stem])
        n = pl.len()
        aggs.extend(x.quantile(q).alias(f'{stem}_{name}') for q , name in zip(QUANTILES , _Q_NAMES))
        aggs.extend([
            x.mean().alias(f'{stem}_pool_mean') ,
            x.std().alias(f'{stem}_pool_std') ,
            pl.when(n >= MIN_STAT_SAMPLES).then(x.skew()).otherwise(None).alias(f'{stem}_pool_skew') ,
            pl.when(n >= MIN_STAT_SAMPLES).then(x.kurtosis(fisher = False)).otherwise(None).alias(f'{stem}_pool_kurt') ,
        ])
    return aggs


def _trail_reduce(col : pl.Expr , how : str) -> pl.Expr:
    """Reduce a (possibly filtered) daily column."""
    if how == 'mean':
        return col.mean()
    if how == 'std':
        return col.std()
    if how == 'sum':
        return col.sum()
    if how == 'max':
        return col.max()
    if how == 'cv':
        return safe_div(col.std() , col.mean())
    raise ValueError(f'unknown trail agg {how}')


def _agg_pool(panel : pl.DataFrame) -> pl.DataFrame:
    """Pooled-minute block; empty schema if no bars."""
    if panel.is_empty():
        return pl.DataFrame(schema = {c : pl.Float64 if c != 'secid' else pl.Int64 for c in ('secid' ,) + POOL_COLUMNS})
    return panel.group_by('secid').agg(_pool_aggs())


def _agg_trail(daily_panel : pl.DataFrame) -> pl.DataFrame:
    """hf trailing block from stacked daily min_chars rows (must include ``date``)."""
    cols = ('secid' ,) + _TRAIL_OUT
    if daily_panel.is_empty():
        return pl.DataFrame(schema = {c : pl.Float64 if c != 'secid' else pl.Int64 for c in cols})

    last_dates = sorted(int(x) for x in daily_panel['date'].unique().to_list())
    aggs : list[pl.Expr] = []
    for src , how , win , out in TRAIL_SPECS:
        src_e : pl.Expr = pl.col(src)
        if win < len(last_dates):
            src_e = pl.col(src).filter(pl.col('date').is_in(last_dates[-win:]))
        aggs.append(_trail_reduce(src_e , how).alias(out))
    return daily_panel.group_by('secid').agg(aggs)


def _join_roll(pool : pl.DataFrame , trail : pl.DataFrame , date : int) -> pd.DataFrame:
    """Left-join trail onto pool so names with only daily history still appear."""
    if pool.is_empty() and trail.is_empty():
        return pd.DataFrame(columns = list(ROLL_COLUMNS))
    if pool.is_empty():
        out = trail
    elif trail.is_empty():
        out = pool
    else:
        out = pool.join(trail , on = 'secid' , how = 'full' , coalesce = True)
    out = out.with_columns(pl.lit(int(date)).alias('date'))
    return to_date_secid(out.to_pandas() , date , ROLL_COLUMNS)


def calc_min_chars_roll(date : int , window : int = ROLL_WINDOW) -> pd.DataFrame:
    """
    Pooled minute distribution plus hf trailing of daily chars ending at ``date``.

    Parameters
    ----------
    date : int
        Last date of the window ``yyyymmdd``.
    window : int
        Number of aligned min+daily dates required.
    """
    days = trailing_aligned_dates(int(date) , window , DAILY_KEY)
    if not days:
        return pd.DataFrame(columns = list(ROLL_COLUMNS))
    min_frames = [load_ret_panel(d) for d in days]
    min_frames = [f for f in min_frames if f.height > 0]
    daily_frames = []
    for d in days:
        one = load_daily_chars(d)
        if one.empty:
            continue
        one = one.copy()
        one['date'] = int(d)
        daily_frames.append(pl.from_pandas(one))
    pool = _agg_pool(pl.concat(min_frames) if min_frames else pl.DataFrame())
    trail = _agg_trail(pl.concat(daily_frames) if daily_frames else pl.DataFrame())
    return _join_roll(pool , trail , int(date))


class MinCharsRollUpdater(BasicCustomUpdater):
    """
    Write ``min_chars/min_chars_roll``.

    Follows ``trade_ts/min ∩ min_chars``.  Skips a date when fewer than
    ``ROLL_WINDOW`` aligned files exist.  Sequential dates reuse a sliding cache.
    """
    ENABLED = True
    UPDATE_ORDER = 111
    START_DATE = START_DATE
    DB_SRC = DB_SRC
    DB_KEY = 'min_chars_roll'
    ROLL_WINDOW = ROLL_WINDOW

    @classmethod
    def proceed_update(
        cls , start : int | None = None , end : int | None = None , overwrite : bool = False , **kwargs
    ) -> Base.UpdateFlag:
        """Compute roll stats for aligned min+daily dates not yet stored."""
        if not cls.ENABLED:
            cls.logger.skipping(f'{cls.__name__} disabled (ENABLED=False)' , idt = 1 , vb = 1)
            return Base.UpdateFlag.SKIPPED

        source = min_dates().intersect(DB.dates(DB_SRC , DAILY_KEY))
        if source.empty:
            cls.logger.skipping(f'{cls.DB_KEY} waits for min ∩ {DAILY_KEY}' , idt = 1 , vb = 1)
            return Base.UpdateFlag.SKIPPED

        target = follow_source_dates(
            cls.DB_KEY , source = source , start = start , end = end ,
            overwrite = overwrite , start_floor = cls.START_DATE ,
        )
        if target.empty:
            cls.logger.skipping(f'{cls.DB_SRC}/{cls.DB_KEY} is up to date vs daily' , idt = 1 , vb = 1)
            return Base.UpdateFlag.SKIPPED

        min_cache : dict[int , pl.DataFrame] = {}
        daily_cache : dict[int , pl.DataFrame] = {}
        wrote = 0
        for date in target:
            date = int(date)
            window = trailing_aligned_dates(date , cls.ROLL_WINDOW , DAILY_KEY)
            if len(window) < cls.ROLL_WINDOW:
                cls.logger.skipping(
                    f'{cls.DB_KEY} window short for {date} (need {cls.ROLL_WINDOW} min+daily)' ,
                    idt = 1 , vb = 1 ,
                )
                continue
            live = set(window)
            for cache in (min_cache , daily_cache):
                for old in [k for k in cache if k not in live]:
                    del cache[old]
            min_frames : list[pl.DataFrame] = []
            daily_frames : list[pl.DataFrame] = []
            for d in window:
                if d not in min_cache:
                    min_cache[d] = load_ret_panel(d)
                if min_cache[d].height > 0:
                    min_frames.append(min_cache[d])
                if d not in daily_cache:
                    raw = load_daily_chars(d)
                    if raw.empty:
                        daily_cache[d] = pl.DataFrame()
                    else:
                        raw = raw.copy()
                        raw['date'] = int(d)
                        daily_cache[d] = pl.from_pandas(raw)
                if daily_cache[d].height > 0:
                    daily_frames.append(daily_cache[d])
            if not min_frames or not daily_frames:
                cls.logger.skipping(f'incomplete window for {date}' , idt = 1 , vb = 1)
                continue
            df = _join_roll(
                _agg_pool(pl.concat(min_frames)) ,
                _agg_trail(pl.concat(daily_frames)) ,
                date ,
            )
            if save_stage_df(
                df , cls.DB_KEY , date ,
                indent = cls.logger.indent + 2 , vb_level = cls.logger.vb_level + 2 ,
            ):
                wrote += 1
            else:
                cls.logger.skipping(f'empty {cls.DB_KEY} for {date}' , idt = 1 , vb = 1)

        if wrote == 0:
            return Base.UpdateFlag.SKIPPED
        cls.logger.success(f'Update {cls.DB_SRC}/{cls.DB_KEY} ({wrote} dates)' , idt = 1 , vb = 1)
        return Base.UpdateFlag.SUCCESS

    @classmethod
    def update_one(cls , date : int) -> None:
        """Compute and save roll stats for one date (loads the full window)."""
        if not save_stage_df(
            calc_min_chars_roll(int(date) , cls.ROLL_WINDOW) , cls.DB_KEY , int(date) ,
            indent = cls.logger.indent + 2 , vb_level = cls.logger.vb_level + 2 ,
        ):
            cls.logger.skipping(
                f'skip {cls.DB_KEY} for {date} (need {cls.ROLL_WINDOW} min+daily)' ,
                idt = 1 , vb = 1 ,
            )
