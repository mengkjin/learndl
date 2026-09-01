"""
Same-day minute characteristics (no lookback).

Writes ``min_chars/min_chars``.  Date universe follows ``trade_ts/min``.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from src.proj import DB , Base , Dates
from src.data.update.custom.basic import BasicCustomUpdater
from src.data.update.custom.min_chars._common import (
    DB_MIN_SRC ,
    DB_SRC ,
    N_SESS ,
    START_DATE ,
    MinCharsSchedule ,
    follow_source_dates ,
    min_dates ,
    prepare_ret_bars ,
    ret_path_expr ,
    safe_div ,
    save_stage_df ,
    to_date_secid ,
)

__all__ = [
    'MinCharsDailyUpdater' ,
    'calc_min_chars' ,
    'COMPUTABLE_FEATURES' ,
    'REDEFINED_FEATURES' ,
    'APPROX_FEATURES' ,
    'DROPPED_FEATURES' ,
    'OMITTED_OHLC' ,
    'OUTPUT_COLUMNS' ,
]

CLOSE_AUCTION_MINUTE = 237
MIN_STAT_SAMPLES = 3
AM_MINUTE = 120
END15_MINUTE = 225
ST5_MINUTE = 5

DAILY_PX : tuple[str, ...] = ('amt' , 'twap' , 'vwap')
DAILY_FLOW : tuple[str, ...] = ('bwap' , 'swap' , 'bamt' , 'samt' , 'ret_path' , 'bopct')
DAILY_RV : tuple[str, ...] = ('ret_std' , 'ret_skew' , 'ret_kurt' , 'vol_std' , 'vol_hhi' , 'ret_jump')
DAILY_HF_VOL : tuple[str, ...] = ('ret_topk_mean' , 'ret_maxdd' , 'ret_vardown' , 'vol_cv')
DAILY_HF_5MIN : tuple[str, ...] = ('ret_std5' , 'ret_skew5' , 'ret_kurt5' , 'ret_vardown5' , 'vol_cv5')
DAILY_HF_CORR : tuple[str, ...] = (
    'mkt_beta' , 'mkt_corr' , 'ret_autocorr' , 'vol_autocorr' , 'vol_retlag_corr' , 'vol_vwap_corr' ,
)
DAILY_HF_LIQ : tuple[str, ...] = (
    'smart_money' , 'stupid_money' , 'vol_end15_share' , 'vol_open5_share' ,
    'vol_highrank_share' , 'vol_lowrank_share' , 'vol_highdev_share' ,
)
DAILY_HF_MOM : tuple[str, ...] = (
    'ret_am' , 'ret_pm' , 'conf_persist' , 'high_time' ,
    'incvol_ret' , 'vwap_trend' , 'vwap_hlvol' ,
)

SESSION_PX : tuple[str, ...] = ('amt' , 'twap')
SESSION_FLOW : tuple[str, ...] = ('bamt' , 'samt' , 'ret_path')
SESSION_RV : tuple[str, ...] = DAILY_RV
SESSION_CORR : tuple[str, ...] = DAILY_HF_CORR
SESSION_STEMS : tuple[str, ...] = SESSION_PX + SESSION_FLOW + SESSION_RV + SESSION_CORR

APPROX_FEATURES : tuple[str, ...] = ('amt_ca' ,)

OMITTED_OHLC : tuple[str, ...] = (
    'opri' , 'hpri' , 'lpri' , 'cpri' , 'volu' ,
) + tuple(
    f'{stem}{k}h'
    for k in range(1 , N_SESS + 1)
    for stem in ('opri' , 'hpri' , 'lpri' , 'cpri' , 'volu')
)


def _session_cols(stems : tuple[str, ...]) -> tuple[str, ...]:
    """Expand stems into ``{stem}{k}h`` grouped by half-hour ``k = 1..8``."""
    return tuple(f'{stem}{k}h' for k in range(1 , N_SESS + 1) for stem in stems)

REDEFINED_FEATURES : tuple[str, ...] = (
    'bamt' , 'samt' , 'bwap' , 'swap' , 'bopct' , 'bopct_h1' ,
) + tuple(f'{stem}{k}h' for k in range(1 , N_SESS + 1) for stem in ('bamt' , 'samt'))

OUTPUT_COLUMNS : tuple[str, ...] = (
    'date' , 'secid' ,
    *DAILY_PX ,
    *DAILY_FLOW ,
    *APPROX_FEATURES ,
    *DAILY_RV ,
    'bopct_h1' ,
    *DAILY_HF_VOL ,
    *DAILY_HF_5MIN ,
    *DAILY_HF_CORR ,
    *DAILY_HF_LIQ ,
    *DAILY_HF_MOM ,
    *_session_cols(SESSION_STEMS) ,
)

COMPUTABLE_FEATURES : tuple[str, ...] = tuple(
    col for col in OUTPUT_COLUMNS
    if col not in ('date' , 'secid')
    and col not in REDEFINED_FEATURES
    and col not in APPROX_FEATURES
)

def _dropped_large_order_names() -> tuple[str, ...]:
    daily = (
        'bwap_p1' , 'bwap_p5' , 'swap_p1' , 'swap_p5' ,
        'bamt_p1' , 'samt_p1' , 'bamt_p5' , 'samt_p5' ,
        'aret_p1' , 'aret_p5' ,
        'bopct_p1' , 'bopct_p5' , 'bopcth1_p1' , 'bopcth1_p5' ,
    )
    hourly = tuple(
        f'{stem}{k}h{suf}'
        for k in range(1 , N_SESS + 1)
        for stem in ('bamt' , 'samt' , 'aret')
        for suf in ('_p1' , '_p5')
    )
    return daily + hourly

DROPPED_FEATURES : tuple[str, ...] = (
    'volu0h' , 'amtoa' ,
    'l2c0' , 'l2c1' , 'l2c2' , 'l2c3' , 'l2c4' ,
) + _dropped_large_order_names()


def _px_rv_aggs(* , include_side_wap : bool) -> list[pl.Expr]:
    """Amount / TWAP / flow / realized-moment aggregations for a day or session."""
    n = pl.len()
    vol = pl.col('volume')
    vol_sum = vol.sum()
    buy_amt = pl.col('buy_amt')
    sell_amt = pl.col('sell_amt')
    ret = pl.col('ret')
    aggs : list[pl.Expr] = [
        pl.col('amount').sum().alias('amt') ,
        pl.col('close').mean().alias('twap') ,
        buy_amt.sum().alias('bamt') ,
        sell_amt.sum().alias('samt') ,
        ret_path_expr(ret).alias('ret_path') ,
        ret.std().alias('ret_std') ,
        pl.when(n >= MIN_STAT_SAMPLES).then(ret.skew()).otherwise(None).alias('ret_skew') ,
        pl.when(n >= MIN_STAT_SAMPLES).then(ret.kurtosis(fisher = False)).otherwise(None).alias('ret_kurt') ,
        vol.std().alias('vol_std') ,
        safe_div(n * (vol ** 2).sum() , vol_sum ** 2).alias('vol_hhi') ,
        (ret * 100).get(ret.abs().arg_max()).alias('ret_jump') ,
    ]
    if include_side_wap:
        aggs.extend([
            pl.col('close').last().alias('_last_close') ,
            safe_div(pl.col('amount').sum() , vol_sum).alias('vwap') ,
            safe_div((buy_amt * pl.col('px')).sum() , buy_amt.sum()).alias('bwap') ,
            safe_div((sell_amt * pl.col('px')).sum() , sell_amt.sum()).alias('swap') ,
        ])
    return aggs


def _corr_aggs(* , sess : bool) -> list[pl.Expr]:
    """Intraday (or within-session) correlation / beta / autocorr."""
    ret_lag = 'ret_lag_s' if sess else 'ret_lag'
    vol_lag = 'vol_lag_s' if sess else 'vol_lag'
    return [
        safe_div(pl.corr('ret' , 'mkt') * pl.std('ret') , pl.std('mkt')).alias('mkt_beta') ,
        pl.corr('ret' , 'mkt').alias('mkt_corr') ,
        pl.corr('ret' , ret_lag).alias('ret_autocorr') ,
        pl.corr('volume' , vol_lag).alias('vol_autocorr') ,
        pl.corr('volume' , ret_lag).alias('vol_retlag_corr') ,
        pl.corr('volume' , 'px').alias('vol_vwap_corr') ,
    ]


def _hf_daily_aggs() -> list[pl.Expr]:
    """Single-day highfreq building blocks that are not already in px/RV/corr."""
    vol = pl.col('volume')
    vol_sum = vol.sum()
    ret = pl.col('ret')
    px = pl.col('px')
    minute = pl.col('minute')
    ret_down = pl.when(ret < 0).then(ret).otherwise(None)
    vwap_h = px.filter(pl.col('_hvol')).mean()
    vwap_l = px.filter(~pl.col('_hvol')).mean()
    return [
        pl.col('amount').filter(minute >= CLOSE_AUCTION_MINUTE).sum().alias('amt_ca') ,
        ret.filter(ret > pl.col('_err_th')).mean().alias('ret_topk_mean') ,
        (1 - safe_div(pl.col('low') , pl.col('_cum_high'))).max().alias('ret_maxdd') ,
        safe_div(ret_down.std() , ret.std()).alias('ret_vardown') ,
        safe_div(vol.std() , vol.mean()).alias('vol_cv') ,
        safe_div(vol.filter(pl.col('_smart')).sum() , vol_sum).alias('smart_money') ,
        safe_div(vol.filter(pl.col('_stupid')).sum() , vol_sum).alias('stupid_money') ,
        safe_div(vol.filter(minute >= END15_MINUTE).sum() , vol_sum).alias('vol_end15_share') ,
        safe_div(vol.filter(minute < ST5_MINUTE).sum() , vol_sum).alias('vol_open5_share') ,
        safe_div(vol.filter(pl.col('_vol_pct') >= 0.9).sum() , vol_sum).alias('vol_highrank_share') ,
        safe_div(vol.filter(pl.col('_vol_pct') <= 0.1).sum() , vol_sum).alias('vol_lowrank_share') ,
        safe_div(vol.filter(pl.col('_devhigh')).sum() , vol_sum).alias('vol_highdev_share') ,
        ret_path_expr(ret.filter(minute < AM_MINUTE) , pct = False).alias('ret_am') ,
        ret_path_expr(ret.filter(minute >= AM_MINUTE) , pct = False).alias('ret_pm') ,
        (
            minute.filter(pl.col('_down')).median()
            - minute.filter(pl.col('_up')).median()
        ).alias('conf_persist') ,
        minute.filter(pl.col('high') >= pl.col('_high_th')).median().alias('high_time') ,
        ret.filter(pl.col('_incvol')).sum().alias('incvol_ret') ,
        safe_div(pl.corr(px , minute) * px.std() , minute.std()).alias('vwap_trend') ,
        safe_div(vwap_h - vwap_l , (vwap_h + vwap_l) / 2).alias('vwap_hlvol') ,
    ]


def _with_derived(df : pl.DataFrame , * , include_side_wap : bool) -> pl.DataFrame:
    """Add ``bopct`` and fill side WAPs from last close (not stored)."""
    df = df.with_columns(
        (safe_div(pl.col('bamt') , pl.col('bamt') + pl.col('samt')) * 100).alias('bopct') ,
    )
    if include_side_wap:
        last = pl.col('_last_close')
        df = df.with_columns(
            pl.col('vwap').fill_null(last) ,
            pl.col('bwap').fill_null(last) ,
            pl.col('swap').fill_null(last) ,
        ).drop('_last_close')
    return df


def _prepare_bars(raw : pd.DataFrame) -> pl.DataFrame:
    """Attach market return, lags, and bar-level flags used by daily HF stats."""
    df = prepare_ret_bars(raw)
    mkt = df.group_by('minute').agg(pl.col('ret').mean().alias('mkt'))
    df = df.join(mkt , on = 'minute')
    df = df.with_columns(
        pl.col('ret').shift(1).over('secid').alias('ret_lag') ,
        pl.col('volume').shift(1).over('secid').alias('vol_lag') ,
        pl.col('ret').shift(1).over(['secid' , 'sess']).alias('ret_lag_s') ,
        pl.col('volume').shift(1).over(['secid' , 'sess']).alias('vol_lag_s') ,
        pl.col('ret').top_k(5).min().over('secid').alias('_err_th') ,
        pl.when(pl.int_range(pl.len()).over('secid') == 0)
        .then(pl.col('open'))
        .otherwise(pl.col('high').cum_max().shift(1).over('secid'))
        .alias('_cum_high') ,
        (
            pl.col('close').pct_change().over('secid')
            .shift(-1).over('secid')
            .rank(method = 'ordinal').over('secid')
        ).alias('_next_rank') ,
        (
            pl.col('volume').rank(method = 'ordinal').over('secid')
            / pl.col('volume').count().over('secid')
        ).alias('_vol_pct') ,
        pl.col('close').diff().abs().over('secid').alias('_px_dev') ,
        pl.col('ret').mean().over('secid').alias('_ret_mean') ,
        pl.col('ret').std().over('secid').alias('_ret_std') ,
        pl.col('high').top_k(5).min().over('secid').alias('_high_th') ,
        (pl.col('volume') > pl.col('volume').shift(1).over('secid')).alias('_incvol') ,
        (pl.col('volume') >= pl.col('volume').median().over('secid')).alias('_hvol') ,
    )
    return df.with_columns(
        (pl.col('_next_rank') / pl.col('_next_rank').max().over('secid') >= 0.9).alias('_smart') ,
        (pl.col('_next_rank') / pl.col('_next_rank').max().over('secid') <= 0.1).alias('_stupid') ,
        (pl.col('_px_dev') / pl.col('_px_dev').max().over('secid') >= 0.9).alias('_devhigh') ,
        (pl.col('ret') > pl.col('_ret_mean') + pl.col('_ret_std')).alias('_up') ,
        (pl.col('ret') < pl.col('_ret_mean') - pl.col('_ret_std')).alias('_down') ,
    )


def _chars_5min(bars : pl.DataFrame) -> pl.DataFrame:
    """Resample 1-minute bars to 5-minute and compute RV / volume-CV moments."""
    df = (
        bars.with_columns((pl.col('minute') // 5).alias('bin5'))
        .group_by(['secid' , 'bin5'])
        .agg(
            pl.col('open').first().alias('open') ,
            pl.col('close').last().alias('close') ,
            pl.col('volume').sum().alias('volume') ,
        )
        .sort(['secid' , 'bin5'])
        .with_columns(
            pl.when(pl.int_range(pl.len()).over('secid') == 0)
            .then(pl.col('open'))
            .otherwise(pl.col('close').shift(1).over('secid'))
            .alias('preclose')
        )
        .with_columns(
            safe_div(pl.col('close') - pl.col('preclose') , pl.col('preclose')).alias('ret')
        )
    )
    n = pl.len()
    ret = pl.col('ret')
    vol = pl.col('volume')
    ret_down = pl.when(ret < 0).then(ret).otherwise(None)
    return df.group_by('secid').agg(
        ret.std().alias('ret_std5') ,
        pl.when(n >= MIN_STAT_SAMPLES).then(ret.skew()).otherwise(None).alias('ret_skew5') ,
        pl.when(n >= MIN_STAT_SAMPLES).then(ret.kurtosis(fisher = False)).otherwise(None).alias('ret_kurt5') ,
        safe_div(ret_down.std() , ret.std()).alias('ret_vardown5') ,
        safe_div(vol.std() , vol.mean()).alias('vol_cv5') ,
    )


def calc_min_chars(date : int) -> pd.DataFrame:
    """
    Compute one stock-day row of minute-reconstructed characteristics.

    Parameters
    ----------
    date : int
        Trading date ``yyyymmdd``.

    Returns
    -------
    pandas.DataFrame
        Columns ``OUTPUT_COLUMNS``.  Empty if no 1-minute bars exist.
    """
    raw = DB.load(DB_MIN_SRC , 'min' , date , use_alt = True , vb_level = 'never')
    if raw.empty:
        return pd.DataFrame(columns = list(OUTPUT_COLUMNS))

    bars = _prepare_bars(raw)

    daily = bars.group_by('secid').agg(
        _px_rv_aggs(include_side_wap = True)
        + _corr_aggs(sess = False)
        + _hf_daily_aggs()
    )
    daily = _with_derived(daily , include_side_wap = True)
    daily = daily.join(_chars_5min(bars) , on = 'secid' , how = 'left')

    hourly = bars.group_by(['secid' , 'sess']).agg(
        _px_rv_aggs(include_side_wap = False)
        + _corr_aggs(sess = True)
    )
    hourly = _with_derived(hourly , include_side_wap = False)

    bopct_h1 = (
        hourly.filter(pl.col('sess') == 0)
        .select('secid' , pl.col('bopct').alias('bopct_h1'))
    )
    daily = daily.join(bopct_h1 , on = 'secid' , how = 'left')

    value_cols = list(SESSION_STEMS)
    pivoted = hourly.pivot(
        index = 'secid' ,
        on = 'sess' ,
        values = value_cols ,
        aggregate_function = 'first' ,
    )
    rename_map = {
        f'{stem}_{sess}' : f'{stem}{sess + 1}h'
        for sess in range(N_SESS)
        for stem in value_cols
    }
    pivoted = pivoted.rename({old : new for old , new in rename_map.items() if old in pivoted.columns})
    for col in _session_cols(SESSION_STEMS):
        if col not in pivoted.columns:
            pivoted = pivoted.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    out = daily.join(pivoted , on = 'secid' , how = 'left').with_columns(
        pl.lit(int(date)).alias('date')
    )
    return to_date_secid(out.to_pandas() , date , OUTPUT_COLUMNS)


class MinCharsDailyUpdater(MinCharsSchedule , BasicCustomUpdater):
    """Same-day ``min_chars/min_chars``.  Runs first in the min_chars stage order."""
    ENABLED = True
    UPDATE_ORDER = 110
    START_DATE = START_DATE
    DB_SRC = DB_SRC
    DB_KEY = 'min_chars'

    @classmethod
    def proceed_update(
        cls , start : int | None = None , end : int | None = None , overwrite : bool = False , **kwargs
    ) -> Base.UpdateFlag:
        """Compute ``min_chars`` for minute-bar dates not yet stored."""
        if not cls.ENABLED:
            cls.logger.skipping(f'{cls.__name__} disabled (ENABLED=False)' , idt = 1 , vb = 1)
            return Base.UpdateFlag.SKIPPED

        source = min_dates()
        if source.empty:
            cls.logger.skipping(f'{DB_MIN_SRC}/min is empty — nothing to build' , idt = 1 , vb = 1)
            return Base.UpdateFlag.SKIPPED

        target = follow_source_dates(
            cls.DB_KEY , source = source , start = start , end = end ,
            overwrite = overwrite , start_floor = cls.START_DATE ,
        )
        if target.empty:
            cls.logger.skipping(f'{cls.DB_SRC}/{cls.DB_KEY} is up to date vs min' , idt = 1 , vb = 1)
            return Base.UpdateFlag.SKIPPED
        for date in target:
            cls.update_one(int(date))
        cls.logger.success(f'Update {cls.DB_SRC}/{cls.DB_KEY} at {Dates(target)}' , idt = 1 , vb = 1)
        return Base.UpdateFlag.SUCCESS

    @classmethod
    def update_one(cls , date : int) -> None:
        """Compute and save ``min_chars`` for one date; skip if minute bars are missing."""
        if not save_stage_df(
            calc_min_chars(int(date)) , cls.DB_KEY , int(date) ,
            indent = cls.logger.indent + 2 , vb_level = cls.logger.vb_level + 2 ,
        ):
            cls.logger.skipping(
                f'no {DB_MIN_SRC}/min for {date} — skip {cls.DB_KEY}' ,
                idt = 1 , vb = 1 ,
            )
