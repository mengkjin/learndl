"""
Dongfang-style tail tags from rolling minute percentiles.

A same-day minute is labeled if its return or amount sits above the 20-day
``p95``/``p99`` or below ``p01``/``p05``.  Each tag stores path return
(``ret_path``), mean minute return (``ret_mean``), and amount share (``amt_share``).
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from src.proj import DB , Base , Dates
from src.data.update.custom.basic import BasicCustomUpdater
from src.data.update.custom.min_chars._common import (
    DB_MIN_SRC ,
    DB_SRC ,
    START_DATE ,
    follow_source_dates ,
    min_dates ,
    prepare_ret_bars ,
    safe_div ,
    save_stage_df ,
    to_date_secid ,
)

__all__ = [
    'MinCharsTaggedUpdater' ,
    'calc_min_chars_tag' ,
    'TAG_COLUMNS' ,
    'TAGS' ,
]

# (name, bar column, op, roll threshold column)
# hi: above the upper quantile (p95/p99); lo: below the lower quantile (p01/p05).
TAGS : tuple[tuple[str , str , str , str], ...] = (
    ('rethi99' , 'ret' , '>' , 'ret_p99') ,
    ('rethi95' , 'ret' , '>' , 'ret_p95') ,
    ('retlo01' , 'ret' , '<' , 'ret_p01') ,
    ('retlo05' , 'ret' , '<' , 'ret_p05') ,
    ('amthi99' , 'amount' , '>' , 'amt_p99') ,
    ('amthi95' , 'amount' , '>' , 'amt_p95') ,
    ('amtlo01' , 'amount' , '<' , 'amt_p01') ,
    ('amtlo05' , 'amount' , '<' , 'amt_p05') ,
)

TAG_METRICS : tuple[str, ...] = ('ret_path' , 'ret_mean' , 'amt_share')

TAG_COLUMNS : tuple[str, ...] = (
    'date' , 'secid' ,
) + tuple(f'{metric}_{tag[0]}' for tag in TAGS for metric in TAG_METRICS)

_ROLL_TH : tuple[str, ...] = tuple(tag[3] for tag in TAGS)


def _tag_flag(col : str , op : str , th : str) -> pl.Expr:
    """Boolean mask of bars on one side of a rolling percentile."""
    left , right = pl.col(col) , pl.col(th)
    return left > right if op == '>' else left < right


def _tag_aggs() -> list[pl.Expr]:
    """Path return, mean return, and amount share for every tail tag."""
    day_amt = pl.col('amount').sum()
    aggs : list[pl.Expr] = []
    for name , col , op , th in TAGS:
        flag = _tag_flag(col , op , th)
        ret_f = pl.col('ret').filter(flag)
        amt_f = pl.col('amount').filter(flag)
        n_f = flag.sum()
        aggs.extend([
            pl.when(n_f > 0).then(((ret_f + 1).product() - 1) * 100).otherwise(None).alias(f'ret_path_{name}') ,
            ret_f.mean().alias(f'ret_mean_{name}') ,
            safe_div(amt_f.sum() , day_amt).alias(f'amt_share_{name}') ,
        ])
    return aggs


def calc_min_chars_tag(date : int) -> pd.DataFrame:
    """
    Tag today's minutes by the same-date ``min_chars_roll`` thresholds.

    Parameters
    ----------
    date : int
        Trading date ``yyyymmdd``.

    Returns
    -------
    pandas.DataFrame
        Columns ``TAG_COLUMNS``.  Empty if min or roll is missing.
    """
    raw = DB.load(DB_MIN_SRC , 'min' , date , use_alt = True , vb_level = 'never')
    roll = DB.load(DB_SRC , 'min_chars_roll' , date , vb_level = 'never')
    if raw.empty or roll.empty:
        return pd.DataFrame(columns = list(TAG_COLUMNS))

    th = pl.from_pandas(roll).select(['secid' , *_ROLL_TH])
    bars = prepare_ret_bars(raw).join(th , on = 'secid' , how = 'inner')
    if bars.is_empty():
        return pd.DataFrame(columns = list(TAG_COLUMNS))

    out = bars.group_by('secid').agg(_tag_aggs()).with_columns(
        pl.lit(int(date)).alias('date')
    )
    return to_date_secid(out.to_pandas() , date , TAG_COLUMNS)


class MinCharsTaggedUpdater(BasicCustomUpdater):
    """
    Write ``min_chars/min_chars_tag``.

    Source dates are the intersection of ``trade_ts/min`` and
    ``min_chars/min_chars_roll``.  Missing either side skips the date.
    """
    ENABLED = True
    UPDATE_ORDER = 112
    START_DATE = START_DATE
    DB_SRC = DB_SRC
    DB_KEY = 'min_chars_tag'

    @classmethod
    def proceed_update(
        cls , start : int | None = None , end : int | None = None , overwrite : bool = False , **kwargs
    ) -> Base.UpdateFlag:
        """Compute tagged stats for dates that already have min and roll."""
        if not cls.ENABLED:
            cls.logger.skipping(f'{cls.__name__} disabled (ENABLED=False)' , idt = 1 , vb = 1)
            return Base.UpdateFlag.SKIPPED

        source = min_dates().intersect(DB.dates(DB_SRC , 'min_chars_roll'))
        if source.empty:
            cls.logger.skipping(
                f'{cls.DB_KEY} waits for min ∩ min_chars_roll' , idt = 1 , vb = 1 ,
            )
            return Base.UpdateFlag.SKIPPED

        target = follow_source_dates(
            cls.DB_KEY , source = source , start = start , end = end ,
            overwrite = overwrite , start_floor = cls.START_DATE ,
        )
        if target.empty:
            cls.logger.skipping(f'{cls.DB_SRC}/{cls.DB_KEY} is up to date vs roll' , idt = 1 , vb = 1)
            return Base.UpdateFlag.SKIPPED
        for date in target:
            cls.update_one(int(date))
        cls.logger.success(f'Update {cls.DB_SRC}/{cls.DB_KEY} at {Dates(target)}' , idt = 1 , vb = 1)
        return Base.UpdateFlag.SUCCESS

    @classmethod
    def update_one(cls , date : int) -> None:
        """Compute and save tagged stats; skip if min or roll is missing."""
        if not save_stage_df(
            calc_min_chars_tag(int(date)) , cls.DB_KEY , int(date) ,
            indent = cls.logger.indent + 2 , vb_level = cls.logger.vb_level + 2 ,
        ):
            cls.logger.skipping(
                f'skip {cls.DB_KEY} for {date} (need min + min_chars_roll)' ,
                idt = 1 , vb = 1 ,
            )
