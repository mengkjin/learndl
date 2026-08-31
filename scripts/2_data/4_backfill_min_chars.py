# coding: utf-8
# author: jinmeng
# date: 2026-08-31
# description: Backfill min_chars history
# content: |
#   按 Daily → Roll → Tag 补全 min_chars 历史（落盘 DB_min_chars）。
#   日期宇宙跟随 trade_ts/min，缺分钟线的交易日不发明。
#   默认只写缺失日；overwrite=True 则重算区间内已有文件。
# email: True
# mode: shell
# parameters:
#   start:
#       type: int
#       desc: inclusive start yyyyMMdd (clamped to min_chars START_DATE)
#       required: False
#       default: 20100101
#   end:
#       type: int
#       desc: inclusive end yyyyMMdd; 0 = last stored trade_ts/min date
#       required: False
#       default: 0
#   overwrite:
#       type: [True, False]
#       desc: rewrite existing feathers in the range
#       required: False
#       default: False
#   do_daily:
#       type: [True, False]
#       desc: fill min_chars/min_chars (same-day)
#       required: False
#       default: True
#   do_roll:
#       type: [True, False]
#       desc: fill min_chars/min_chars_roll (needs 20 aligned min+daily)
#       required: False
#       default: True
#   do_tag:
#       type: [True, False]
#       desc: fill min_chars/min_chars_tag (needs same-day roll)
#       required: False
#       default: True

from __future__ import annotations

from src.proj import DB , Dates , Logger
from src.proj.util.script import ScriptTool
from src.api.util.wrapper import wrap_update
from src.data.update.custom.min_chars import (
    MinCharsDailyUpdater ,
    MinCharsRollUpdater ,
    MinCharsTaggedUpdater ,
)
from src.data.update.custom.min_chars._common import (
    DB_MIN_SRC ,
    DB_SRC ,
    START_DATE ,
    follow_source_dates ,
    min_dates ,
)


def _resolve_end(end : int) -> int:
    if int(end) > 0:
        return int(end)
    md = min_dates()
    assert not md.empty , f'{DB_MIN_SRC}/min is empty — nothing to backfill'
    return int(md.max)


def _pending(db_key : str , source : Dates , start : int , end : int , overwrite : bool) -> Dates:
    return follow_source_dates(
        db_key , source = source , start = start , end = end ,
        overwrite = overwrite , start_floor = START_DATE ,
    )


@ScriptTool('backfill_min_chars')
def main(
    start : int = START_DATE ,
    end : int = 0 ,
    overwrite : bool = False ,
    do_daily : bool = True ,
    do_roll : bool = True ,
    do_tag : bool = True ,
    **kwargs ,
):
    """Backfill ``min_chars`` stages in dependency order over ``[start, end]``."""
    start = max(int(start) , START_DATE)
    end = _resolve_end(end)
    overwrite = bool(overwrite)
    assert start <= end , f'empty range: start={start} end={end}'

    source_min = Dates(min_dates() , start , end)
    Logger.note(
        f'min_chars backfill  start={start} end={end} overwrite={overwrite}  '
        f'min_in_range={len(source_min)}'
    )
    if source_min.empty:
        Logger.warning(f'no {DB_MIN_SRC}/min dates in [{start}, {end}]')
        return

    after_daily = source_min if do_daily else source_min.intersect(DB.dates(DB_SRC , 'min_chars'))
    after_roll = after_daily if do_roll else source_min.intersect(DB.dates(DB_SRC , 'min_chars_roll'))
    Logger.stdout(
        f'pending  daily={len(_pending("min_chars" , source_min , start , end , overwrite))}  '
        f'roll={len(_pending("min_chars_roll" , after_daily , start , end , overwrite))}  '
        f'tag={len(_pending("min_chars_tag" , after_roll , start , end , overwrite))}  '
        f'(roll/tag vs upstream after this run; first {MinCharsRollUpdater.ROLL_WINDOW - 1} roll dates skip if window short)' ,
        indent = 1 ,
    )
    Logger.stdout(
        'path  data/DataBase/DB_min_chars/{min_chars,min_chars_roll,min_chars_tag}/' ,
        indent = 1 ,
    )

    if do_daily:
        wrap_update(
            MinCharsDailyUpdater.proceed_update ,
            'backfill min_chars/min_chars' ,
            start = start ,
            end = end ,
            overwrite = overwrite ,
        )
    if do_roll:
        wrap_update(
            MinCharsRollUpdater.proceed_update ,
            'backfill min_chars/min_chars_roll' ,
            start = start ,
            end = end ,
            overwrite = overwrite ,
        )
    if do_tag:
        wrap_update(
            MinCharsTaggedUpdater.proceed_update ,
            'backfill min_chars/min_chars_tag' ,
            start = start ,
            end = end ,
            overwrite = overwrite ,
        )
    Logger.success(f'min_chars backfill finished  [{start}, {end}]')


if __name__ == '__main__':
    main()
