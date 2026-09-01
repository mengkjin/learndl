# coding: utf-8
# author: jinmeng
# date: 2026-09-01
# description: Recalc min_chars_tag overflow
# content: |
#   覆盖重算历史 min_chars/min_chars_tag（float32 overflow 的 ret_path_*）。
#   不碰 daily / roll；只改已有 roll 的日期。需先部署 log1p 路径收益后再跑。
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
#       desc: inclusive end yyyyMMdd; 0 = last stored min ∩ roll date
#       required: False
#       default: 0

from __future__ import annotations

from src.proj import DB , Dates , Logger
from src.proj.util.script import ScriptTool
from src.api.util.wrapper import wrap_update
from src.data.update.custom.min_chars import MinCharsTaggedUpdater
from src.data.update.custom.min_chars._common import DB_MIN_SRC , DB_SRC , START_DATE , min_dates


def _resolve_end(end : int) -> int:
    if int(end) > 0:
        return int(end)
    source = min_dates().intersect(DB.dates(DB_SRC , 'min_chars_roll'))
    assert not source.empty , f'{DB_MIN_SRC}/min ∩ {DB_SRC}/min_chars_roll is empty'
    return int(source.max)


@ScriptTool('recalc_min_chars_tag')
def main(start : int = START_DATE , end : int = 0 , **kwargs):
    """Overwrite ``min_chars/min_chars_tag`` over ``[start, end]`` (roll must exist)."""
    start = max(int(start) , START_DATE)
    end = _resolve_end(end)
    assert start <= end , f'empty range: start={start} end={end}'

    source = Dates(min_dates().intersect(DB.dates(DB_SRC , 'min_chars_roll')) , start , end)
    Logger.note(f'recalc min_chars_tag  start={start} end={end}  overwrite  n={len(source)}')
    if source.empty:
        Logger.warning(f'no min ∩ roll dates in [{start}, {end}]')
        return
    Logger.stdout('path  data/DataBase/DB_min_chars/min_chars_tag/' , indent = 1)

    wrap_update(
        MinCharsTaggedUpdater.proceed_update ,
        'overwrite min_chars/min_chars_tag' ,
        start = start ,
        end = end ,
        overwrite = True ,
    )
    Logger.success(f'min_chars_tag recalc finished  [{start}, {end}]')


if __name__ == '__main__':
    main()
