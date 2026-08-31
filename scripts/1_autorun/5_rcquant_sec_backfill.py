#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# author: jinmeng
# date: 2026-08-31
# description: Backfill RcQuant sec minute bars
# content: 夜间向前补全 RCQuant 股票分钟线（20110101 起至日更起点前），额度超限或 00:00 立即退出
# blacklist:
#   machine: ['Mathews-Mac']
# email: True
# mode: shell
# parameters:
#   force :
#       type : bool
#       desc : skip 23:xx window and daily_update gate (quota and midnight still apply)
#       required : False
#       default : False

from src.data.download.other_source.rcquant import RcquantMinBarDownloader
from src.proj import CALENDAR
from src.proj.util.script import ScriptTool

@ScriptTool('rcquant_sec_backfill' , CALENDAR.today() , forfeit_if_done = True)
def main(force : bool = False , **kwargs):
    RcquantMinBarDownloader.backfill_sec_min(force = bool(force))

if __name__ == '__main__':
    main()
