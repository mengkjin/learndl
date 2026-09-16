#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# author: jinmeng
# date: 2026-08-31
# description: Backfill RcQuant sec minute bars
# content: 手动/可选向前补全 RCQuant 股票分钟线；日更已在 sec 最新日下载成功后自动限幅补全（默认最多5天）。本脚本保留用于补跑。
# blacklist:
#   machine: ['Mathews-Mac']
# email: True
# mode: shell
# parameters:
#   force :
#       type : bool
#       desc : skip evening window and daily_update gate (quota still applies)
#       required : False
#       default : False
#   max_days :
#       type : int
#       desc : newest missing trading days; default 5, none/null or explicit empty string means unlimited
#       required : False
#       default : 5

from src.data.download.other_source.rcquant import RcquantMinBarDownloader
from src.proj import CALENDAR
from src.proj.util.script import ScriptTool

@ScriptTool('rcquant_sec_backfill' , CALENDAR.today() , forfeit_if_done = True)
def main(force : bool = False , max_days : int | str | None = 5 , **kwargs):
    RcquantMinBarDownloader.backfill_sec_min(
        force = bool(force) , max_days = max_days ,
    )

if __name__ == '__main__':
    main()
