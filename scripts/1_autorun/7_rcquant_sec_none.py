#!/usr/bin/env python
# description: Repair historical RCQuant minute bars and rebuild their DataBlocks
# content: 北京时间每日21:00检查缺档和逐日因子，续做历史分钟价量还原，独立记录VWAP越界，并在发布后重算分钟DataBlock。
# blacklist:
#   machine: ['Mathews-Mac']
# email: False
# mode: shell

from src.proj import CALENDAR
from src.proj.util.script import ScriptTool


@ScriptTool('rcquant_sec_none', CALENDAR.today(), lock_name='rcquant_sec_backfill')
def main():
    from src.data.download.other_source.rcquant.sec_none_migration import run
    run()


if __name__ == '__main__':
    main()
