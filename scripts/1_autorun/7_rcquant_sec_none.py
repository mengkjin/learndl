#!/usr/bin/env python
# description: Restore historical unadjusted RCQuant equity minute bars
# content: 北京时间每日21:00按检查点恢复截至2026-11-24的历史分钟线及衍生周期。
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
