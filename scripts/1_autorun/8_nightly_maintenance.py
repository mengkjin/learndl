#!/usr/bin/env python
# description: Resume scheduled nightly maintenance
# content: 北京时间每日21:00执行有检查点的维护任务。
# mode: shell

from src.proj import CALENDAR
from src.proj.util.script import ScriptTool


@ScriptTool('nightly_maintenance', CALENDAR.today(), lock_name='rcquant_sec_backfill')
def main():
    from src.automation.nightly_maintenance import run
    run()


if __name__ == '__main__':
    main()
