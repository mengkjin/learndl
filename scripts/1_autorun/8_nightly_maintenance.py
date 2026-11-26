#!/usr/bin/env python
# description: Resume scheduled nightly maintenance
# content: 可手动立即开始。北京时间21:00自动触发，正在运行时跳过，默认午夜停止开始新步骤。
# mode: shell

if __name__ == '__main__':
    from src.automation.scheduling.runner import supervise_nightly_script
    supervise_nightly_script('nightly_maintenance')

from src.proj import CALENDAR
from src.proj.util.script import ScriptTool


@ScriptTool('nightly_maintenance', CALENDAR.today(), lock_name='rcquant_sec_backfill')
def main():
    from src.automation.nightly_maintenance import run
    run()


if __name__ == '__main__':
    main()
