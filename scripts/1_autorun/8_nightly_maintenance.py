#!/usr/bin/env python
# description: Resume scheduled nightly maintenance
# content: 可手动立即开始。北京时间21:00自动触发，正在运行时跳过，默认午夜停止开始新步骤。
# mode: shell

def main(**kwargs):
    from src.automation.scheduling.runner import supervise_nightly_script

    # The Hub imports main in its own process; only forward options for this script.
    args = None if __name__ == '__main__' else [f'--{key}={value}' for key, value in kwargs.items()]
    supervise_nightly_script('nightly_maintenance', script_args=args)

    from src.proj import CALENDAR
    from src.proj.util.script import ScriptTool
    from src.automation.nightly_maintenance import run
    task = ScriptTool('nightly_maintenance', CALENDAR.today(), lock_name='rcquant_sec_backfill')
    return task(run)(**kwargs)


if __name__ == '__main__':
    main()
