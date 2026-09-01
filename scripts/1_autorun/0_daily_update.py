#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# author: jinmeng
# date: 2024-11-27
# description: Run Daily Update
# content: 每日更新数据,因子,模型隐变量,模型推理,运行定时任务
# email: True
# mode: shell
# parameters:
#   forfeit_if_done:
#       type: bool
#       desc: skip when already finished today and source=bash; pass False from crontab to force rerun
#       required: False
#       default: True

from src.api.pkgs.update import UpdateAPI

from src.proj import CALENDAR
from src.proj.util.script import ScriptTool , TaskScheduler
from src.proj.util.filesys.shared_sync import SharedSync

@ScriptTool('daily_update' , CALENDAR.update_to() , forfeit_if_done = True)
def main(forfeit_if_done : bool = True , **kwargs):
    SharedSync.sync()
    UpdateAPI.daily()
    TaskScheduler.print_machine_tasks()
            
def run_schedulers():
    TaskScheduler.run_machine_tasks(exclude_script = __file__)
        
if __name__ == '__main__':
    main()
    run_schedulers()
        
    