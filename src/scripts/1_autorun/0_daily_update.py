#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Run Daily Update
# content: 每日更新数据,因子,模型隐变量,模型推理,运行定时任务
# email: True
# mode: shell

from src.api import UpdateAPI
from src.proj import SharedSync
from src.basic import CALENDAR , TaskScheduler
from src.app import ScriptTool

@ScriptTool('daily_update' , CALENDAR.update_to() , forfeit_if_done = True)
def main(**kwargs):
    SharedSync.sync()
    UpdateAPI.daily()
    TaskScheduler.print_machine_tasks()
            
def run_schedulers():
    TaskScheduler.run_machine_tasks(exclude_script = __file__)
        
if __name__ == '__main__':
    main()
    run_schedulers()
        
    