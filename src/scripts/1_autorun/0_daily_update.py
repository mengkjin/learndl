#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Run Daily Update
# content: 每日更新数据,因子,模型隐变量,模型推理,运行定时任务
# email: True
# mode: shell

from src.res.api import DataAPI , ModelAPI , TradingAPI , NotificationAPI
from src.proj import MACHINE , SharedSync
from src.basic import CALENDAR , TaskScheduler
from src.app.script_tool import ScriptTool

@ScriptTool('daily_update' , CALENDAR.update_to() , forfeit_if_done = True)
def main(**kwargs):
    SharedSync.sync()
    if not MACHINE.updateable:
        ScriptTool.error(f'{MACHINE.name} is not updateable, skip daily update')
    else:
        DataAPI.update()
        if not DataAPI.is_updated():
            ScriptTool.error(f'Data is not updated to the latest date, skip model update')
        else:
            ModelAPI.update()
            TradingAPI.update()
            NotificationAPI.proceed()
            TaskScheduler.run_all_tasks()
        
if __name__ == '__main__':
    main()
        
    