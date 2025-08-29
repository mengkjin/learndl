#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Run Daily Update
# content: 每日更新数据,因子,模型隐变量,模型推理
# email: True
# mode: shell

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.res.api import DataAPI , ModelAPI , TradingAPI , NotificationAPI
from src.proj import MACHINE
from src.basic import AutoRunTask , CALENDAR
from src.app import BackendTaskRecorder , ScriptLock

@BackendTaskRecorder(email = 1)
@ScriptLock('daily_update' , wait_time = 60)
def main(**kwargs):
    with AutoRunTask('daily_update' , CALENDAR.update_to() , **kwargs) as runner:
        if not MACHINE.updateable:
            runner.error(f'{MACHINE.name} is not updateable, skip daily update')
        elif runner.forfeit_task: 
            runner.error(f'task is forfeit, most likely due to finished autoupdate, skip daily update')
        else:
            DataAPI.update()
            ModelAPI.update()
            TradingAPI.update()
            NotificationAPI.proceed()
            runner.critical(f'Daily update of {runner.update_to} completed')

    return runner
        
if __name__ == '__main__':
    main()
        
    