#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2025-08-18
# description: Run Rollback Update
# content: 回滚Tushare更新数据和因子，不重新更新其他数据(当做已经发生)
# email: True
# mode: shell
# parameters:
#   rollback_date : 
#       type : int
#       desc : from which date to rollback and update subsequent data (update start at +1 date)
#       required : True
#       default : 20250814

from src.res.api import DataAPI
from src.proj import MACHINE
from src.basic import AutoRunTask , CALENDAR
from src.app import BackendTaskRecorder , ScriptLock

@BackendTaskRecorder(email = 1)
@ScriptLock('rollback_update' , timeout = 1 , wait_time = 60)
def main(**kwargs):
    rollback_date = int(kwargs.pop('rollback_date'))
    with AutoRunTask('rollback_update' , rollback_date , **kwargs) as runner:
        CALENDAR.check_rollback_date(rollback_date)
        if not MACHINE.updateable:
            runner.error(f'{MACHINE.name} is not updateable, skip rollback update')
        else:
            DataAPI.update_rollback(rollback_date = rollback_date)
            runner.critical(f'Rollback update of {runner.update_to} completed')

    return runner
        
if __name__ == '__main__':
    main()
        
    