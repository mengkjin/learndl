#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Factors Within
# content: 更新区间内所有未更新的因子数据
# email: True
# mode: shell
# parameters:
#   start : 
#       type : int
#       desc : start yyyymmdd
#       min : 20250101
#       max : 99991231
#       required : True
#   end : 
#       type : int
#       desc : end yyyymmdd
#       min : 20250101
#       max : 99991231
#       required : True

from src.res.factor.api import FactorCalculatorAPI
from src.basic import AutoRunTask
from src.app import BackendTaskRecorder , ScriptLock    

@BackendTaskRecorder()
@ScriptLock('update_factors' , timeout = 10)
def main(**kwargs):
    with AutoRunTask('update_factors_all' , **kwargs) as runner:
        FactorCalculatorAPI.update(start = int(kwargs.pop('start')) , 
                                   end = int(kwargs.pop('end')) , 
                                   groups_in_one_update = None , verbosity = 2)
        runner.critical(f'Update factors at {runner.update_to} completed')

    return runner

if __name__ == '__main__':
    main()
