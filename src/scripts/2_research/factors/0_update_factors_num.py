#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Factors
# content: 更新前num(默认100)组level , date的因子数据
# email: True
# mode: shell
# parameters:
#   num : 
#       type : int
#       desc : update group num
#       default : 100
#       required : True

from src.res.factor.api import FactorCalculatorAPI
from src.basic import AutoRunTask
from src.app import BackendTaskRecorder , ScriptLock

@BackendTaskRecorder()
@ScriptLock('update_factors' , timeout = 10)
def main(**kwargs):
    with AutoRunTask('update_factors_num' , **kwargs) as runner:
        FactorCalculatorAPI.update(groups_in_one_update=int(kwargs.pop('num')) , verbosity = 2)
        runner.critical(f'Update factors at {runner.update_to} completed')

    return runner

if __name__ == '__main__':
    main()
