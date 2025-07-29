#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Factors
# content: 更新前num(默认100)组level , date的因子数据
# email: False
# close_after_run: False
# param_inputs:
#   num : 
#       type : int
#       desc : update group num
#       default : 100
#       required : True

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.factor.api import FactorCalculatorAPI
from src.basic import AutoRunTask
from src_ui import BackendTaskRecorder

@BackendTaskRecorder()
def main(**kwargs):
    with AutoRunTask('update factors' , **kwargs) as runner:
        FactorCalculatorAPI.update(groups_in_one_update=int(kwargs.pop('num')))
        runner.critical(f'Update factors at {runner.update_to} completed')

    return runner

if __name__ == '__main__':
    main()
