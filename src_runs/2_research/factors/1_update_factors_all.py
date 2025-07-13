#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Factors Within
# content: 更新区间内所有未更新的因子数据
# email: False
# close_after_run: False
# param_inputs:
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

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.factor.api import FactorCalculatorAPI
from src.basic import AutoRunTask
from src_runs.util import BackendTaskManager

@BackendTaskManager.manage()
def main(**kwargs):
    with AutoRunTask('update factors' , **kwargs) as runner:
        FactorCalculatorAPI.update(start = int(kwargs.pop('start')) , 
                                    end = int(kwargs.pop('end')) , 
                                    groups_in_one_update = None)

if __name__ == '__main__':
    main()
