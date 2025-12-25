#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
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
#       min : 20110101
#       max : 99991231
#       required : False
#   end : 
#       type : int
#       desc : end yyyymmdd
#       min : 20110101
#       max : 99991231
#       required : False
#   timeout : 
#       type : float
#       desc : timeout for updating factors in hours
#       default : 10

from src.res.factor.api import FactorCalculatorAPI
from src.app import ScriptTool

@ScriptTool('update_factors' , lock_name = 'update_factors' , verbosity = 10)
def main(start : int | None = None , end : int | None = None , timeout : float | None = 10 , **kwargs):
    timeout = float(timeout) if timeout is not None else None
    FactorCalculatorAPI.update(start = start , end = end , timeout = timeout)

if __name__ == '__main__':
    main()
