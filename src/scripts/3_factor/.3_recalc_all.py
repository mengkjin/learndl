#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-09-05
# description: Recalculate Factors Within
# content: 重新计算区间内所有因子数据
# email: True
# mode: shell
# parameters:
#   start : 
#       type : int
#       desc : start yyyymmdd
#       min : 20250101
#       max : 99991231
#       required : True
#       default : 20250904
#   end : 
#       type : int
#       desc : end yyyymmdd
#       min : 20250101
#       max : 99991231
#       required : True
#       default : 20250904
#   timeout : 
#       type : float
#       desc : timeout for recalculating factors in hours
#       default : 10

from src.res.factor.api import FactorCalculatorAPI
from src.app import ScriptTool

@ScriptTool('recalc_all' , lock_name = 'update_factors')
def main(start : int | None = None, end : int | None = None , timeout : float | None = 10 , **kwargs):
    assert start is not None and end is not None , 'start and end are required'
    timeout = float(timeout) if timeout is not None else None
    FactorCalculatorAPI.recalculate(start = int(start) , end = int(end) , timeout = timeout, verbosity = 10)
    
if __name__ == '__main__':
    main()
