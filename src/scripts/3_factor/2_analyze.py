#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-09-05
# description: Recalculate Factors Within
# content: 重新计算区间内所有因子数据
# email: True
# mode: shell
# parameters:
#   factor_name : 
#       type : Options.available_factors()
#       desc : factor name
#       required : True
#       prefix : "factor/"
#   test_type : 
#       type : ['fast' , 'full']
#       desc : test type
#       required : True
#       default : 'fast'
#   start : 
#       type : int
#       desc : start yyyymmdd
#       min : 2010101
#       max : 99991231
#       required : True
#       default : 20170101
#   end : 
#       type : int
#       desc : end yyyymmdd
#       min : 20250101
#       max : 99991231
#       required : True
#       default : 20250904

from src.res.factor.api import FactorTestAPI
from src.app import ScriptTool

@ScriptTool('analyze_factor' , lock_name = 'analyze_factor')
def main(factor_name : str | None = None , test_type : str = 'fast' , start : int | None = 20170101 , end : int | None = None ,  **kwargs):
    assert factor_name is not None , 'factor_name is required'
    if test_type == 'fast':
        FactorTestAPI.FastAnalyze(factor_name , start = start , end = end)
    elif test_type == 'full':
        FactorTestAPI.FullAnalyze(factor_name , start = start , end = end)
    else:
        raise ValueError(f'Invalid test type: {test_type}')
    
if __name__ == '__main__':
    main()
