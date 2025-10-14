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
#   num_per_group : 
#       type : int
#       desc : number of updates in a group
#       default : 100

from src.res.factor.api import FactorCalculatorAPI
from src.app.script_tool import ScriptTool

@ScriptTool('recalc_factors' , lock_name = 'update_factors')
def main(start : int | None = None, end : int | None = None , num_per_group : int | None = 100 , **kwargs):
    assert start is not None and end is not None , 'start and end are required'
    FactorCalculatorAPI.recalculate(start = int(start) , end = int(end) , groups_in_one_update=num_per_group, verbosity = 2)

if __name__ == '__main__':
    main()
