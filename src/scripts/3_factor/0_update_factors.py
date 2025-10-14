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
#       min : 20250101
#       max : 99991231
#       required : True
#   end : 
#       type : int
#       desc : end yyyymmdd
#       min : 20250101
#       max : 99991231
#       required : True
#   num_per_group : 
#       type : int
#       desc : number of updates in a group
#       default : 100

from src.res.factor.api import FactorCalculatorAPI
from src.app.script_tool import ScriptTool

@ScriptTool('update_factors' , lock_name = 'update_factors')
def main(start : int | None = None , end : int | None = None , num_per_group : int | None = 100 , **kwargs):
    FactorCalculatorAPI.update(start = start , end = end , groups_in_one_update=num_per_group, verbosity = 2)
        
if __name__ == '__main__':
    main()
