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

from src.res.factor.api import FactorCalculatorAPI
from src.app.script_tool import ScriptTool

@ScriptTool('update_factors_all' , lock_name = 'update_factors')
def main(start : int | None = None , end : int | None = None , **kwargs):
    assert start is not None and end is not None , 'start and end are required'
    FactorCalculatorAPI.update(start = int(start) , end = int(end) , groups_in_one_update = None , verbosity = 2)
        
if __name__ == '__main__':
    main()
