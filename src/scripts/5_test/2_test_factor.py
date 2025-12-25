#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2025-11-24
# description: Test Factor
# content: 测试某个因子
# email: True
# mode: shell
# parameters:
#   factor_name : 
#       type : Options.available_factors()
#       desc : input a factor name
#       required : True
#       prefix : "factor@"
#   resume : 
#       type : [True , False]
#       prefix : "resume/"
#       default : "resume/True"
#       desc : resume training , if change schedule file, set to False
#       required : False
#   start : 
#       type : int
#       default : -1
#       desc : start date
#   end : 
#       type : int
#       default : 99991231
#       desc : end date

from src.api import ModelAPI
from src.app import ScriptTool

@ScriptTool('test_factor' , '@factor_name' , lock_num = 0 , verbosity = 9)
def main(factor_name : str | None = None , resume : bool | None = None , start : int | None = None , end : int | None = None , **kwargs):
    assert factor_name is not None , 'factor_name is required'
    ModelAPI.test_factor(factor_name , resume = 0 if resume is None else int(resume) , start = start , end = end)

if __name__ == '__main__':
    main()
