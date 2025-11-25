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
#       type : str
#       desc : input a factor name
#       required : True
#   short_test : 
#       type : [True , False]
#       desc : short test
#       prefix : "short_test/"
#   start : 
#       type : int
#       desc : start date
#   end : 
#       type : int
#       desc : end date

from src.api import ModelAPI
from src.app import ScriptTool

@ScriptTool('test_factor' , '@factor_name' , lock_num = 0)
def main(factor_name : str | None = None , short_test : bool | None = None , start : int | None = None , end : int | None = None , **kwargs):
    assert factor_name is not None , 'factor_name is required'
    ModelAPI.test_factor(factor_name , short_test , start = start , end = end)

if __name__ == '__main__':
    main()
