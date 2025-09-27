#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Factors
# content: 更新前num(默认100)组level , date的因子数据
# email: True
# mode: shell
# parameters:
#   num : 
#       type : int
#       desc : update group num
#       default : 100
#       required : True

from src.res.factor.api import FactorCalculatorAPI
from src.app.script_tool import ScriptTool

@ScriptTool('update_factors_num' , lock_name = 'update_factors')
def main(num : int | None = None , **kwargs):
    assert num is not None , 'num is required'
    FactorCalculatorAPI.update(groups_in_one_update=int(num) , verbosity = 2)

if __name__ == '__main__':
    main()
