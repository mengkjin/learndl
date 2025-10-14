#! /usr/bin/env User/mengkjin/workspace/learndl/.venv/bin/python
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Fix Factors
# content: 修正某些因子
# email: True
# mode: shell
# parameters:
#   factor_names : 
#       type : str
#       desc : factor names by ","
#       required : True

from src.res.factor.api import FactorCalculatorAPI
from src.app.script_tool import ScriptTool

@ScriptTool('fix_factors' , lock_name = 'update_factors')
def main(factor_names : str | None = None , **kwargs):
    assert factor_names is not None , 'factor_names is required'
    FactorCalculatorAPI.fix(factors = [s.strip() for s in factor_names.split(',')] , verbosity = 2)

if __name__ == '__main__':
    main()
