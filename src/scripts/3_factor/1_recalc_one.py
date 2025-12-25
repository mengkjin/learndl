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
from src.app import ScriptTool

@ScriptTool('recalc_one' , lock_name = 'update_factors' , verbosity = 10)
def main(factor_names : str | None = None , **kwargs):
    assert factor_names is not None , 'factor_names is required'
    factors = [s.strip() for s in factor_names.split(',')]
    assert len(factors) == 1 , f'only one factor is supported , got {factors}'
    FactorCalculatorAPI.fix(factors = factors)
    
if __name__ == '__main__':
    main()
