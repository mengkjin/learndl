#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Fix Factors
# content: 修正某些因子
# email: False
# close_after_run: False
# param_inputs:
#   factor_names : 
#       type : str
#       desc : factor names by ","
#       required : True

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.factor.api import FactorCalculatorAPI
from src.basic import AutoRunTask
from src_runs.util import argparse_dict

def main():
    params = argparse_dict()
    factors = [s.strip() for s in params.pop('factor_names').split(',')]
    with AutoRunTask('fix factors' , **params) as runner:
        FactorCalculatorAPI.fix(factors = factors)

if __name__ == '__main__':
    main()
