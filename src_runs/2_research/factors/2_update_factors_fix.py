#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Fix Factors
# content: 修正某些因子
# param_input: True
# param_placeholder: factor_names
# email: False
# close_after_run: False

import sys

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

from src.factor.api import FactorCalculatorAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict()
    if params['param'] and ',' in params['param']:
        factors = [s.strip() for s in str(params['param']).split(',')]
    else:
        factors = []
    with AutoRunTask('fix factors' , **params) as runner:
        FactorCalculatorAPI.fix(factors = factors)

if __name__ == '__main__':
    main()
