#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Fix Factors
# content: 修正所有因子数据
# param_input: True
# param_placeholder: start,end

import sys , pathlib

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

from src.factor.api import FactorCalculatorAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict()
    if params['param'] and ',' in params['param']:
        start , end = [int(s.strip()) for s in str(params['param']).split(',')]
    else:
        start , end = None , None
    with AutoRunTask('fix factors' , **params) as runner:
        FactorCalculatorAPI.fix(start = start , end = end)

if __name__ == '__main__':
    main()
