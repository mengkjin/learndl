#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Factors Within
# content: 更新区间内所有未更新的因子数据
# param_input: True
# param_placeholder: start,end
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
    start , end = [int(s.strip()) for s in str(params['param']).split(',')]
    with AutoRunTask('update factors' , **params) as runner:
        FactorCalculatorAPI.update(start = start , end = end , groups_in_one_update = None)

if __name__ == '__main__':
    main()
