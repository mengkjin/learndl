#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Factors
# content: 更新前num(默认100)组level , date的因子数据
# param_input: True
# param_placeholder: 100
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
    num = str(params['param'])
    num = int(num) if num.isdigit() else 100
    with AutoRunTask('update factors' , **params) as runner:
        FactorCalculatorAPI.update(groups_in_one_update=num)

if __name__ == '__main__':
    main()
