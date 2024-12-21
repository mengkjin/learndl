#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Factors Unlimited
# content: 更新所有未更新的因子数据

import sys , pathlib

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

from src.factor.api import FactorCalculatorAPI
from src_runs.widget import get_argparse_dict

if __name__ == '__main__':
    args = get_argparse_dict()
    FactorCalculatorAPI.update(groups_in_one_update=int(args.get('param',100)))
