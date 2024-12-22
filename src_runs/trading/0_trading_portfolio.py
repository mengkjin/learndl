#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-22
# description: Update Trading Portfolios
# content: 更新所有交易组合, 无重置指定组合

import sys , pathlib

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

from src.api import TradingAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict()
    with AutoRunTask('update trading portfolios' , **params) as runner:
        TradingAPI.update()

if __name__ == '__main__':
    main()
