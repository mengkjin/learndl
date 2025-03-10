#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-22
# description: Reset Trading Portfolios
# content: 更新指定组合, 输入参数为重置组合
# param_input: True
# param_placeholder: 组合名称, 多个组合用逗号分隔
# email: True
# close_after_run: False

import sys

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

from src.api import TradingAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict()
    with AutoRunTask('reset trading portfolios' , **params , email_if_attachment = True) as runner:
        TradingAPI.update(reset_ports = params['param'].split(','))

if __name__ == '__main__':
    main()
