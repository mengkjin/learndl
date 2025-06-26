#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2025-06-22
# description: Analyze Trading Portfolios
# content: 分析交易组合
# email: False
# close_after_run: False
# param_inputs:
#   port_name : 
#       type : str
#       desc : trade port name
#       required : True
#   start : 
#       type : int
#       desc : start yyyymmdd (or -1)
#   end : 
#       type : int
#       desc : end yyyymmdd

import sys

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

from src.api import TradingAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict()
    port_name = params.pop('port_name')
    start = int(params.pop('start' , -1))
    end = int(params.pop('end' , 99991231))
    with AutoRunTask(f'analyze trading portfolio [{port_name}]' , **params) as runner:
        TradingAPI.Analyze(port_name = port_name , start = start , end = end)

if __name__ == '__main__':
    main()
