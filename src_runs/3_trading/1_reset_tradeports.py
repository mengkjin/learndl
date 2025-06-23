#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-22
# description: Reset Trading Portfolios
# content: 更新指定组合, 输入参数为重置组合
# email: True
# close_after_run: False
# param_inputs:
#   reset_ports : 
#       type : str
#       desc : port names by ","
#       required : True

import sys

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

from src.api import TradingAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict()
    reset_ports = [s.strip() for s in params['port_names'].split(',')]
    with AutoRunTask('reset trading portfolios' , **params , email_if_attachment = True , message_capturer=True) as runner:
        TradingAPI.update(reset_ports = reset_ports)

if __name__ == '__main__':
    main()
