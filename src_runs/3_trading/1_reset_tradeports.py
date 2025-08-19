#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-22
# description: Reset Trading Portfolios
# content: 更新指定组合, 输入参数为重置组合
# email: True
# mode: shell
# parameters:
#   reset_port_name : 
#       type : "[p.name for p in Path('data/export/trading_portfolio').iterdir() if not p.name.startswith('.')]"
#       prefix : "tradeport/"
#       desc : port names by ","
#       required : True

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.api import TradingAPI
from src.basic import AutoRunTask
from src_app import BackendTaskRecorder

@BackendTaskRecorder()
def main(**kwargs):
    with AutoRunTask('reset trading portfolios' , **kwargs) as runner:
        TradingAPI.update(reset_ports = [runner['reset_port_name']])
        runner.critical(f'Reset trading portfolios at {runner.update_to} completed')

    return runner

if __name__ == '__main__':
    main()
