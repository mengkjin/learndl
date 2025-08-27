#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2025-06-22
# description: Analyze Trading Portfolios
# content: 分析交易组合
# email: False
# mode: shell
# parameters:
#   port_name : 
#       type : "[p.name for p in Path('data/Export/trading_portfolio').iterdir() if not p.name.startswith('.')]"
#       prefix : "tradeport/"
#       desc : trade port name
#       required : True
#   start : 
#       type : int
#       desc : start yyyymmdd (or -1)
#   end : 
#       type : int
#       desc : end yyyymmdd

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.res.api import TradingAPI
from src.basic import AutoRunTask
from src.app import BackendTaskRecorder

@BackendTaskRecorder()
def main(**kwargs):
    port_name = kwargs.pop('port_name')
    with AutoRunTask(f'analyze trading portfolio [{port_name}]' , **kwargs) as runner:
        TradingAPI.Analyze(port_name = port_name, start = runner.get('start') , end = runner.get('end'))
        runner.critical(f'Analyze trading portfolio [{port_name}] at {runner.update_to} completed')
    return runner

if __name__ == '__main__':
    main()
