#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-22
# description: Update Trading Portfolios
# content: 更新所有交易组合, 无重置指定组合
# email: True
# mode: shell

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
    with AutoRunTask('update_tradeports' , **kwargs) as runner:
        TradingAPI.update()
        runner.critical(f'Update trading portfolios at {runner.update_to} completed')
    return runner

if __name__ == '__main__':
    main()
