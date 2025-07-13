#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-22
# description: Update Trading Portfolios
# content: 更新所有交易组合, 无重置指定组合
# email: True
# close_after_run: False

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.api import TradingAPI
from src.basic import AutoRunTask
from src_runs.util import BackendTaskManager

@BackendTaskManager.manage()
def main(**kwargs):
    with AutoRunTask('update trading portfolios' , **kwargs) as runner:
        TradingAPI.update()

if __name__ == '__main__':
    main()
