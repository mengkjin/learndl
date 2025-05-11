#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Run Daily Update
# content: 每日更新数据,因子,模型隐变量,模型推理
# email: True
# close_after_run: False

import sys 

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

from src.api import DataAPI , ModelAPI , TradingAPI , NotificationAPI
from src.basic import AutoRunTask , CALENDAR
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict(email = 1)
    with AutoRunTask(f'daily update {CALENDAR.update_to()}' , **params) as runner:
        if runner.already_done and runner.source == 'bash': return
        DataAPI.update()
        ModelAPI.update()
        TradingAPI.update()
        NotificationAPI.proceed()
        
if __name__ == '__main__':
    main()
        
    