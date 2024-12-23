#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Run Daily Update
# content: 每日更新数据,因子,模型隐变量,模型推理

import sys , pathlib 

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

from src.api import DataAPI , ModelAPI , TradingAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict(email = 1)
    with AutoRunTask('daily update' , **params) as runner:
        DataAPI.update()
        ModelAPI.update()
        TradingAPI.update()
        
if __name__ == '__main__':
    main()
        
    