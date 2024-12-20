#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: 每日更新数据模型
import argparse , sys , pathlib , traceback
from datetime import datetime

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

from src.api import DataAPI , ModelAPI
from src.basic import AutoRunTask

def main():
    with AutoRunTask('daily update' , **AutoRunTask.get_args()) as runner:
        DataAPI.update()
        ModelAPI.update()

if __name__ == '__main__':
    main()
        
    