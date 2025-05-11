#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Run Weekly Update
# content: 每周更新模型(只在服务器上)
# email: True
# close_after_run: False

import sys

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

from src.api import ModelAPI
from src.basic import AutoRunTask , CALENDAR
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict(email = 1)
    with AutoRunTask(f'weekly update {CALENDAR.update_to()}' , **params) as runner:
        if runner.already_done and runner.source == 'bash': return
        ModelAPI.update_models()

if __name__ == '__main__':
    main()