#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-24
# description: Reconstruct Train Data
# content: 重建历史训练数据, 用于模型从2017年开始训练
# email: False
# close_after_run: False

import sys

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

from src.api import DataAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict()
    with AutoRunTask('reconstruct train data' , **params) as runner:
        DataAPI.reconstruct_train_data(confirm = 1)

if __name__ == '__main__':
    main()