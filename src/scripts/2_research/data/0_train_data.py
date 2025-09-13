#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-24
# description: Reconstruct Train Data
# content: 重建历史训练数据, 用于模型从2017年开始训练
# email: True
# mode: shell

from src.res.api import DataAPI
from src.basic import AutoRunTask
from src.app import BackendTaskRecorder , ScriptLock

@BackendTaskRecorder()
@ScriptLock('reconstruct_train_data' , timeout = 10)
def main(**kwargs):
    with AutoRunTask('reconstruct_train_data' , **kwargs) as runner:
        DataAPI.reconstruct_train_data(confirm = 1)
        runner.critical(f'Reconstruct train data at {runner.update_to} completed')

    return runner

if __name__ == '__main__':
    main()
