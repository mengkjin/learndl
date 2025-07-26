#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-24
# description: Reconstruct Train Data
# content: 重建历史训练数据, 用于模型从2017年开始训练
# email: False
# close_after_run: False

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.api import DataAPI
from src.basic import AutoRunTask
from src_ui import BackendTaskManager

@BackendTaskManager()
def main(**kwargs):
    with AutoRunTask('reconstruct train data' , **kwargs) as runner:
        DataAPI.reconstruct_train_data(confirm = 1)
        runner.critical(f'Reconstruct train data at {runner.update_to} completed')

    return runner

if __name__ == '__main__':
    main()
