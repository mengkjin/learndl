#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Model Predictions
# content: 更新所有Registered模型的预测结果
# email: False
# close_after_run: False

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.api import ModelAPI
from src.basic import AutoRunTask
from src_runs.util import BackendTaskManager

@BackendTaskManager()
def main(**kwargs):
    with AutoRunTask('update preds' , **kwargs) as runner:
        ModelAPI.update_preds() 
        runner.critical(f'Update preds at {runner.update_to} completed')

    return runner

if __name__ == '__main__':
    main()
