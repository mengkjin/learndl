#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Model Predictions
# content: 更新所有Registered模型的预测结果
# email: False
# mode: shell

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.res.api import ModelAPI
from src.basic import AutoRunTask
from src.app import BackendTaskRecorder , ScriptLock

@BackendTaskRecorder()
@ScriptLock('update_preds' , timeout = 10)
def main(**kwargs):
    with AutoRunTask('update_preds' , **kwargs) as runner:
        ModelAPI.update_preds() 
        runner.critical(f'Update preds at {runner.update_to} completed')

    return runner

if __name__ == '__main__':
    main()
