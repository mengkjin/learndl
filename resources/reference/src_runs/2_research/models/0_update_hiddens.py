#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Hiddens Extraction
# content: 更新模型隐变量,用于做其他模型的输入
# email: False
# mode: shell

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.api import ModelAPI
from src.basic import AutoRunTask
from src_app import BackendTaskRecorder

@BackendTaskRecorder()
def main(**kwargs):
    with AutoRunTask('update hiddens' , **kwargs) as runner:
        ModelAPI.update_hidden()
        runner.critical(f'Update hiddens at {runner.update_to} completed')

    return runner

if __name__ == '__main__':
    main()
