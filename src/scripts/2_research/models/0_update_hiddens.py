#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Hiddens Extraction
# content: 更新模型隐变量,用于做其他模型的输入
# email: True
# mode: shell

from src.res.api import ModelAPI
from src.basic import AutoRunTask
from src.app import BackendTaskRecorder , ScriptLock

@BackendTaskRecorder()
@ScriptLock('update_hiddens' , timeout = 10)
def main(**kwargs):
    with AutoRunTask('update_hiddens' , **kwargs) as runner:
        ModelAPI.update_hidden()
        runner.critical(f'Update hiddens at {runner.update_to} completed')

    return runner

if __name__ == '__main__':
    main()
