#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update AI Models
# content: 更新训练所有Registered模型
# email: False
# mode: shell


from src.res.api import ModelAPI
from src.basic import AutoRunTask
from src.app import BackendTaskRecorder , ScriptLock

@BackendTaskRecorder()
@ScriptLock('update_models' , timeout = 10)
def main(**kwargs):
    with AutoRunTask('update_models' , **kwargs) as runner:
        ModelAPI.update_models()
        runner.critical(f'Update models at {runner.update_to} completed')

    return runner

if __name__ == '__main__':
    main()
