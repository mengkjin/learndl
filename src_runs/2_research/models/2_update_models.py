#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update AI Models
# content: 更新训练所有Registered模型
# email: False
# close_after_run: False

import sys

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

from src.api import ModelAPI
from src.basic import AutoRunTask
from src_runs.util import BackendTaskManager

@BackendTaskManager()
def main(**kwargs):
    with AutoRunTask('update models' , **kwargs) as runner:
        ModelAPI.update_models()
        runner.critical(f'Update models at {runner.update_to} completed')

    return runner

if __name__ == '__main__':
    main()
