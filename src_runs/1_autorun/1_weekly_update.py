#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Run Weekly Update
# content: 每周更新模型(只在服务器上)
# email: True
# close_after_run: False

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from src.api import ModelAPI
from src.basic import MACHINE , AutoRunTask , CALENDAR
from src_ui import BackendTaskRecorder

@BackendTaskRecorder(email = 1)
def main(**kwargs):
    with AutoRunTask(f'weekly update {CALENDAR.update_to()}' , **kwargs) as runner:
        if not MACHINE.server:
            runner.error(f'{MACHINE.name} is not a server, skip weekly update')
        elif runner.forfeit_task:
            runner.error(f'task is forfeit, most likely due to finished autoupdate, skip weekly update')
        else:
            ModelAPI.update_models()
            runner.critical(f'Weekly update of {runner.update_to} completed')

    return runner

if __name__ == '__main__':
    main()