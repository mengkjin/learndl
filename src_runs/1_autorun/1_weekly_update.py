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
from src_runs.util import argparse_dict

def main():
    params = argparse_dict(email = 1)
    if not MACHINE.server:
        print(f'{MACHINE.name} is not a server, skip weekly update')
        return
    with AutoRunTask(f'weekly update {CALENDAR.update_to()}' , **params) as runner:
        if runner.forfeit_task: return
        ModelAPI.update_models()

if __name__ == '__main__':
    main()