#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-24
# description: Edit Train Config
# content: 编辑训练配置文件
# email: False
# close_after_run: True

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

from pathlib import Path
from src_runs.util import BackendTaskManager , edit_file

@BackendTaskManager.manage()
def main(**kwargs):
    edit_path = Path(path).joinpath('configs' , 'train' , 'model.yaml')
    print(f'Editing file : {edit_path}')
    edit_file(edit_path)

if __name__ == '__main__':
    main()
