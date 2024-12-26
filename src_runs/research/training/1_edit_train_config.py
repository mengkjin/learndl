#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-24
# description: Edit Train Config
# content: 编辑训练配置文件
# email: False
# close_after_run: True

import sys

assert 'learndl' in __file__ , f'learndl path not found , do not know where to find src file : {__file__}'
path = __file__.removesuffix(__file__.split('learndl')[-1])
sys.path.append(path)

from pathlib import Path
from src_runs._abc import edit_file

def main():
    edit_path = Path(path).joinpath('configs' , 'train' , 'model.yaml')
    print(f'Editing file : {edit_path}')
    edit_file(edit_path)

if __name__ == '__main__':
    main()
