#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-12-24
# description: Edit Train Config
# content: 编辑训练配置文件
# email: False
# close_after_run: True

import sys , pathlib

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

from src_runs._abc import edit_file

def main():
    path = paths[0].joinpath('configs/train/model.yaml')
    print(f'Editing file : {path}')
    edit_file(path)

if __name__ == '__main__':
    main()
