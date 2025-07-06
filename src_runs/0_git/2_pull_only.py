#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2025-01-02
# description: pull only
# content: 自动拉取最新代码
# email: False
# close_after_run: False

import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

import subprocess 
if __name__ == '__main__':
    subprocess.run("git pull", shell=True, check=True)