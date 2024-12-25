#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Update Hiddens Extraction
# content: 更新模型隐变量,用于做其他模型的输入
# TODO:若有模型依赖与其他模型的隐变量和结果,则需要给定更新顺序，否则可能出现循环依赖
# email: False
# close_after_run: False

import sys , pathlib

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

from src.api import ModelAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict()
    with AutoRunTask('update hiddens' , **params) as runner:
        ModelAPI.update_hidden()

if __name__ == '__main__':
    main()
