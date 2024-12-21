#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: 更新所有模型相关细节,包括模型本身,隐变量,预测结果.
# TODO：若有模型依赖与其他模型的隐变量和结果,则需要给定更新顺序，否则可能出现循环依赖

import sys , pathlib

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

from src.api import DataAPI , ModelAPI

if __name__ == '__main__':
    ModelAPI.update_preds()
