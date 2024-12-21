#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: 更新模型隐变量,用于做其他模型的输入

import sys , pathlib

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

from src.api import DataAPI , ModelAPI

if __name__ == '__main__':
    DataAPI.prepare_predict_data()
    ModelAPI.update_hidden()
