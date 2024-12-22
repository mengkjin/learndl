#! /usr/bin/env python3.10
# coding: utf-8
# author: jinmeng
# date: 2024-11-27
# description: Train Model
# content: 训练某个新模型,模型的参数在configs/train/model.yaml里定义,也可以改变其他configs

import sys , pathlib

paths = [p for p in pathlib.Path(__file__).absolute().parents if p.name == 'learndl']
assert paths , f'learndl path not found , do not know where to find src file : {__file__}'
sys.path.append(str(paths[0]))

from src.api import ModelAPI
from src.basic import AutoRunTask
from src_runs.widget import argparse_dict

def main():
    params = argparse_dict()
    with AutoRunTask('train model' , **params) as runner:
        trainer = ModelAPI.initialize_trainer(stage = 0 , resume = 0 , checkname= 1)
        trainer.go()

if __name__ == '__main__':
    main()
